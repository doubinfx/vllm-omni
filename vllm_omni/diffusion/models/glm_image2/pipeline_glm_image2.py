# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
GlmImage2Pipeline for vLLM-Omni.

Native port of the diffusers ``GlmImage2Pipeline`` (text-to-image and
reference-image generation for ``zai-org/GLM-Image-2``):

- ``GlmImage2Processor`` (native) builds the packed prompt, 3D mRoPE positions
  and reference ViT patches.
- ``GlmImage2Transformer`` (native) is the shared Transfusion decoder. The
  prompt/reference prefix is computed once per request and CFG branch; each
  denoising step only processes the target latent tokens.
- A FLUX.2 VAE (32 latent channels, 2x2 latent patches, BN-statistics
  normalisation) encodes references and decodes the target.
- Sampling runs on resolution-shifted flow sigmas with plain CFG
  ``v = uncond + guidance_scale * (cond - uncond)``; the conditional and
  unconditional branches can run on separate ranks (CFG parallel). The
  ``ode`` sampler takes Euler steps; ``renoise`` (the DMD2 student recipe)
  predicts ``x0 = x_t - t * v`` and re-noises it to the next sigma.
- ``extra_args["enable_thinking"]`` (text-to-image, CFG 1) first samples a
  chain of thought and a final caption with the decoder's language-model head,
  then generates the image conditioned on them; the caption is returned as
  text metadata.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import DistributedAutoencoderKLFlux2
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.glm_image2.glm_image2_processor import GlmImage2Inputs, GlmImage2Processor
from vllm_omni.diffusion.models.glm_image2.glm_image2_thinking import generate_reasoning, split_reasoning
from vllm_omni.diffusion.models.glm_image2.glm_image2_transformer import (
    GlmImage2ModelConfig,
    GlmImage2PrefixCache,
    GlmImage2TeaCacheState,
    GlmImage2Transformer,
)
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)

_DEFAULT_SAMPLE_SIZE = 1024
_DEFAULT_NUM_INFERENCE_STEPS = 40
_DEFAULT_GUIDANCE_SCALE = 4.0
_SAMPLERS = ("ode", "renoise")
_TRANSFORMER_SUBFOLDER = "vision_language_encoder"
# Checkpoint tensors the pipeline loads (everything but the VQ codebook).
_TRANSFORMER_WEIGHT_PATTERNS = (
    "model.embed_tokens.*",
    "model.layers.*",
    "model.norm.*",
    "model.vq_visual.*",
    "model.vq_projector.pre_pool.*",
    "vq_to_llm_proj.*",
    "cond_vit_aligner.*",
    "lm_head.*",
    "diff_*",
)


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _resolve_model_path(model: str, revision: str | None = None) -> str:
    """Resolve a local model directory or download the weights from the Hub."""
    if os.path.exists(model):
        return model
    return download_weights_from_hf_specific(model, revision, ["*"])


def _normalize_reference_images(raw_image) -> list[PIL.Image.Image]:
    """Normalize reference image input(s) to a flat list of RGB PIL images."""
    if not isinstance(raw_image, (list, tuple)):
        raw_image = [raw_image]
    images: list[PIL.Image.Image] = []
    for item in raw_image:
        if isinstance(item, PIL.Image.Image):
            images.append(item.convert("RGB"))
        elif isinstance(item, str):
            with PIL.Image.open(item) as image:
                images.append(image.convert("RGB"))
        elif isinstance(item, np.ndarray):
            images.append(PIL.Image.fromarray(item).convert("RGB"))
        else:
            raise TypeError(f"Unsupported reference image type: {type(item)}")
    return images


def get_glm_image2_pre_process_func(od_config: OmniDiffusionConfig):
    """Normalize ``multi_modal_data.image`` into ``additional_information.preprocessed_image``.

    Resizing is left to the native processor (smart resize, factor 32, capped
    at 2048x2048), matching the reference pipeline.
    """

    def pre_process_func(request: OmniDiffusionRequest):
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}
        if raw_image is not None:
            prompt["additional_information"]["preprocessed_image"] = _normalize_reference_images(raw_image)
        request.prompt = prompt
        return request

    return pre_process_func


def get_glm_image2_post_process_func(od_config: OmniDiffusionConfig):
    """Convert decoded images in [-1, 1] to PIL; thinking outputs keep their text metadata."""

    def to_pil(images: torch.Tensor) -> list[PIL.Image.Image]:
        if images.device.type == "npu":
            arrays = (
                (images / 2 + 0.5)
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            return [PIL.Image.fromarray(array) for array in arrays]
        arrays = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        return [PIL.Image.fromarray(array) for array in (arrays * 255).round().astype("uint8")]

    def post_process_func(output):
        if isinstance(output, dict):
            return {**output, "payload": {**output["payload"], "image": to_pil(output["payload"]["image"])}}
        return to_pil(output)

    return post_process_func


class GlmImage2Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    SupportImageInput,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """GLM-Image2 text-to-image and reference-image generation (one image per request)."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]

    support_image_input = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model_path = _resolve_model_path(od_config.model, od_config.revision)

        model_index = _read_json(os.path.join(model_path, "model_index.json"))
        scheduler_config = _read_json(os.path.join(model_path, "scheduler", "scheduler_config.json"))
        generation_config = _read_json(os.path.join(model_path, "generation_config.json"))
        self.timeinput_scaler = float(model_index.get("timeinput_scaler", 1000.0))
        self.base_seqlen = int(model_index.get("base_seqlen", scheduler_config.get("base_image_seq_len", 256)))
        self.shift_seqlen = int(model_index.get("shift_seqlen", scheduler_config.get("max_image_seq_len", 4096)))
        self.base_shift = float(model_index.get("base_shift", scheduler_config.get("base_shift", 0.5)))
        self.shift = float(model_index.get("shift", scheduler_config.get("max_shift", 1.15)))
        self.use_fix_shift = bool(model_index.get("use_fix_shift", False))
        self.train_weight = model_index.get("train_weight", "logit_normal")
        if self.train_weight not in ("logit_normal", "uniform"):
            raise ValueError("train_weight must be 'logit_normal' or 'uniform'.")
        self.default_num_inference_steps = int(
            generation_config.get("num_inference_steps", _DEFAULT_NUM_INFERENCE_STEPS)
        )
        self.default_guidance_scale = float(generation_config.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE))
        self.default_sampler = generation_config.get(
            "sampler", "renoise" if scheduler_config.get("stochastic_sampling") else "ode"
        )
        if self.default_sampler not in _SAMPLERS:
            raise ValueError(f"Unsupported GLM-Image2 sampler {self.default_sampler!r}.")

        with open(os.path.join(model_path, _TRANSFORMER_SUBFOLDER, "config.json")) as f:
            model_config = GlmImage2ModelConfig.from_dict(json.load(f))
        self.processor = GlmImage2Processor(os.path.join(model_path, "processor"))

        self.transformer = GlmImage2Transformer(model_config, quant_config=od_config.quantization_config)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=_TRANSFORMER_SUBFOLDER,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
                weight_name_patterns=_TRANSFORMER_WEIGHT_PATTERNS,
            )
        ]

        vae_class = (
            DistributedAutoencoderKLFlux2
            if od_config.parallel_config.vae_parallel_mode == "batch"
            else AutoencoderKLFlux2
        )
        self.vae = vae_class.from_pretrained(
            model_path, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.vae.eval()
        if (
            self.vae.config.latent_channels != model_config.diff_vae_latent_dim
            or tuple(self.vae.config.patch_size) != (model_config.diff_patch_size,) * 2
        ):
            raise ValueError("GLM-Image2 requires a FLUX.2 VAE with 32 latent channels and 2x2 patches.")
        self.latent_channels = model_config.latent_dim

        self._prefix_caches: dict[str, GlmImage2PrefixCache] = {}
        # Per-request TeaCache states (keyed like the prefix caches, or "batched").
        self._teacache_states: dict[str, GlmImage2TeaCacheState] = {}
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["vae.encode", "vae.decode", "_build_prefix_cache", "denoise_step"],
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

    # ============================================================ latents

    def _vae_bn_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.vae.bn.running_mean.reshape(1, -1, 1, 1)
        std = (self.vae.bn.running_var.to(dtype=torch.float32) + self.vae.bn.eps).sqrt().reshape(1, -1, 1, 1)
        return mean, std

    def encode_references(self, inputs: GlmImage2Inputs) -> torch.Tensor | None:
        """VAE-encode reference images into normalized, 2x2-patchified latent tokens."""
        if not inputs.reference_images:
            return None
        mean, std = self._vae_bn_stats()
        tokens = []
        for image in inputs.reference_images:
            pixels = image.unsqueeze(0).to(device=self.device, dtype=self.vae.dtype)
            encoded = self.vae.encode(pixels).latent_dist.mode()
            _, channels, latent_h, latent_w = encoded.shape
            encoded = encoded.reshape(1, channels, latent_h // 2, 2, latent_w // 2, 2)
            encoded = encoded.permute(0, 1, 3, 5, 2, 4).reshape(1, channels * 4, latent_h // 2, latent_w // 2)
            encoded = (encoded - mean.to(encoded)) / std.to(encoded.dtype)
            tokens.append(encoded.flatten(2).transpose(1, 2).squeeze(0))
        return torch.cat(tokens)

    def prepare_noise(self, inputs: GlmImage2Inputs, generator: torch.Generator | None) -> torch.Tensor:
        """Target fp32 Gaussian noise, drawn exactly as the reference pipeline does.

        The reference samples noise over the whole (references + target) span
        and discards the reference positions, then draws one training-weight
        value per image; both are reproduced so a seed maps to the same image.
        """
        lengths = inputs.diff_grid_thw.prod(-1).tolist()
        total, target = sum(lengths), lengths[-1]
        noise = randn_tensor(
            (1, total, self.latent_channels), generator=generator, device=self.device, dtype=torch.float32
        )[0]
        random_device = generator.device if generator is not None else self.device
        if self.train_weight == "logit_normal":
            torch.normal(0.0, 1.0, size=(len(lengths),), generator=generator, device=random_device)
        else:
            torch.rand(len(lengths), generator=generator, device=random_device)
        return noise[total - target :].to(self.vae.dtype)

    def get_sigmas(self, grid: torch.Tensor, num_inference_steps: int) -> list[float]:
        """Resolution-shifted flow sigmas: ``alpha*t / (1 + (alpha-1)*t)`` with ``alpha = e^mu``."""
        timesteps = torch.linspace(1, 0, num_inference_steps + 1, dtype=torch.float32)
        if self.use_fix_shift:
            alpha = torch.tensor(self.shift, dtype=torch.float32)
        else:
            seq_len = float(grid.prod())
            mu = self.base_shift + (self.shift - self.base_shift) * (
                (seq_len - self.base_seqlen) / (self.shift_seqlen - self.base_seqlen)
            )
            alpha = torch.tensor(mu, dtype=torch.float32).exp()
        return (alpha * timesteps / (1 + (alpha - 1) * timesteps)).tolist()

    def decode_latents(self, latents: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Unpack ``[T, 128]`` target latents and decode to an image in [-1, 1]."""
        _, height, width = grid.tolist()
        latents = latents.to(dtype=self.vae.dtype).transpose(0, 1).reshape(1, self.latent_channels, height, width)
        mean, std = self._vae_bn_stats()
        latents = latents * std.to(latents.dtype) + mean.to(latents)
        channels = self.latent_channels // 4
        latents = latents.reshape(1, channels, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(1, channels, height * 2, width * 2)
        return self.vae.decode(latents, return_dict=False)[0].clamp(-1, 1).float()

    # ========================================================= denoising

    def _branch_needed(self, branch: str, do_cfg: bool) -> bool:
        if branch == "positive":
            return (
                not do_cfg
                or get_classifier_free_guidance_world_size() == 1
                or (get_classifier_free_guidance_rank() == 0)
            )
        return do_cfg and (get_classifier_free_guidance_world_size() == 1 or get_classifier_free_guidance_rank() != 0)

    def _build_prefix_cache(
        self, inputs: GlmImage2Inputs, reference_latents: torch.Tensor | None
    ) -> GlmImage2PrefixCache:
        device = self.device
        return self.transformer.build_prefix_cache(
            input_ids=inputs.input_ids.to(device),
            mm_token_type_ids=inputs.mm_token_type_ids.to(device),
            position_ids=inputs.position_ids.to(device),
            pixel_values=inputs.pixel_values.to(device),
            image_grid_thw=inputs.image_grid_thw,
            diff_grid_thw=inputs.diff_grid_thw,
            reference_latents=reference_latents,
            image_block_spans=inputs.image_block_spans(),
            target_start=inputs.target_start,
        )

    def predict_noise(
        self, branch: str, latents: torch.Tensor, timestep: torch.Tensor, force_compute: bool = False
    ) -> torch.Tensor:
        teacache = self._teacache_states.get(branch)
        return self.transformer(latents, timestep, [self._prefix_caches[branch]], teacache, force_compute)[0]

    def _new_teacache_state(self) -> GlmImage2TeaCacheState | None:
        config = getattr(self, "_tea_cache_config", None)
        if config is None:
            return None
        return GlmImage2TeaCacheState(config.rel_l1_thresh, list(config.coefficients))

    def denoise_step(
        self,
        latents: torch.Tensor,
        sigma: float,
        sigma_next: float,
        guidance_scale: float,
        do_cfg: bool,
        sampler: str = "ode",
        generator: torch.Generator | None = None,
        force_compute: bool = False,
    ) -> torch.Tensor:
        """One CFG velocity prediction and sampler update.

        ``force_compute`` keeps TeaCache from reusing the decoder residual on this step.
        """
        timestep = torch.tensor(sigma * self.timeinput_scaler, dtype=torch.float32, device=self.device)
        if do_cfg and len(self._prefix_caches) == 2:
            # Single-rank CFG: both branches share one transformer pass.
            positive, negative = self.transformer(
                latents,
                timestep,
                [self._prefix_caches["positive"], self._prefix_caches["negative"]],
                self._teacache_states.get("batched"),
                force_compute,
            )
            velocity = self.combine_cfg_noise(positive, negative, guidance_scale, cfg_normalize=False)
        else:
            velocity = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_cfg,
                true_cfg_scale=guidance_scale,
                positive_kwargs={
                    "branch": "positive",
                    "latents": latents,
                    "timestep": timestep,
                    "force_compute": force_compute,
                },
                negative_kwargs={
                    "branch": "negative",
                    "latents": latents,
                    "timestep": timestep,
                    "force_compute": force_compute,
                },
                cfg_normalize=False,
            )
        if sampler == "renoise":
            x0 = latents.float() - sigma * velocity.float()
            noise = randn_tensor(latents.shape, generator=generator, device=self.device, dtype=torch.float32)
            return ((1 - sigma_next) * x0 + sigma_next * noise).to(latents.dtype)
        return (latents.float() + (sigma_next - sigma) * velocity.float()).to(latents.dtype)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("GlmImage2Pipeline handles one prompt per request; using the first prompt only.")
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be of type `str` but is {type(prompt)}")

        reference_images = None
        if not isinstance(first_prompt, str):
            reference_images = first_prompt.get("additional_information", {}).get("preprocessed_image")
            if reference_images is None:
                raw_image = (first_prompt.get("multi_modal_data", {}) or {}).get("image")
                if raw_image is not None:
                    reference_images = _normalize_reference_images(raw_image)
        if req.is_dummy_run():
            reference_images = None

        sampling_params = req.sampling_params
        height = sampling_params.height or _DEFAULT_SAMPLE_SIZE
        width = sampling_params.width or _DEFAULT_SAMPLE_SIZE
        num_inference_steps = sampling_params.num_inference_steps or self.default_num_inference_steps
        guidance_scale = (
            sampling_params.guidance_scale if sampling_params.guidance_scale_provided else self.default_guidance_scale
        )
        extra_args = sampling_params.extra_args or {}
        prompt_format = extra_args.get("prompt_format", "plain")
        sampler = extra_args.get("sampler", self.default_sampler)
        if sampler not in _SAMPLERS:
            raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}.")
        if getattr(sampling_params, "num_outputs_per_prompt", 1) > 1:
            logger.warning("GlmImage2Pipeline generates a single image per request; ignoring num_outputs_per_prompt.")
        for name, value in (("height", height), ("width", width), ("num_inference_steps", num_inference_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative.")
        do_cfg = guidance_scale != 1.0
        enable_thinking = bool(extra_args.get("enable_thinking", False))
        if enable_thinking and (reference_images or do_cfg or prompt_format != "plain"):
            raise OmniClientError(
                "GLM-Image2 thinking supports text-to-image requests with guidance_scale=1 and plain prompts."
            )

        generator = None
        if sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)

        # 1. Prompt / reference preparation and the per-branch prefix caches.
        thinking = None
        if enable_thinking:
            seed = sampling_params.seed if sampling_params.seed is not None else torch.Generator().seed()
            prefix_ids = self.processor.thinking_prefix(prompt, height, width)
            reasoning_ids = generate_reasoning(self.transformer, prefix_ids, seed=seed)
            thinking = split_reasoning(reasoning_ids, self.processor.tokenizer)
            logger.info("GLM-Image2 reasoning produced %d tokens; final caption: %s", len(reasoning_ids), thinking[1])
            inputs = {"positive": self.processor.thinking_inputs(prefix_ids + reasoning_ids, height, width)}
        else:
            inputs = {"positive": self.processor(prompt, reference_images, height, width, prompt_format)}
        if do_cfg:
            # The reference keeps the default "plain" format for the empty unconditional caption.
            inputs["negative"] = self.processor("", reference_images, height, width)
        reference_latents = self.encode_references(inputs["positive"])
        self._prefix_caches = {
            branch: self._build_prefix_cache(branch_inputs, reference_latents)
            for branch, branch_inputs in inputs.items()
            if self._branch_needed(branch, do_cfg)
        }

        # TeaCache decisions depend only on the shared latents and timestep, so CFG-parallel
        # ranks skip the same steps.
        branches = ["batched"] if do_cfg and len(self._prefix_caches) == 2 else list(self._prefix_caches)
        self._teacache_states = {
            branch: state for branch in branches if (state := self._new_teacache_state()) is not None
        }

        # 2. Noise and resolution-shifted sigmas.
        target_grid = inputs["positive"].diff_grid_thw[-1]
        latents = self.prepare_noise(inputs["positive"], generator)
        sigmas = self.get_sigmas(target_grid, num_inference_steps)

        # 3. Sampling loop.
        try:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                    last_step = index == num_inference_steps - 1
                    latents = self.denoise_step(
                        latents,
                        sigma,
                        sigma_next,
                        guidance_scale,
                        do_cfg,
                        sampler=sampler,
                        generator=generator,
                        force_compute=last_step,
                    )
                    progress_bar.update()
            if self._teacache_states:
                skipped = max(state.num_skipped for state in self._teacache_states.values())
                logger.debug("GLM-Image2 TeaCache skipped %d of %d steps.", skipped, num_inference_steps)
        finally:
            self._prefix_caches = {}
            self._teacache_states = {}

        # 4. VAE decode.
        image = self.decode_latents(latents, target_grid)
        output = image
        if thinking is not None:
            thought, caption = thinking
            output = {
                "payload": {"image": image},
                "metadata": {"text": {"think_text": thought, "text_output": caption}},
            }
        return DiffusionOutput(
            output=output, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        prefix = "transformer."
        transformer_weights = (
            (name.removeprefix(prefix), tensor) for name, tensor in weights if name.startswith(prefix)
        )
        return {prefix + name for name in self.transformer.load_weights(transformer_weights)}
