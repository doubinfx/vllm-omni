# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline for the native GLM-Image HY/MoE diffusion stage."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.glm_image_moe_transfusion.native_diffusion import (
    NativeGlmImageGenerator,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import (
    DiffusionRequestBatch,
    split_diffusion_output_by_request,
)
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)


def _custom_arg(od_config: OmniDiffusionConfig, name: str) -> str | None:
    args = getattr(od_config, "custom_pipeline_args", None) or {}
    value = args.get(name)
    return os.fspath(value) if value is not None else None


def _setting(od_config: OmniDiffusionConfig, name: str) -> str | None:
    return _custom_arg(od_config, name) or os.environ.get(f"GLM_IMAGE_MOE_{name.upper()}")


def _generated_ids(prompt: Any, sampling_params: Any) -> list[int]:
    extra = sampling_params.extra_args or {}
    value = extra.get("generated_token_ids")
    if value is None and isinstance(prompt, dict):
        value = (prompt.get("extra") or {}).get("generated_token_ids")
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(token) for token in value.reshape(-1).tolist()]
    return [int(token) for token in value]


def _prompt_fields(prompt: Any) -> tuple[str, str | None, dict[str, Any]]:
    if isinstance(prompt, str):
        return prompt, None, {}
    if not isinstance(prompt, dict):
        return str(prompt or ""), None, {}
    media = prompt.get("multi_modal_data") or {}
    if not isinstance(media, dict):
        raise OmniClientError("GLM-Image multi_modal_data must be a mapping")
    return (
        str(prompt.get("prompt") or ""),
        prompt.get("negative_prompt"),
        media,
    )


@dataclass(frozen=True)
class _GlmImageRequestSpec:
    prompt: str
    height: int
    width: int
    num_steps: int
    seed: int
    generated_ids: list[int]
    generator: torch.Generator | None


def _validate_request_spec(
    prompt: Any,
    sampling: Any,
    *,
    dummy: bool = False,
) -> _GlmImageRequestSpec:
    prompt_text, negative_prompt, media = _prompt_fields(prompt)
    if not prompt_text.strip():
        raise OmniClientError("GLM-Image diffusion prompt must not be empty")
    if negative_prompt:
        raise OmniClientError(
            "negative_prompt is not supported: GLM-Image DMD2 serves one positive conditioning branch"
        )
    if isinstance(prompt, dict):
        negative_fields = [
            name
            for name, value in prompt.items()
            if name.startswith("negative_") and value is not None
        ]
        if negative_fields:
            raise OmniClientError(
                f"negative conditioning is not supported: {', '.join(sorted(negative_fields))}"
            )
    if any(value is not None for value in media.values()):
        raise OmniClientError("GLM-Image text-to-image does not accept reference or source images")

    num_outputs = int(getattr(sampling, "num_outputs_per_prompt", 1) or 1)
    if num_outputs != 1:
        raise OmniClientError(
            f"GLM-Image currently produces one image per request, got num_outputs_per_prompt={num_outputs}"
        )
    if not dummy:
        if getattr(sampling, "guidance_scale_provided", False) and sampling.guidance_scale not in (
            None,
            1,
            1.0,
        ):
            raise OmniClientError("GLM-Image DMD2 is CFG-distilled: guidance_scale must be 1 or omitted")
        unsupported = {
            "do_classifier_free_guidance": getattr(sampling, "do_classifier_free_guidance", False),
            "guidance_scale_2_provided": getattr(sampling, "guidance_scale_2_provided", False),
            "true_cfg_scale": getattr(sampling, "true_cfg_scale", None),
            "strength": getattr(sampling, "strength", None),
            "sigmas": getattr(sampling, "sigmas", None),
            "latents": getattr(sampling, "latents", None),
        }
        requested = [
            name
            for name, value in unsupported.items()
            if value is not None and value is not False
        ]
        if requested:
            raise OmniClientError(f"unsupported GLM-Image sampling parameters: {', '.join(requested)}")
    if int(getattr(sampling, "num_frames", 1) or 1) != 1:
        raise OmniClientError("GLM-Image text-to-image produces one frame")
    output_type = getattr(sampling, "output_type", None)
    if output_type not in (None, "pil"):
        raise OmniClientError(f"unsupported GLM-Image output_type: {output_type!r}")

    height = int(sampling.height or 1024)
    width = int(sampling.width or 1024)
    num_steps = int(sampling.num_inference_steps or 4)
    if num_steps <= 0:
        raise OmniClientError(f"num_inference_steps must be positive, got {num_steps}")
    seed = int(sampling.seed if sampling.seed is not None else 0)

    generator = getattr(sampling, "generator", None)
    if isinstance(generator, (list, tuple)):
        if len(generator) != num_outputs:
            raise OmniClientError(
                "GLM-Image per-request generator count must match num_outputs_per_prompt"
            )
        generator = generator[0] if generator else None
    if generator is not None and not isinstance(generator, torch.Generator):
        raise OmniClientError("GLM-Image generator must be a torch.Generator")
    return _GlmImageRequestSpec(
        prompt=prompt_text,
        height=height,
        width=width,
        num_steps=num_steps,
        seed=seed,
        generated_ids=_generated_ids(prompt, sampling),
        generator=generator,
    )


def get_glm_image_moe_transfusion_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config
    processor = VaeImageProcessor(vae_scale_factor=16)

    def post_process(images: torch.Tensor) -> list[PIL.Image.Image]:
        return processor.postprocess(images, output_type="pil")

    return post_process


class GlmImageMoeTransfusionPipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """Translate a diffusion request into the native GLM-Image generator."""

    supports_request_batch = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.generator = NativeGlmImageGenerator(
            checkpoint=_setting(od_config, "checkpoint") or od_config.model,
            stage_config=_setting(od_config, "stage_config") or os.path.join(od_config.model, "stage_config"),
            tokenizer_path=_setting(od_config, "tokenizer_path") or os.path.join(od_config.model, "tokenizer"),
            vae_path=_setting(od_config, "vae_path") or os.path.join(od_config.model, "vae"),
            device=self.device,
            vae_device=_setting(od_config, "vae_device") or "npu:0",
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.generator.model.backbone.dtype

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        if not req.prompts:
            raise OmniClientError("GLM-Image diffusion request is missing a prompt")
        is_dummy = req.is_dummy_run()
        specs = [
            _validate_request_spec(prompt, sampling, dummy=is_dummy)
            for prompt, sampling in zip(req.prompts, req.sampling_params_list, strict=True)
        ]

        if is_dummy:
            # A2 lacks a VAE Conv2D kernel for the tiny 64x64 warmup shape.
            # Model construction and checkpoint loading already completed, so
            # return a shape-only response for the engine's internal probe.
            image = torch.zeros((1, 3, 64, 64), device=self.device, dtype=self.dtype)
            return [
                DiffusionOutput(
                    output=image,
                    stage_durations=getattr(self, "stage_durations", None),
                )
                for _ in req.prompts
            ]

        heights = {spec.height for spec in specs}
        widths = {spec.width for spec in specs}
        step_counts = {spec.num_steps for spec in specs}
        if len(heights) != 1 or len(widths) != 1:
            raise OmniClientError(
                "GLM-Image request batching requires one common height and width; "
                f"got heights={sorted(heights)}, widths={sorted(widths)}"
            )
        if len(step_counts) != 1:
            raise OmniClientError(
                f"GLM-Image request batching requires one num_inference_steps, got {sorted(step_counts)}"
            )
        height = specs[0].height
        width = specs[0].width
        num_steps = specs[0].num_steps
        logger.info(
            "Native GLM-Image batch: requests=%s size=%sx%s steps=%s seeds=%s generated_ids=%s",
            len(specs),
            height,
            width,
            num_steps,
            [spec.seed for spec in specs],
            [len(spec.generated_ids) for spec in specs],
        )
        images = self.generator.generate_batch(
            [spec.prompt for spec in specs],
            height=height,
            width=width,
            num_steps=num_steps,
            seeds=[spec.seed for spec in specs],
            generated_ids=[spec.generated_ids for spec in specs],
            generators=[spec.generator for spec in specs],
        )
        result = DiffusionOutput(
            output=images,
            stage_durations=getattr(self, "stage_durations", None),
        )
        return split_diffusion_output_by_request(result, req, num_outputs_per_prompt=1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """The generator loads the shared checkpoint directly during construction."""
        del weights
        return {name for name, _ in self.named_parameters()}
