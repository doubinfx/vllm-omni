# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline for the native GLM-Image HY/MoE diffusion stage."""

from __future__ import annotations

import os
from collections.abc import Iterable
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
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


def _custom_arg(od_config: OmniDiffusionConfig, name: str) -> str | None:
    args = getattr(od_config, "custom_pipeline_args", None) or {}
    value = args.get(name)
    return os.fspath(value) if value is not None else None


def _setting(od_config: OmniDiffusionConfig, name: str) -> str | None:
    return _custom_arg(od_config, name) or os.environ.get(f"GLM_IMAGE_MOE_{name.upper()}")


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("prompt") or "")
    return str(getattr(prompt, "prompt", "") or "")


def _generated_ids(prompt: Any, sampling_params: OmniDiffusionSamplingParams) -> list[int]:
    extra = sampling_params.extra_args or {}
    value = extra.get("generated_token_ids")
    if value is None and isinstance(prompt, dict):
        value = (prompt.get("extra") or {}).get("generated_token_ids")
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(token) for token in value.reshape(-1).tolist()]
    return [int(token) for token in value]


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
            raise ValueError("GLM-Image diffusion request is missing a prompt")
        prompt_texts = [_prompt_text(prompt) for prompt in req.prompts]
        if any(not prompt_text for prompt_text in prompt_texts):
            raise ValueError("GLM-Image diffusion prompt must not be empty")

        is_dummy = req.is_dummy_run()
        if is_dummy:
            # A2 lacks a VAE Conv2D kernel for the tiny 64x64 warmup shape.
            # Model construction and checkpoint loading already completed, so
            # return a shape-only response for the engine's internal probe.
            return [
                DiffusionOutput(
                    output=torch.zeros((1, 3, 64, 64), device=self.device, dtype=self.dtype),
                    stage_durations=getattr(self, "stage_durations", None),
                )
                for _ in req.prompts
            ]

        common = req.sampling_params_list[0]
        height = common.height or 1024
        width = common.width or 1024
        num_steps = common.num_inference_steps or 4
        for sampling_params in req.sampling_params_list[1:]:
            if (
                (sampling_params.height or 1024) != height
                or (sampling_params.width or 1024) != width
                or (sampling_params.num_inference_steps or 4) != num_steps
            ):
                raise ValueError("GLM-Image request batch requires common size and inference-step settings")

        batch_prompts: list[str] = []
        batch_generated_ids: list[list[int]] = []
        batch_seeds: list[int] = []
        outputs_per_request: list[int] = []
        for prompt, prompt_text, sampling_params in zip(
            req.prompts,
            prompt_texts,
            req.sampling_params_list,
            strict=True,
        ):
            num_outputs = max(1, int(sampling_params.num_outputs_per_prompt or 1))
            seed = sampling_params.seed if sampling_params.seed is not None else 0
            generated_ids = _generated_ids(prompt, sampling_params)
            outputs_per_request.append(num_outputs)
            for output_index in range(num_outputs):
                batch_prompts.append(prompt_text)
                batch_generated_ids.append(generated_ids)
                batch_seeds.append(int(seed) + output_index)

        logger.info(
            "Native GLM-Image batch: requests=%s samples=%s size=%sx%s steps=%s",
            req.num_reqs,
            len(batch_prompts),
            height,
            width,
            num_steps,
        )
        images = self.generator.generate_batch(
            batch_prompts,
            height=height,
            width=width,
            num_steps=num_steps,
            seeds=batch_seeds,
            generated_ids=batch_generated_ids,
        )
        outputs: list[DiffusionOutput] = []
        offset = 0
        for num_outputs in outputs_per_request:
            sample_indices = torch.arange(
                offset,
                offset + num_outputs,
                device=images.device,
            )
            outputs.append(
                DiffusionOutput(
                    output=torch.index_select(images, 0, sample_indices),
                    stage_durations=getattr(self, "stage_durations", None),
                )
            )
            offset += num_outputs
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """The generator loads the shared checkpoint directly during construction."""
        del weights
        return {name for name, _ in self.named_parameters()}
