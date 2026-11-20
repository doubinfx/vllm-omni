"""vLLM-Omni compatibility pipeline for the GLM-Image HY/MoE student weights.

The model implementation currently lives in the
``hygen_release-main`` dependency tree, which requires transformers 4.56.
vLLM-Omni 0.32 uses transformers 5.19, so this pipeline keeps the model in a
long-lived local worker process and exposes it as a normal diffusion pipeline.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.request
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)


class GlmImageMoeTransfusionPipeline(nn.Module, SupportsComponentDiscovery):
    """Serve the GLM-Image HY/MoE checkpoint through a persistent worker."""

    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        custom_args = od_config.custom_pipeline_args or {}
        self.endpoint = str(
            custom_args.get("hygen_endpoint")
            or os.environ.get("HYGEN_WORKER_ENDPOINT", "http://127.0.0.1:18080")
        ).rstrip("/")
        self.timeout = float(custom_args.get("hygen_timeout", 1800))
        self.default_sampler = str(custom_args.get("sampler", "ode"))
        self.default_seed = int(custom_args.get("seed", 1234))
        self.device = torch.device(str(custom_args.get("worker_device", "cpu")))
        self.prefix = prefix

    def _request_generation(
        self,
        *,
        prompt: str,
        height: int,
        width: int,
        seed: int,
        num_steps: int | None,
        cfg_scale: float | None,
        sampler: str,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "seed": seed,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "think": False,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        if body.get("error"):
            raise RuntimeError(body["error"])
        return body

    def _generate_one(self, request: Any) -> DiffusionOutput:
        prompt_value = request.prompt
        if isinstance(prompt_value, str):
            prompt = prompt_value
        else:
            prompt = str(prompt_value.get("prompt") or "")
        if not prompt:
            return DiffusionOutput(error="prompt must not be empty")

        params = request.sampling_params
        height = int(params.height or 1024)
        width = int(params.width or height)
        seed = int(params.seed if params.seed is not None else self.default_seed)
        num_steps = int(params.num_inference_steps) if params.num_inference_steps is not None else None
        cfg_scale = (
            float(params.guidance_scale)
            if getattr(params, "guidance_scale_provided", False)
            else None
        )

        body = self._request_generation(
            prompt=prompt,
            height=height,
            width=width,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            sampler=self.default_sampler,
        )
        image = Image.open(io.BytesIO(base64.b64decode(body["image_base64"]))).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
        logger.info(
            "hygen tensor shape=%s min=%.4f max=%.4f mean=%.4f",
            tuple(tensor.shape),
            float(tensor.min()),
            float(tensor.max()),
            float(tensor.mean()),
        )
        return DiffusionOutput(
            output=tensor,
            custom_output={
                "width": image.width,
                "height": image.height,
                "meta": body.get("meta") or {},
            },
        )

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        if req.is_dummy_run():
            return [
                DiffusionOutput(
                    output=torch.zeros((1, 3, 64, 64), dtype=torch.float32),
                    custom_output={"dummy": True},
                )
                for _ in req.requests
            ]
        return [self._generate_one(request) for request in req.requests]

    def load_weights(self, weights: Any) -> set[str]:
        """Weights live in the external hygen worker, not in this module."""
        del weights
        return set()


def get_glm_image_moe_transfusion_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config

    def post_process_func(images: torch.Tensor) -> list[Image.Image]:
        if isinstance(images, torch.Tensor):
            logger.info(
                "postprocess tensor shape=%s min=%.4f max=%.4f mean=%.4f",
                tuple(images.shape),
                float(images.min()),
                float(images.max()),
                float(images.mean()),
            )
        if isinstance(images, list):
            return images
        tensor = images.detach().float().cpu()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.clamp(-1.0, 1.0)
        tensor = (tensor + 1.0) * 0.5
        result: list[Image.Image] = []
        for item in tensor:
            array = (item.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            result.append(Image.fromarray(array))
        return result

    return post_process_func
