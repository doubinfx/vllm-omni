"""Native vLLM-Omni pipeline for the GLM-Image HY/MoE student checkpoint."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent
_VENDORED_CONFIGS = _MODEL_DIR / "vendor" / "configs"


def _prompt_text(prompt_value: Any) -> str:
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, dict):
        return str(prompt_value.get("prompt") or "")
    return ""


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


class GlmImageMoeTransfusionPipeline(nn.Module, SupportsComponentDiscovery):
    """Run the vendored GLM-Image HY/MoE model inside vLLM-Omni."""

    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        custom_args = od_config.custom_pipeline_args or {}
        self.custom_args = custom_args
        self.prefix = prefix
        self.device = torch.device(str(custom_args.get("worker_device", "cpu")))

        self.spec = str(custom_args.get("spec", "dmd2_lyzsft10000_s2000"))
        self.ckpt = str(
            custom_args.get(
                "ckpt",
                "/mnt/share/d00911077/glm-image/g/dmd2_lyzsft10000_s2000",
            )
        )
        raw_devices = custom_args.get("devices", os.environ.get("VLLM_OMNI_GLM_IMAGE_DEVICES", "0,1,2,3"))
        if isinstance(raw_devices, str):
            self.devices = [int(item) for item in raw_devices.split(",") if item.strip()]
        else:
            self.devices = [int(item) for item in raw_devices]
        self.default_sampler = str(custom_args.get("sampler", "ode"))
        self.default_seed = int(custom_args.get("seed", 1234))

        self._prepare_environment(custom_args)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="glm-image-model",
        )
        # Model loading and generation must stay on one thread because torch_npu
        # keeps its device context thread-local.
        self._inferencer = self._executor.submit(self._load_model).result()

    def _prepare_environment(self, custom_args: dict[str, Any]) -> None:
        os.environ.setdefault(
            "VLLM_OMNI_GLM_IMAGE_WEIGHTS_ROOT",
            str(Path(self.ckpt).parent),
        )
        os.environ.setdefault(
            "VLLM_OMNI_GLM_IMAGE_TOKENIZER_PATH",
            str(custom_args.get("tokenizer_path", "/mnt/share/d00911077/glm-image/assets/glm_image_tokenizer")),
        )
        os.environ.setdefault(
            "VLLM_OMNI_GLM_IMAGE_VAE_PATH",
            str(
                custom_args.get(
                    "vae_path",
                    "/mnt/share/d00911077/glm-image/assets/mnt/cogview_data_cloud/zwd/ckpt/flux2_vae_hf",
                )
            ),
        )
        os.environ.setdefault("VLLM_OMNI_GLM_IMAGE_CONFIGS_ROOT", str(_VENDORED_CONFIGS))
        os.environ.setdefault("VLLM_OMNI_GLM_IMAGE_DEVICE_TYPE", "npu")

    def _load_model(self) -> Any:
        from .vendor.model_impl import Inferencer

        logger.info(
            "loading GLM-Image model: spec=%s devices=%s ckpt=%s",
            self.spec,
            self.devices,
            self.ckpt,
        )
        return Inferencer.from_spec(self.spec, devices=self.devices, ckpt=self.ckpt)

    def _generate_image(
        self,
        prompt: str,
        height: int,
        width: int,
        seed: int,
        num_steps: int | None,
        cfg_scale: float | None,
        sampler: str,
    ) -> Image.Image:
        result = self._inferencer.generate(
            prompt,
            think=False,
            size=(height, width),
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
        )
        return result.image

    def _generate_one(self, request: Any) -> DiffusionOutput:
        prompt = _prompt_text(request.prompt)
        if not prompt:
            return DiffusionOutput(error="prompt must not be empty")

        params = request.sampling_params
        height = int(params.height or 1024)
        width = int(params.width or height)
        seed = int(params.seed if params.seed is not None else self.default_seed)
        num_steps = int(params.num_inference_steps) if params.num_inference_steps is not None else None
        cfg_scale = float(params.guidance_scale) if getattr(params, "guidance_scale_provided", False) else None
        sampler = str(self.custom_args.get("sampler") or params.extra_args.get("sampler") or self.default_sampler)

        image = self._executor.submit(
            self._generate_image,
            prompt,
            height,
            width,
            seed,
            num_steps,
            cfg_scale,
            sampler,
        ).result()
        return DiffusionOutput(
            output=_image_to_tensor(image),
            custom_output={
                "width": image.width,
                "height": image.height,
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
        """The internal model loader owns the checkpoint; no local params."""
        del weights
        return set()


def get_glm_image_moe_transfusion_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config

    def post_process_func(images: torch.Tensor) -> list[Image.Image]:
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

