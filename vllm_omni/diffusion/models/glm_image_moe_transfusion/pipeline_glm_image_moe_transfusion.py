"""vLLM-Omni pipeline for the GLM-Image HY/MoE student checkpoint.

The model implementation is vendored under ``vendor/hygen`` so vLLM-Omni owns
the model code and execution path. A worker mode remains available only as a
parity/debug fallback.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import threading
import urllib.request
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
_VENDOR_ROOT = _MODEL_DIR / "vendor"
_VENDORED_CONFIGS = _VENDOR_ROOT / "configs"


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
    """Run the GLM-Image HY/MoE model inside the vLLM-Omni worker process."""

    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        custom_args = od_config.custom_pipeline_args or {}
        self.custom_args = custom_args
        self.mode = str(custom_args.get("mode", "native"))
        self.prefix = prefix
        self.device = torch.device(str(custom_args.get("worker_device", "cpu")))

        # Worker mode is retained for parity/debug only.
        self.endpoint = str(
            custom_args.get("hygen_endpoint")
            or os.environ.get("HYGEN_WORKER_ENDPOINT", "http://127.0.0.1:18080")
        ).rstrip("/")
        self.timeout = float(custom_args.get("hygen_timeout", 1800))
        self.default_sampler = str(custom_args.get("sampler", "ode"))
        self.default_seed = int(custom_args.get("seed", 1234))

        self.spec = str(custom_args.get("spec", "dmd2_lyzsft10000_s2000"))
        self.ckpt = str(
            custom_args.get(
                "ckpt",
                "/mnt/share/d00911077/glm-image/g/dmd2_lyzsft10000_s2000",
            )
        )
        raw_devices = custom_args.get("devices", os.environ.get("HYGEN_DEVICES", "4,5,6,7"))
        if isinstance(raw_devices, str):
            self.devices = [int(item) for item in raw_devices.split(",") if item.strip()]
        else:
            self.devices = [int(item) for item in raw_devices]

        self._executor: ThreadPoolExecutor | None = None
        self._inferencer: Any | None = None
        self._load_thread_id: int | None = None
        self._load_lock = threading.Lock()

        if self.mode == "native":
            self._prepare_native_environment(custom_args)
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="glm-image-native",
            )
            # Loading and generation must happen on the same thread because
            # torch_npu keeps its device context thread-local.
            self._inferencer = self._executor.submit(self._load_native).result()
            self._load_thread_id = threading.get_ident()
        elif self.mode != "worker":
            raise ValueError(f"unsupported GLM-Image MoE pipeline mode: {self.mode}")

    def _prepare_native_environment(self, custom_args: dict[str, Any]) -> None:
        vendor_root = str(_VENDOR_ROOT)
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)

        os.environ.setdefault("HYGEN_WEIGHTS_ROOT", str(Path(self.ckpt).parent))
        os.environ.setdefault(
            "HYGEN_TOKENIZER_PATH",
            str(custom_args.get("tokenizer_path", "/mnt/share/d00911077/glm-image/assets/glm_image_tokenizer")),
        )
        os.environ.setdefault(
            "HYGEN_VAE_PATH",
            str(
                custom_args.get(
                    "vae_path",
                    "/mnt/share/d00911077/glm-image/assets/mnt/cogview_data_cloud/zwd/ckpt/flux2_vae_hf",
                )
            ),
        )
        os.environ.setdefault("HYGEN_CONFIGS_ROOT", str(_VENDORED_CONFIGS))
        os.environ.setdefault("HYGEN_DEVICE_TYPE", "npu")

    def _load_native(self) -> Any:
        from hygen import Inferencer

        logger.info("loading vendored hygen model: spec=%s devices=%s ckpt=%s", self.spec, self.devices, self.ckpt)
        return Inferencer.from_spec(self.spec, devices=self.devices, ckpt=self.ckpt)

    def _request_worker_generation(
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

    def _generate_native(
        self,
        prompt: str,
        height: int,
        width: int,
        seed: int,
        num_steps: int | None,
        cfg_scale: float | None,
        sampler: str,
    ) -> Image.Image:
        if self._inferencer is None:
            raise RuntimeError("native hygen inferencer is not initialized")
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

        if self.mode == "native":
            assert self._executor is not None
            image = self._executor.submit(
                self._generate_native,
                prompt,
                height,
                width,
                seed,
                num_steps,
                cfg_scale,
                sampler,
            ).result()
        else:
            body = self._request_worker_generation(
                prompt=prompt,
                height=height,
                width=width,
                seed=seed,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
            )
            image = Image.open(io.BytesIO(base64.b64decode(body["image_base64"]))).convert("RGB")

        tensor = _image_to_tensor(image)
        return DiffusionOutput(
            output=tensor,
            custom_output={
                "width": image.width,
                "height": image.height,
                "mode": self.mode,
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
        """The vendored inferencer owns the checkpoint; loader sees no local params."""
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

