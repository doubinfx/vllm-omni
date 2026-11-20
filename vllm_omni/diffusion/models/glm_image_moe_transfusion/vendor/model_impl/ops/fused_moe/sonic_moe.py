# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SonicMoE adapter for VeOmni MoE experts.

This module deliberately wraps only the expert segment. Router logits, sigmoid,
score-correction bias, top-k selection, statistics and bias balancing stay in the
model code and keep their original fp32 semantics.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache, partial

import torch

from ... import base as _glm_image_base
from ...base import is_torch_npu_available


logger = _glm_image_base.get_logger(__name__)


def _sonic_import_error_message(exc: BaseException | None = None) -> str:
    extra = f" Original error: {exc!r}" if exc is not None else ""
    return (
        "moe_implementation='sonic' requires the `sonic-moe` extra. "
        "Install with `uv sync --extra gpu --extra sonic-moe` and run on NVIDIA Hopper/Blackwell. "
        "If libcuda reports < 12090 on a CUDA 12.9 image, prepend "
        "`/usr/local/cuda-12.9/compat/lib` to LD_LIBRARY_PATH."
        f"{extra}"
    )


@lru_cache(maxsize=1)
def assert_sonic_moe_available() -> None:
    if is_torch_npu_available():
        raise RuntimeError(
            "moe_implementation='sonic' is GPU-only; NPU should continue to use moe_implementation='fused'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("moe_implementation='sonic' requires CUDA to be available.")
    capability = torch.cuda.get_device_capability()
    if capability < (9, 0):
        raise RuntimeError(
            f"moe_implementation='sonic' requires NVIDIA Hopper/Blackwell (SM >= 90), got {capability}."
        )

    try:
        import cutlass  # noqa: F401
        import quack.cache_utils
        import quack.gemm_interface  # noqa: F401
        import sonicmoe  # noqa: F401
        from sonicmoe.functional import moe_general_routing_inputs  # noqa: F401
    except Exception as exc:
        raise RuntimeError(_sonic_import_error_message(exc)) from exc

    # CUTLASS 4.4.2 SM90 kernels need the CUDA 12.9 driver API. On machines with
    # older kernel drivers this is normally provided by the CUDA forward-compat
    # libcuda from /usr/local/cuda-12.9/compat/lib.
    try:
        driver_version = ctypes.c_int()
        ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(driver_version))
    except Exception as exc:
        raise RuntimeError(_sonic_import_error_message(exc)) from exc
    if driver_version.value < 12090:
        raise RuntimeError(
            "moe_implementation='sonic' requires cuDriverGetVersion >= 12090, "
            f"got {driver_version.value}. Set "
            "LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}."
        )

    # Make QuACK's cache persistent and include sonicmoe sources in its cache key
    # when the package is installed editable/from git cache.
    os.environ.setdefault("QUACK_CACHE_DIR", os.path.expanduser("~/.cache/quack"))
    try:
        import pathlib

        import quack.cache_utils
        import sonicmoe

        sonic_root = str(pathlib.Path(sonicmoe.__file__).resolve().parent)
        if sonic_root not in quack.cache_utils.EXTRA_SOURCE_DIRS:
            quack.cache_utils.EXTRA_SOURCE_DIRS.append(sonic_root)
    except Exception:
        logger.warning_once("Failed to register sonicmoe source directory into QuACK cache key.", exc_info=True)


@lru_cache(maxsize=1)
def _configure_quack_autotune() -> bool:
    """Disable QuACK autotune by default for dynamic token counts."""
    import quack.gemm_interface as quack_gemm
    import sonicmoe.functional.backward as sonic_backward
    import sonicmoe.functional.forward as sonic_forward

    tuned = os.getenv("VEOMNI_SONIC_AUTOTUNE", "0") == "1"
    if not tuned:
        sonic_forward.gemm = partial(quack_gemm.gemm, tuned=False)
        sonic_forward.gemm_gated = partial(quack_gemm.gemm_gated, tuned=False)
        sonic_backward.gemm = partial(quack_gemm.gemm, tuned=False)
        sonic_backward.gemm_dgated = partial(quack_gemm.gemm_dgated, tuned=False)
    logger.info_rank0(f"[sonic-moe] QuACK autotune {'enabled' if tuned else 'disabled'} (tuned={tuned})")
    return tuned


@lru_cache(maxsize=1)
def _get_sonic_symbols():
    assert_sonic_moe_available()
    _configure_quack_autotune()
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs

    return moe_general_routing_inputs, ActivationType


def sonic_moe_forward(
    *,
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    training: bool,
) -> torch.Tensor:
    """Run SonicMoE expert kernels with precomputed VeOmni router outputs.

    Args:
        hidden_states: [T, H], bf16/fp16.
        gate_up_weight: [E, 2I, H], concatenated [gate; up] expert weights.
        down_weight: [E, H, I].
        top_k_index: [T, K] selected experts from VeOmni fp32 router.
        top_k_weights: [T, K] routing weights. Kept fp32 by default.
    """
    moe_general_routing_inputs, ActivationType = _get_sonic_symbols()

    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise AssertionError(
            f"hidden_states dtype must be bfloat16 or float16 for SonicMoE, got {hidden_states.dtype}"
        )
    if gate_up_weight.dtype != hidden_states.dtype or down_weight.dtype != hidden_states.dtype:
        raise AssertionError(
            "SonicMoE expert weights must have the same dtype as hidden_states: "
            f"hidden={hidden_states.dtype}, gate_up={gate_up_weight.dtype}, down={down_weight.dtype}"
        )
    if not hidden_states.is_cuda:
        raise AssertionError("SonicMoE requires CUDA tensors")

    top_k_index = top_k_index.to(torch.int32)
    router_scores = top_k_weights.reshape(-1).contiguous()
    token_indices = torch.arange(
        hidden_states.shape[0], device=hidden_states.device, dtype=torch.int32
    ).repeat_interleave(top_k_index.shape[1])
    expert_indices = top_k_index.reshape(-1).contiguous()

    out, _expert_frequency = moe_general_routing_inputs(
        hidden_states.contiguous(),
        router_scores,
        token_indices,
        expert_indices,
        gate_up_weight.permute(1, 2, 0),
        None,
        down_weight.permute(1, 2, 0),
        None,
        num_experts,
        torch.cuda.current_stream().cuda_stream,
        ActivationType.SWIGLU,
        is_inference_mode_enabled=not training,
        concat_layout=True,
    )
    return out


def warmup_sonic_moe() -> None:
    """Tiny fwd+bwd preflight used by launch scripts/tests."""
    assert_sonic_moe_available()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    T, H, I, E, K = 8, 128, 128, 4, 2
    x = torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
    gate_up = torch.randn(E, 2 * I, H, device=device, dtype=dtype, requires_grad=True) * 0.02
    gate_up.retain_grad()
    down = torch.randn(E, H, I, device=device, dtype=dtype, requires_grad=True) * 0.02
    down.retain_grad()
    topk = torch.randint(0, E, (T, K), device=device, dtype=torch.int64)
    weights = torch.rand(T, K, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    out = sonic_moe_forward(
        hidden_states=x,
        gate_up_weight=gate_up,
        down_weight=down,
        top_k_index=topk,
        top_k_weights=weights,
        num_experts=E,
        training=True,
    )
    out.float().sum().backward()
    torch.cuda.synchronize()
    logger.info_rank0("[sonic-moe] warmup passed")
