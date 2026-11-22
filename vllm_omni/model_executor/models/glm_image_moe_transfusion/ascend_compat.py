# SPDX-License-Identifier: Apache-2.0
"""Scoped Ascend A2 compatibility for native GLM-Image model paths."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch


def _get_torch_npu() -> Any:
    import torch_npu

    return torch_npu


def is_ascend_a2() -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return False
    device_name = npu.get_device_name(npu.current_device()).upper()
    return "910B" in device_name


def configure_a2_ar_model(model: torch.nn.Module) -> None:
    """Apply the A2-only MLA compatibility required by the shared checkpoint."""
    if not is_ascend_a2():
        return

    for module in model.modules():
        impl: Any = getattr(module, "impl", None)
        if callable(getattr(impl, "supports_prolog_v3_writer", None)):
            impl.supports_prolog_v3_writer = MethodType(lambda _self: False, impl)

    for module in model.modules():
        cos_sin_cache = getattr(module, "cos_sin_cache", None)
        mrope_section = getattr(module, "mrope_section", None)
        if cos_sin_cache is None or not mrope_section:
            continue
        cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
        if not module.is_neox_style:
            # GLM-Image uses adjacent rotary pairs while npu_interleave_rope
            # pair-transposes D before RotateHalf.
            cos_cache = torch.cat((cos_cache, cos_cache), dim=-1)
            sin_cache = torch.cat((sin_cache, sin_cache), dim=-1)
        from vllm_ascend.ops.rotary_embedding import _record_cos_and_sin_cache

        _record_cos_and_sin_cache(cos_cache, sin_cache)
        return


def configure_a2_diffusion() -> None:
    """Disable the A3-only JIT descriptor path before native diffusion starts."""
    npu = getattr(torch, "npu", None)
    if npu is None or not is_ascend_a2():
        return
    npu.set_compile_mode(jit_compile=False)


@contextmanager
def standard_conv_internal_format() -> Iterator[None]:
    """Disable NZ allocation only for the A2 Flux2 Conv2D VAE decode."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available() or not is_ascend_a2():
        yield
        return

    torch_npu = _get_torch_npu()
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    was_enabled = previous == b"enable"
    npu.config.allow_internal_format = False
    try:
        yield
    finally:
        npu.config.allow_internal_format = was_enabled


__all__ = [
    "configure_a2_ar_model",
    "configure_a2_diffusion",
    "is_ascend_a2",
    "standard_conv_internal_format",
]
