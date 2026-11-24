# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A2 kernels for the official GLM-Image2 single-stage pipeline."""

from __future__ import annotations

from types import ModuleType

import torch

_A2_ROTARY_PATCHED = False


def _get_torch_npu() -> ModuleType:
    import torch_npu

    return torch_npu


def _is_a2() -> bool:
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    return get_ascend_device_type() is AscendDeviceType.A2


def _repeat_rotary_tables(
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand adjacent rotary pairs for torch_npu's interleaved input layout."""
    return (
        cos.repeat_interleave(2, dim=-1),
        sin.repeat_interleave(2, dim=-1),
    )


def _apply_npu_rotary_interleave(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved rotary embedding with one A2 fused operator."""
    output = _get_torch_npu().npu_rotary_mul(
        hidden_states.float().unsqueeze(0),
        cos.float().unsqueeze(0).unsqueeze(2),
        sin.float().unsqueeze(0).unsqueeze(2),
        rotary_mode="interleave",
    )
    return output.squeeze(0).to(hidden_states.dtype)


def _install_a2_rotary_patches() -> None:
    global _A2_ROTARY_PATCHED
    if _A2_ROTARY_PATCHED:
        return

    from vllm_omni.diffusion.models.glm_image2 import glm_image2_transformer

    original_rotary = glm_image2_transformer.GlmImage2RotaryEmbedding.forward
    original_apply = glm_image2_transformer.apply_rotary_interleave

    def rotary_forward(self, position_ids):
        cos, sin = original_rotary(self, position_ids)
        if position_ids.device.type == "npu":
            return _repeat_rotary_tables(cos, sin)
        return cos, sin

    def apply_rotary_interleave(hidden_states, cos, sin):
        if hidden_states.device.type != "npu":
            return original_apply(hidden_states, cos, sin)
        if cos.shape[-1] != hidden_states.shape[-1]:
            cos, sin = _repeat_rotary_tables(cos, sin)
        return _apply_npu_rotary_interleave(hidden_states, cos, sin)

    glm_image2_transformer.GlmImage2RotaryEmbedding.forward = rotary_forward
    glm_image2_transformer.apply_rotary_interleave = apply_rotary_interleave
    _A2_ROTARY_PATCHED = True


def _forward_a2_dense_attention(
    attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
) -> torch.Tensor:
    """Run GLM-Image2 dense attention through native A2 FlashAttention."""
    from vllm_omni.diffusion.attention.backends.sdpa import (
        _maybe_reshape_attn_mask,
    )

    attention_mask = attn_metadata.attn_mask if attn_metadata else None
    if attention_mask is not None:
        attention_mask = _maybe_reshape_attn_mask(
            query,
            key,
            attention_mask,
            mask_mode="full_qk",
        ).to(device=query.device, dtype=torch.bool)
        attention_mask = (~attention_mask).contiguous()
        sparse_mode = 1
    else:
        sparse_mode = 0
    return _get_torch_npu().npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        head_num=query.shape[2],
        input_layout="BSND",
        atten_mask=attention_mask,
        scale=float(attention.softmax_scale),
        keep_prob=1.0,
        sparse_mode=sparse_mode,
    )[0]


def _install_a2_attention_patch() -> None:
    from vllm_omni.diffusion.attention.backends.flash_attn import (
        FlashAttentionImpl,
    )

    if getattr(FlashAttentionImpl.forward_fa_npu, "_glm_image2_a2", False):
        return
    original = FlashAttentionImpl.forward_fa_npu

    def forward_fa_npu(self, query, key, value, attn_metadata=None):
        extra = attn_metadata.extra if attn_metadata is not None else {}
        if (
            not _is_a2()
            or self.causal
            or query.shape[2] != key.shape[2]
            or (attn_metadata is not None and attn_metadata.full_attn_spans is not None)
            or extra.get("npu_attn_varlen", False)
        ):
            return original(self, query, key, value, attn_metadata)
        return _forward_a2_dense_attention(
            self,
            query,
            key,
            value,
            attn_metadata,
        )

    forward_fa_npu._glm_image2_a2 = True
    FlashAttentionImpl.forward_fa_npu = forward_fa_npu


def apply_glm_image2_npu_patches(model, od_config) -> bool:
    """Install A2 execution paths without changing official model code."""
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    from vllm_omni.diffusion.models.glm_image2.pipeline_glm_image2 import (
        GlmImage2Pipeline,
    )

    if not isinstance(model, GlmImage2Pipeline):
        return False
    if get_ascend_device_type() is not AscendDeviceType.A2:
        return False
    _install_a2_rotary_patches()
    _install_a2_attention_patch()
    del od_config
    return True
