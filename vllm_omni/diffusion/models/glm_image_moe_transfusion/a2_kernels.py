# SPDX-License-Identifier: Apache-2.0
"""A2 fused kernel adapters for GLM-Image native diffusion."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch import nn

from vllm_omni.model_executor.models.glm_image_moe_transfusion.ascend_compat import (
    is_ascend_a2,
)

logger = logging.getLogger(__name__)

A2_FUSED_OPS_ENV = "GLM_IMAGE_MOE_A2_FUSED_OPS"
A2_FA_VENDOR_ENV = "GLM_IMAGE_MOE_A2_FA_VENDOR"
A2_FA_VENDOR_FILES = (
    Path("op_api/lib/libcust_opapi.so"),
    Path("op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"),
    Path("op_impl/ai_core/tbe/kernel/ascend910b/flash_attention_score"),
    Path("op_impl/ai_core/tbe/kernel/ascend910b/flash_attention_score_grad"),
)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no", "off"}


def registered_a2_fa_vendor() -> Path | None:
    """Return a registered custom FA package that carries 910B kernels.

    ``ASCEND_CUSTOM_OPP_PATH`` is process-wide and may contain A3-only packages
    from earlier workloads. An explicitly configured GLM-Image vendor is
    validated strictly so an accidental A3 fallback fails during model startup.
    """
    configured = os.environ.get(A2_FA_VENDOR_ENV, "").strip()
    candidates = [
        Path(value)
        for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if value
    ]
    if configured:
        candidates.insert(0, Path(configured))

    for candidate in candidates:
        vendor = candidate.expanduser().resolve()
        if all((vendor / relative).exists() for relative in A2_FA_VENDOR_FILES):
            return vendor

    if configured:
        missing = [
            str(relative)
            for relative in A2_FA_VENDOR_FILES
            if not (Path(configured) / relative).exists()
        ]
        raise RuntimeError(
            f"A2 FlashAttention vendor {configured!r} is missing required "
            f"ascend910b artifacts: {', '.join(missing)}"
        )
    return None


def _get_npu():
    import torch_npu

    return torch_npu


def use_a2_fused_attention() -> bool:
    if not _env_enabled(A2_FUSED_OPS_ENV) or not is_ascend_a2():
        return False
    npu = _get_npu()
    available = all(
        hasattr(npu, op)
        for op in (
            "npu_fusion_attention",
            "npu_rotary_mul",
        )
    )
    vendor = registered_a2_fa_vendor()
    if available and vendor is not None:
        logger.info("Using A2 custom FlashAttentionScore vendor: %s", vendor)
    return available


def use_a2_fused_moe() -> bool:
    if not _env_enabled(A2_FUSED_OPS_ENV) or not is_ascend_a2():
        return False
    npu = _get_npu()
    return all(
        hasattr(npu, op)
        for op in (
            "npu_grouped_matmul",
            "npu_moe_gating_top_k",
            "npu_moe_init_routing_v2",
            "npu_moe_token_unpermute",
            "npu_swiglu",
        )
    )


def use_a2_fused_rms_norm() -> bool:
    if not _env_enabled(A2_FUSED_OPS_ENV) or not is_ascend_a2():
        return False
    return hasattr(_get_npu(), "npu_rms_norm")


def a2_rms_norm_forward(norm: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Use the A2 fused RMSNorm for Transformers' FP32 reference module."""
    return _get_npu().npu_rms_norm(
        hidden_states,
        norm.weight,
        epsilon=float(norm.variance_epsilon),
    )[0]


def a2_moe_router_forward(
    router: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse GLM router sigmoid, grouped top-k, renorm, and scaling."""
    if not router.norm_topk_prob:
        return router(hidden_states)
    hidden_states = hidden_states.view(-1, router.hidden_dim)
    router_logits = torch.nn.functional.linear(
        hidden_states.float(),
        router.weight.float(),
    )
    topk_weights, topk_indices, _ = _get_npu().npu_moe_gating_top_k(
        router_logits,
        router.top_k,
        bias=router.e_score_correction_bias,
        k_group=router.topk_group,
        group_count=router.num_group,
        group_select_mode=1,
        renorm=0,
        norm_type=1,
        out_flag=False,
        routed_scaling_factor=float(router.routed_scaling_factor),
    )
    return router_logits, topk_weights, topk_indices


def a2_moe_init_routing(
    hidden_states: torch.Tensor,
    top_k_indices: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand, sort, and count routed tokens with one A2 operator."""
    expanded, row_ids, expert_counts, _ = _get_npu().npu_moe_init_routing_v2(
        hidden_states,
        top_k_indices.to(torch.int32),
        active_num=-1,
        expert_capacity=-1,
        expert_num=num_experts,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, num_experts],
        row_idx_type=0,
    )
    return expanded, row_ids, expert_counts


def prepare_a2_rotary_tables(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare RoPE tables once for reuse by every attention layer."""
    if not interleaved:
        return cos, sin
    half = cos.shape[-1] // 2
    return (
        cos[..., :half].repeat_interleave(2, dim=-1),
        sin[..., :half].repeat_interleave(2, dim=-1),
    )


def _apply_rotary(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    if hidden_states.shape[0] > 1:
        # aclnnRotaryPositionEmbeddingV2 on A2 rejects batched 4D inputs.
        return torch.cat(
            [
                _apply_rotary(
                    hidden_states[index : index + 1],
                    cos[index : index + 1],
                    sin[index : index + 1],
                    interleaved=interleaved,
                )
                for index in range(hidden_states.shape[0])
            ],
            dim=0,
        )
    npu = _get_npu()
    output_dtype = hidden_states.dtype
    cos = cos.to(hidden_states.device)
    sin = sin.to(hidden_states.device)
    rotary_dtype = cos.dtype
    return npu.npu_rotary_mul(
        hidden_states.to(rotary_dtype),
        cos.unsqueeze(2),
        sin.unsqueeze(2),
        rotary_mode="interleave" if interleaved else "half",
    ).to(output_dtype)


def a2_attention_forward(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, None]:
    """Run the GLM MLA attention math with A2 fused RoPE and FlashAttention."""
    batch_size, seq_length = hidden_states.shape[:-1]
    q_states = attention.q_b_proj(attention.q_a_layernorm(attention.q_a_proj(hidden_states)))
    q_states = q_states.view(batch_size, seq_length, attention.num_heads, attention.qk_head_dim)
    q_pass, q_rot = torch.split(q_states, [attention.qk_nope_head_dim, attention.qk_rope_head_dim], dim=-1)

    compressed_kv = attention.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_rot = torch.split(compressed_kv, [attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1)
    kv_nope = attention.kv_a_layernorm(kv_nope)
    expanded_kv = attention.kv_b_proj(kv_nope).view(
        batch_size,
        seq_length,
        attention.num_heads,
        attention.qk_nope_head_dim + attention.v_head_dim,
    )
    k_pass, value_states = torch.split(
        expanded_kv,
        [attention.qk_nope_head_dim, attention.v_head_dim],
        dim=-1,
    )
    k_rot = k_rot.view(batch_size, seq_length, 1, attention.qk_rope_head_dim)

    cos, sin = position_embeddings
    q_rot = _apply_rotary(q_rot, cos, sin, interleaved=attention.config.rope_interleave)
    k_rot = _apply_rotary(k_rot, cos, sin, interleaved=attention.config.rope_interleave)
    k_rot = k_rot.expand(-1, -1, attention.num_heads, -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1).contiguous()
    key_states = torch.cat((k_pass, k_rot), dim=-1).contiguous()
    attention_output = _get_npu().npu_fusion_attention(
        query_states,
        key_states,
        value_states.contiguous(),
        head_num=attention.num_heads,
        input_layout="BSND",
        atten_mask=attention_mask.to(query_states.device),
        scale=float(attention.scaling),
        keep_prob=1.0,
        sparse_mode=0,
    )[0]
    attention_output = attention_output.reshape(batch_size, seq_length, -1).contiguous()
    return attention.o_proj(attention_output), None


def a2_moe_forward(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Route sparse experts through Ascend grouped GEMM and token permutation."""
    npu = _get_npu()
    residuals = hidden_states
    original_shape = hidden_states.shape
    _, top_k_weights, top_k_indices = a2_moe_router_forward(mlp.gate, hidden_states)
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    experts = mlp.experts

    permuted, row_ids, tokens_per_expert = a2_moe_init_routing(
        flattened,
        top_k_indices,
        num_experts=experts.num_experts,
    )
    gate_up = npu.npu_grouped_matmul(
        [permuted],
        [experts.gate_up_proj.transpose(1, 2)],
        bias=None,
        group_list=tokens_per_expert,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    activated = npu.npu_swiglu(gate_up, dim=-1)
    expert_output = npu.npu_grouped_matmul(
        [activated],
        [experts.down_proj.transpose(1, 2)],
        bias=None,
        group_list=tokens_per_expert,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    routed = npu.npu_moe_token_unpermute(expert_output, row_ids, probs=top_k_weights)
    routed = routed.view(original_shape)
    return routed + mlp.shared_experts(residuals)


__all__ = [
    "A2_FUSED_OPS_ENV",
    "A2_FA_VENDOR_ENV",
    "a2_attention_forward",
    "a2_moe_forward",
    "a2_moe_init_routing",
    "a2_moe_router_forward",
    "a2_rms_norm_forward",
    "prepare_a2_rotary_tables",
    "registered_a2_fa_vendor",
    "use_a2_fused_attention",
    "use_a2_fused_moe",
    "use_a2_fused_rms_norm",
]
