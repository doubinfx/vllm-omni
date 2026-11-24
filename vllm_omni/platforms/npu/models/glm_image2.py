# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A2 kernels for the official GLM-Image2 single-stage pipeline."""

from __future__ import annotations

import logging
from types import MethodType, ModuleType

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

_A2_ROTARY_PATCHED = False
_A2_RMS_NORM_CHUNK = 32


if HAS_TRITON:

    @triton.jit
    def _exact_add_rms_norm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        residual_output_ptr,
        num_cols: tl.constexpr,
        eps: tl.constexpr,
        block_cols: tl.constexpr,
        has_residual: tl.constexpr,
        reduction_groups: tl.constexpr,
    ):
        """Match vLLM's native RMSNorm dtype and rounding boundaries exactly."""
        row = tl.program_id(0)
        cols = tl.arange(0, block_cols)
        mask = cols < num_cols
        x = tl.load(x_ptr + row * num_cols + cols, mask, other=0.0)
        if has_residual:
            residual = tl.load(
                residual_ptr + row * num_cols + cols,
                mask,
                other=0.0,
            )
            value_fp32 = x.to(tl.float32) + residual.to(tl.float32)
            tl.store(
                residual_output_ptr + row * num_cols + cols,
                value_fp32.to(tl.bfloat16),
                mask,
            )
        else:
            value_fp32 = x.to(tl.float32)

        squares = tl.reshape(
            value_fp32 * value_fp32,
            (reduction_groups, block_cols // reduction_groups),
        )
        variance = tl.sum(tl.sum(squares, axis=1), axis=0) / num_cols
        normalized = value_fp32 * tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask, other=0.0)
        # vLLM casts the normalized value back to the weight dtype before the
        # weighted multiply. Preserve that intermediate BF16 rounding.
        tl.store(
            output_ptr + row * num_cols + cols,
            normalized.to(tl.bfloat16) * weight,
            mask,
        )


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


def _pad_query_rotary_tables(
    cos: torch.Tensor,
    sin: torch.Tensor,
    target_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad leading non-rotary pairs with an identity transform."""
    padding = target_dim - cos.shape[-1]
    if padding < 0:
        raise ValueError("rotary table is wider than the query embedding")
    if padding == 0:
        return cos, sin
    return (
        torch.cat((cos.new_ones(*cos.shape[:-1], padding), cos), dim=-1),
        torch.cat((sin.new_zeros(*sin.shape[:-1], padding), sin), dim=-1),
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
    original_query = glm_image2_transformer.apply_query_rotary
    cached_query_source: torch.Tensor | None = None
    cached_query_dim: int | None = None
    cached_query_tables: tuple[torch.Tensor, torch.Tensor] | None = None

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

    def apply_query_rotary(query, rope_offset, cos, sin):
        nonlocal cached_query_source, cached_query_dim, cached_query_tables
        if query.device.type != "npu":
            return original_query(query, rope_offset, cos, sin)
        if cached_query_source is not cos or cached_query_dim != query.shape[-1]:
            cached_query_source = cos
            cached_query_dim = query.shape[-1]
            cached_query_tables = _pad_query_rotary_tables(
                cos,
                sin,
                query.shape[-1],
            )
        assert cached_query_tables is not None
        query_cos, query_sin = cached_query_tables
        return _apply_npu_rotary_interleave(
            query,
            query_cos,
            query_sin,
        )

    glm_image2_transformer.GlmImage2RotaryEmbedding.forward = rotary_forward
    glm_image2_transformer.apply_rotary_interleave = apply_rotary_interleave
    glm_image2_transformer.apply_query_rotary = apply_query_rotary
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


def _apply_exact_a2_rms_norm(
    module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fuse small-row RMSNorm while preserving vLLM's exact BF16 rounding."""
    if (
        residual is None
        and hidden_states.device.type == "npu"
        and hidden_states.dtype == torch.bfloat16
        and module.weight.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and getattr(module, "variance_size_override", None) is None
        and not torch.compiler.is_compiling()
    ):
        return _get_torch_npu().npu_rms_norm(
            hidden_states,
            module.weight,
            module.variance_epsilon,
        )[0]

    if (
        residual is not None
        and hidden_states.shape[0] > 64
        and hidden_states.device.type == "npu"
        and hidden_states.dtype == torch.bfloat16
        and module.weight.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and getattr(module, "variance_size_override", None) is None
        and not torch.compiler.is_compiling()
    ):
        summed = hidden_states.float() + residual.float()
        normalized = F.rms_norm(
            summed,
            (summed.shape[-1],),
            eps=module.variance_epsilon,
        )
        output = normalized.to(torch.bfloat16) * module.weight
        return output.to(torch.bfloat16), summed.to(torch.bfloat16)

    eligible = (
        HAS_TRITON
        and hidden_states.device.type == "npu"
        and hidden_states.dtype == torch.bfloat16
        and module.weight.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and hidden_states.shape[0] <= 64
        and hidden_states.is_contiguous()
        and not torch.compiler.is_compiling()
    )
    if not eligible:
        return module._a2_rms_norm_fallback(hidden_states, residual)

    output = torch.empty_like(hidden_states)
    residual_output = torch.empty_like(hidden_states) if residual is not None else hidden_states
    num_cols = hidden_states.shape[1]
    _exact_add_rms_norm_kernel[(hidden_states.shape[0],)](
        hidden_states,
        residual if residual is not None else hidden_states,
        module.weight,
        output,
        residual_output,
        num_cols,
        module.variance_epsilon,
        triton.next_power_of_2(num_cols),
        residual is not None,
        _A2_RMS_NORM_CHUNK,
        enable_fp_fusion=False,
    )
    if residual is not None:
        return output, residual_output
    return output


def _install_exact_a2_rms_norm_patch(model) -> None:
    from vllm.model_executor.layers.layernorm import RMSNorm

    for module in model.modules():
        if not isinstance(module, RMSNorm):
            continue
        if getattr(module, "_a2_exact_rms_norm", False):
            continue
        module._a2_rms_norm_fallback = module._forward_method
        module._forward_method = MethodType(_apply_exact_a2_rms_norm, module)
        module._a2_exact_rms_norm = True


def _a2_row_parallel_forward(self, input_):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    eligible = (
        input_.device.type == "npu"
        and input_.dtype == torch.bfloat16
        and self.weight.dtype == torch.bfloat16
        and self.weight.ndim == 2
        and input_.is_contiguous()
        and self.input_is_parallel
        and self.reduce_results
        and self.bias is None
        and isinstance(self.quant_method, UnquantizedLinearMethod)
        and not torch.compiler.is_compiling()
    )
    if not eligible:
        return self._a2_mm_all_reduce_fallback(input_)
    output = _get_torch_npu().npu_mm_all_reduce_base(
        input_,
        self.weight.T,
        self._a2_hcom,
    )
    return output, None


def _install_a2_mm_all_reduce_patch(model) -> int:
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.linear import (
        RowParallelLinear,
        UnquantizedLinearMethod,
    )

    tp_group = get_tp_group()
    if tp_group.world_size <= 1:
        return 0
    backend = tp_group.device_group._get_backend(torch.device("npu"))
    hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
    patched = 0
    for module in model.modules():
        if not isinstance(module, RowParallelLinear):
            continue
        if (
            module.weight.dtype != torch.bfloat16
            or not module.input_is_parallel
            or not module.reduce_results
            or module.bias is not None
            or not isinstance(module.quant_method, UnquantizedLinearMethod)
        ):
            continue
        module._a2_mm_all_reduce_fallback = module.forward
        module._a2_hcom = hcom
        module.forward = MethodType(_a2_row_parallel_forward, module)
        patched += 1
    return patched


def _a2_moe_forward(self, hidden_states: torch.Tensor, num_branches: int = 1):
    """Combine routed and shared expert partial outputs in one collective."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if num_branches == 1:
        router_logits, _ = self.gate(hidden_states)
    else:
        # Preserve the official branch-wise routing path. The FP32 router GEMM
        # is not row-invariant in M and may otherwise flip near-tied experts.
        router_logits = torch.cat(
            [self.gate(chunk)[0] for chunk in hidden_states.chunk(num_branches)]
        )

    routed = self.experts(hidden_states=hidden_states, router_logits=router_logits)
    shared_experts = self.shared_experts
    down_proj = shared_experts.down_proj
    gate_up, _ = shared_experts.gate_up_proj(hidden_states)
    shared_input = shared_experts.act_fn(gate_up)
    eligible = (
        routed.device.type == "npu"
        and routed.dtype == torch.bfloat16
        and shared_input.dtype == torch.bfloat16
        and shared_input.is_contiguous()
        and down_proj.weight.dtype == torch.bfloat16
        and down_proj.weight.ndim == 2
        and down_proj.bias is None
        and isinstance(down_proj.quant_method, UnquantizedLinearMethod)
        and routed.shape == (*shared_input.shape[:-1], down_proj.weight.shape[0])
        and not torch.compiler.is_compiling()
    )
    if not eligible:
        # Routed output is intentionally unreduced. Fall back to reducing the
        # shared branch separately so the mathematical result stays correct.
        return routed + shared_experts(hidden_states)
    shared_partial = F.linear(
        shared_input,
        down_proj.weight,
        down_proj.bias,
    )
    return tensor_model_parallel_all_reduce(shared_partial + routed)


def _defer_a2_moe_final_reduce(
    self,
    states: torch.Tensor,
    trunc_size: int | None,
    output_is_reduced: bool | None = None,
):
    """Honor the runner's deferred-reduction mode on the Ascend runner."""
    del self, output_is_reduced
    if trunc_size is not None and trunc_size > 0:
        return states[..., :trunc_size]
    return states


def _install_a2_moe_combined_all_reduce_patch(model) -> int:
    """Defer routed reduction and combine both expert branches before one AR."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_omni.diffusion.models.glm_image2.glm_image2_transformer import (
        GlmImage2MoE,
    )

    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    if tp_group.world_size <= 1:
        return 0
    patched = 0
    for module in model.modules():
        if not isinstance(module, GlmImage2MoE):
            continue
        moe_config = getattr(module.experts, "moe_config", None)
        down_proj = module.shared_experts.down_proj
        if (
            moe_config is None
            or down_proj.weight.dtype != torch.bfloat16
            or down_proj.bias is not None
            or not isinstance(down_proj.quant_method, UnquantizedLinearMethod)
        ):
            continue
        # The runner supports deferring its final reduction for downstream
        # fusion. The Ascend runner currently ignores the static skip flag, so
        # override the per-layer hook as well. Shared-expert down remains an
        # unquantized BF16 RowParallel.
        moe_config.skip_final_all_reduce = True
        module.experts._maybe_reduce_final_output = MethodType(
            _defer_a2_moe_final_reduce,
            module.experts,
        )
        module.forward = MethodType(_a2_moe_forward, module)
        patched += 1
    return patched


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
    _install_exact_a2_rms_norm_patch(model)
    patched_row_parallel = _install_a2_mm_all_reduce_patch(model)
    patched_moe = _install_a2_moe_combined_all_reduce_patch(model)
    if patched_row_parallel:
        logging.getLogger(__name__).info(
            "Patched %d GLM-Image2 A2 row-parallel projections with MM+AllReduce",
            patched_row_parallel,
        )
    if patched_moe:
        logging.getLogger(__name__).info(
            "Patched %d GLM-Image2 A2 MoE layers with combined expert AllReduce",
            patched_moe,
        )
    del od_config
    return True
