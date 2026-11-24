# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused GLM-Image2 denoising-step kernels that keep eager rounding bit for bit.

The interleaved RoPE is evaluated like the eager expression
``(even * cos - odd * sin, odd * cos + even * sin)``: bf16 inputs are widened
to fp32, every multiply and add is rounded separately (FMA contraction is
disabled) and the result is rounded once to bf16.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


def fused_rope_kv_supported(query: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and query.device.type == "cuda"
        and torch.version.hip is None
        and query.dtype == torch.bfloat16
        and not torch.compiler.is_compiling()
    )


if HAS_TRITON:

    @triton.jit
    def _rope_pair(even, odd, cos, sin):
        even = even.to(tl.float32)
        odd = odd.to(tl.float32)
        return (even * cos - odd * sin).to(tl.bfloat16), (odd * cos + even * sin).to(tl.bfloat16)

    @triton.jit
    def _query_rope_kernel(
        query_ptr,
        cos_ptr,
        sin_ptr,
        num_rows,
        num_heads: tl.constexpr,
        row_stride: tl.constexpr,
        rope_offset: tl.constexpr,
        half_rope: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)[:, None]
        pairs = tl.arange(0, half_rope)[None, :]
        mask = rows < num_rows
        tokens = rows // num_heads
        base = query_ptr + rows * row_stride + rope_offset + 2 * pairs
        cos = tl.load(cos_ptr + tokens * half_rope + pairs, mask)
        sin = tl.load(sin_ptr + tokens * half_rope + pairs, mask)
        even, odd = _rope_pair(tl.load(base, mask), tl.load(base + 1, mask), cos, sin)
        tl.store(base, even, mask)
        tl.store(base + 1, odd, mask)

    @triton.jit
    def _stage_kv_kernel(
        key_ptr,
        value_ptr,
        prefix_key_ptr,
        prefix_value_ptr,
        kv_ptr,
        k_rot_ptr,
        cos_ptr,
        sin_ptr,
        prefix_len,
        num_rows,
        kv_token_stride,
        k_rot_token_stride,
        num_heads: tl.constexpr,
        nope_dim: tl.constexpr,
        nope_block: tl.constexpr,
        half_rope: tl.constexpr,
        v_dim: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Fill ``key/value[: P + T]`` rows: prefix rows from the cache, target rows from this step."""
        head_dim: tl.constexpr = nope_dim + 2 * half_rope
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)[:, None]
        valid = rows < num_rows
        token = rows // num_heads
        head = rows % num_heads
        is_prefix = token < prefix_len
        target = token - prefix_len
        from_prefix = valid & is_prefix
        from_step = valid & (token >= prefix_len)

        nope = tl.arange(0, nope_block)[None, :]
        in_nope = nope < nope_dim
        step_kv = kv_ptr + target * kv_token_stride + head * (nope_dim + v_dim)
        k_nope = tl.where(
            is_prefix,
            tl.load(prefix_key_ptr + rows * head_dim + nope, from_prefix & in_nope),
            tl.load(step_kv + nope, from_step & in_nope),
        )
        tl.store(key_ptr + rows * head_dim + nope, k_nope, valid & in_nope)

        pairs = tl.arange(0, half_rope)[None, :]
        prefix_rot = prefix_key_ptr + rows * head_dim + nope_dim + 2 * pairs
        step_rot = k_rot_ptr + target * k_rot_token_stride + 2 * pairs
        cos = tl.load(cos_ptr + target * half_rope + pairs, from_step)
        sin = tl.load(sin_ptr + target * half_rope + pairs, from_step)
        rot_even, rot_odd = _rope_pair(tl.load(step_rot, from_step), tl.load(step_rot + 1, from_step), cos, sin)
        rot_even = tl.where(is_prefix, tl.load(prefix_rot, from_prefix), rot_even)
        rot_odd = tl.where(is_prefix, tl.load(prefix_rot + 1, from_prefix), rot_odd)
        key_rot = key_ptr + rows * head_dim + nope_dim + 2 * pairs
        tl.store(key_rot, rot_even, valid)
        tl.store(key_rot + 1, rot_odd, valid)

        v = tl.arange(0, v_dim)[None, :]
        value = tl.where(
            is_prefix,
            tl.load(prefix_value_ptr + rows * v_dim + v, from_prefix),
            tl.load(step_kv + nope_dim + v, from_step),
        )
        tl.store(value_ptr + rows * v_dim + v, value, valid)


def query_rope_(query: torch.Tensor, rope_offset: int, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """In place: rotate ``query[..., rope_offset:]`` of a contiguous ``[L, H, D]`` bf16 tensor."""
    assert query.is_contiguous()
    # The prefix pass produces column-major cos/sin (transposed position ids).
    cos, sin = cos.contiguous(), sin.contiguous()
    num_tokens, num_heads, head_dim = query.shape
    half_rope = cos.shape[-1]
    num_rows = num_tokens * num_heads
    block_rows = 64
    _query_rope_kernel[(triton.cdiv(num_rows, block_rows),)](
        query,
        cos,
        sin,
        num_rows,
        num_heads,
        head_dim,
        rope_offset,
        half_rope,
        block_rows,
        enable_fp_fusion=False,
    )


def stage_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    kv: torch.Tensor,
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """Write ``key = [prefix_key; (kv_nope, rope(k_rot))]`` and ``value = [prefix_value; kv_v]``.

    ``key/value``: contiguous ``[P + T, H, D]`` outputs; ``kv``: ``[T, H, nope + v]``
    rows of ``kv_b_proj``; ``k_rot``: ``[T, rope]`` (row-strided) before RoPE.
    """
    assert key.is_contiguous() and value.is_contiguous()
    assert prefix_key.is_contiguous() and prefix_value.is_contiguous()
    assert kv.stride(-1) == 1 and kv.stride(1) == kv.shape[-1] and k_rot.stride(-1) == 1
    cos, sin = cos.contiguous(), sin.contiguous()
    prefix_len, num_heads, head_dim = prefix_key.shape
    num_tokens = kv.shape[0]
    half_rope = cos.shape[-1]
    nope_dim = head_dim - 2 * half_rope
    num_rows = (prefix_len + num_tokens) * num_heads
    block_rows = 8
    _stage_kv_kernel[(triton.cdiv(num_rows, block_rows),)](
        key,
        value,
        prefix_key,
        prefix_value,
        kv,
        k_rot,
        cos,
        sin,
        prefix_len,
        num_rows,
        kv.stride(0),
        k_rot.stride(0),
        num_heads,
        nope_dim,
        triton.next_power_of_2(nope_dim),
        half_rope,
        value.shape[-1],
        block_rows,
        enable_fp_fusion=False,
    )
