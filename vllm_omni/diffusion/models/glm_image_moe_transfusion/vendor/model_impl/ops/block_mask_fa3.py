import math

import torch
from transformers import AttentionInterface


try:
    from flash_attn_interface import _flash_attn_backward, _flash_attn_forward

    FLASH_ATTN_3_BLOCK_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_BLOCK_AVAILABLE = False


def block_mask_meta(
    cu_seqlens: torch.IntTensor,
    blk_start: torch.IntTensor,
    blk_end: torch.IntTensor,
    total_len: int,
    check: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 (blk_hi, blk_lo)。两者都按全局 flatten 行号索引、存 item-local 值。

    缓存在 cu_seqlens 对象上（和 prefix_lm_meta 一个套路）：调用方每个 pack 建一次、
    transformer blocks 多层复用不重复构建。
    """
    meta = getattr(cu_seqlens, "_block_mask_fused_meta", None)
    if meta is not None and meta[0] is blk_start and meta[1] is blk_end:
        return meta[2]

    dev = cu_seqlens.device
    bounds = cu_seqlens.tolist()
    items = list(zip(bounds[:-1], bounds[1:]))
    hi = torch.empty(total_len, dtype=torch.int32)
    lo = torch.empty(total_len, dtype=torch.int32)
    for s, e in items:  # 默认：纯 causal
        rng = torch.arange(e - s, dtype=torch.int32)
        hi[s:e] = rng
        lo[s:e] = rng
    for a, b in zip(blk_start.tolist(), blk_end.tolist()):
        s = next(s_ for s_, e_ in items if s_ <= a and b <= e_)
        hi[a:b] = (b - 1) - s  # 整块抬到 block 末列
        lo[a:b] = a - s  # 整块降到 block 首行

    if check:  # block skip 依赖 hi 单调
        for s, e in items:
            seg = hi[s:e]
            assert bool((seg[1:] >= seg[:-1]).all()), f"blk_hi 在 item [{s},{e}) 内非单调"

    out = (hi.to(dev), lo.to(dev))
    try:
        cu_seqlens._block_mask_fused_meta = (blk_start, blk_end, out)
    except Exception:  # pragma: no cover
        pass
    return out


class FusedBlockMaskAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, item_cu, blk_hi, blk_lo, max_item_len, scale):
        out = torch.zeros_like(q)
        o, lse, *_ = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,
            None,
            out,
            item_cu,
            item_cu,
            None,
            None,
            None,
            max_item_len,
            max_item_len,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            scale,
            True,
            (-1, -1),
            0,
            0.0,
            True,
            None,
            1,
            False,  # pack_gqa=False：见下方说明
            0,
            blk_hi,
            blk_lo,
        )
        ctx.save_for_backward(q, k, v, o, lse, item_cu, blk_hi, blk_lo)
        ctx.consts = (max_item_len, scale)
        return o

    @staticmethod
    def backward(ctx, dout):
        q, k, v, o, lse, item_cu, blk_hi, blk_lo = ctx.saved_tensors
        max_item_len, scale = ctx.consts
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        _flash_attn_backward(
            dout.contiguous(),
            q,
            k,
            v,
            o,
            lse,
            item_cu,
            item_cu,
            None,
            None,
            max_item_len,
            max_item_len,
            dq,
            dk,
            dv,
            scale,
            True,
            (-1, -1),
            0.0,
            False,
            0,
            blk_hi,
            blk_lo,
        )
        return dq, dk, dv, None, None, None, None, None


def fused_block_mask_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.IntTensor,
    blk_start: torch.IntTensor,
    blk_end: torch.IntTensor,
    max_seqlen: int,
    scale: float | None = None,
    check_meta: bool = False,
) -> torch.Tensor:
    """q/k/v: [L, H, D] bf16 contiguous, rope 已加过。-> [L, H, D]

    cu_seqlens: [N+1] int32，item 边界前缀和，item_cu[0] = 0
    blk_start : [M]   int32，各双向 block 的起点（全局坐标，须落在某个 item 内）
    blk_end   : [M]   int32，对应终点（开区间）
    max_item_len : int

    block 必须两两不相交且不跨 item 边界。调用方保证（47 层热路径不查）；
    check_meta=True 会额外验一次 hi 的单调性，调试时用。
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    hi, lo = block_mask_meta(cu_seqlens, blk_start, blk_end, q.shape[0], check=check_meta)
    return FusedBlockMaskAttn.apply(q, k, v, cu_seqlens, hi, lo, max_seqlen, scale)


AttentionInterface.register("fa3_block", fused_block_mask_attn)
