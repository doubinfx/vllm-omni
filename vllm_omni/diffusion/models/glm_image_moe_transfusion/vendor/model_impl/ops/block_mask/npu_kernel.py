#!/usr/bin/env python3
"""Scheme C: NPU block-mask attention in ONE call, on a patched FlashAttentionScore.

The mask (docs/block_mask_attn_kernel.md) is, per packed item,

    allowed(i, j) = (j <= i)  or  (i, j in the same bidirectional block)

Scheme A cuts it into row groups (1 + max_blocks calls), scheme B cuts it by key
set (2 calls + an LSE merge). Both pay their split in fixed per-call overhead --
~12 ms/layer for B -- which under gradient checkpointing is spent three times
per step across 47 layers.

This one does not split. In TND varlen the key index is item-local, so the lower
bound is already implied and the whole mask collapses to ONE inclusive upper
bound per query row:

    hi[i] = e - 1   if i lies in a bidirectional block [a, e)
    hi[i] = i       otherwise

exact because [0, i] u [a, e) == [0, e-1] whenever a <= i < e. The grad kernel
iterates key-outer, so it needs the inverse bound as well:

    lo[j] = a       if j lies in [a, e)          ( = min{ i : hi[i] >= j } )
    lo[j] = j       otherwise

Both are non-decreasing within an item, so one lookup bounds a whole tile.

── how the bounds reach the kernel ──────────────────────────────────────────

``npu_fusion_attention``'s signature is fixed and there is no free tensor slot,
but there is a free INT ARRAY slot: ``prefix``, declared ``.ValueDepend(OPTIONAL)``
and therefore both readable on the host during tiling and handed to the kernel as
a GM pointer. It is only ever consumed by sparse_mode 5/6, which var-len rejects.
So both bounds ride there, packed into one int64 per query row:

    code[i] = (lo[i] << 32) | hi[i]          item-local indices, T < 2**31

and the patched tiling recognises the combination ``sparse_mode == 0`` +
``layout == TND`` + ``len(prefix) == T`` as a request for the internal
BLOCK_MASK mode. It has to be recognised by SIGNATURE rather than by a new
sparse_mode number because torch_npu's op-plugin rejects any sparse_mode above 8
before the tiling ever runs ("must be in range of [0,5) or (5,8]"), and
op-plugin is compiled into libtorch_npu.so, outside the CANN vendor package.

── what the patch actually changes ──────────────────────────────────────────

Only the SCHEDULE: the forward's s2 loop range and the grad's cube-block
selection. Correctness comes from an explicit block-diagonal [T, T] atten_mask,
which the forward already supported (ATTEN_1_1_1_T_T) and which the grad now
supports too (a new ATTEN_MASK_SHAPE_TYPE_TT). Mask traffic is limited to the
tiles the schedule actually visits, so it costs the same order as the causal
floor's own compressed template reads.

Requires the patched vendor package built from ops-transformer v9.1.0. Without it
the call fails loudly (161001, "unsupported atten_mask shape"): stock tiling does
not know the [T, maxItemLen] mask shape. Note that a [T, T] mask WOULD have been
accepted, and then ``prefix`` would be ignored and the result silently causal.
See docs/npu_block_mask_kernel_impl.md.

    python3 -m model_impl.ops.block_mask.npu_kernel
"""

import argparse
import bisect

import torch

from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as _glm_image_base


logger = _glm_image_base.get_logger(__name__)

from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.npu_lse import (  # noqa: E402
    LAYOUT,
    _build,
    _causal,
    _grads,
    _ref,
    _rel,
    _time,
)


LO_SHIFT = 32
HI_FIELD = (1 << 32) - 1


def block_mask_meta_kernel(item_cu, blk_start, blk_end):
    """-> (cu_host, codes, mask). Cached on item_cu, keyed by the bound tensors.

    ``codes`` is the packed (lo << 32) | hi bound per query row, item-local, as a
    host tuple -- it becomes the ``prefix`` int array. ``mask`` is [T, max_len]
    uint8, 1 == masked out: the row index is global but the column index stays
    item-local, so each item only occupies its own rows instead of a [T, T]
    block-diagonal whose off-diagonal blocks are never read anyway.
    """
    got = getattr(item_cu, "_npu_bm_kernel_meta", None)
    if got is not None and got[0] is blk_start and got[1] is blk_end:
        return got[2]

    items = [int(x) for x in item_cu.cpu().tolist()]
    T = items[-1]
    blocks = sorted(
        (int(a), int(b)) for a, b in zip(blk_start.cpu().tolist(), blk_end.cpu().tolist()) if int(b) > int(a)
    )
    for a, b in blocks:
        n = bisect.bisect_right(items, a) - 1
        if b > items[n + 1]:
            raise ValueError(f"block [{a}, {b}) crosses the item boundary {items[n + 1]}")

    dev = item_cu.device
    hi = torch.arange(T, device=dev, dtype=torch.int64)
    lo = hi.clone()
    for a, b in blocks:
        hi[a:b] = b - 1
        lo[a:b] = a
    base = torch.tensor(
        [items[n] for n in range(len(items) - 1) for _ in range(items[n + 1] - items[n])],
        device=dev,
        dtype=torch.int64,
    )
    hi -= base  # item-local, as the tiling checks
    lo -= base

    max_len = max(items[n + 1] - items[n] for n in range(len(items) - 1))
    mask = torch.ones(T, max_len, device=dev, dtype=torch.uint8)
    cols = torch.arange(max_len, device=dev, dtype=torch.int64)
    for n in range(len(items) - 1):
        s, e = items[n], items[n + 1]
        mask[s:e] = (cols[None, :] > hi[s:e, None]).to(torch.uint8)

    out = (tuple(items[1:]), tuple(((lo << LO_SHIFT) | hi).cpu().tolist()), mask)
    try:
        item_cu._npu_bm_kernel_meta = (blk_start, blk_end, out)
    except Exception:  # pragma: no cover
        pass
    return out


class _BlockMaskKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, codes, mask, scale):
        from torch_npu import npu_fusion_attention

        o, m, s = npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            "TND",
            atten_mask=mask,
            scale=scale,
            sparse_mode=0,
            prefix=codes,
            actual_seq_qlen=cu,
            actual_seq_kvlen=cu,
        )[:3]
        ctx.save_for_backward(q, k, v, o, m, s, mask)
        ctx.meta = (cu, codes, scale)
        return o

    @staticmethod
    def backward(ctx, dy):
        from torch_npu import npu_fusion_attention_grad

        q, k, v, o, m, s, mask = ctx.saved_tensors
        before_attention_backward = getattr(ctx, '_hy106_before_attention_backward', None)
        if before_attention_backward is not None:
            before_attention_backward()
        cu, codes, scale = ctx.meta
        g = npu_fusion_attention_grad(
            q,
            k,
            v,
            dy.contiguous(),
            q.shape[1],
            "TND",
            atten_mask=mask,
            softmax_max=m.contiguous(),
            softmax_sum=s.contiguous(),
            softmax_in=q.new_empty(0),
            attention_in=o,
            scale_value=scale,
            sparse_mode=0,
            prefix=codes,
            actual_seq_qlen=cu,
            actual_seq_kvlen=cu,
        )[:3]
        return g[0], g[1], g[2], None, None, None, None


def block_mask_attn_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.IntTensor,
    blk_start: torch.IntTensor,
    blk_end: torch.IntTensor,
    scale: float | None = None,
    max_seqlen: int | None = None,  # MARK: 跟 fa 侧保持一致，实际上没用的
):
    """causal OR same-block, for arbitrary in-item bidirectional blocks.

    q/k/v [T, H, D] -> [T, H, D]. One npu_fusion_attention call.
    """
    cu, codes, mask = block_mask_meta_kernel(cu_seqlens, blk_start, blk_end)
    return _BlockMaskKernel.apply(q, k, v, cu, codes, mask, scale)


if __name__ == "__main__":
    # ────────────────────────── self-test ──────────────────────────────────────
    def _accuracy(dev, H=8, D=128):
        scale = D**-0.5
        T, blocks, text_rows, mask, item_cu, bs, be = _build(LAYOUT, dev)
        torch.manual_seed(0)
        q, k, v = (torch.randn(T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        dy = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)

        ref = _grads(lambda a, b, c: _ref(a, b, c, scale, mask), q, k, v, dy)
        ri = torch.tensor([i for a, b in blocks for i in range(a, b)], device=dev)
        ti = torch.tensor(text_rows, device=dev)
        logger.info(f"多块布局: {len(LAYOUT)} items, {len(blocks)} blocks, T={T}, H={H}, D={D}")
        logger.info(f"\n{'实现':<34}{'out':>11}{'block 行':>11}{'text 行':>11}{'dq':>11}{'dk':>11}{'dv':>11}")
        got = _grads(lambda a, b, c: block_mask_attn_kernel(a, b, c, item_cu, bs, be, scale), q, k, v, dy)
        logger.info(
            f"{'scheme C (1 call)':<34}{_rel(got[0], ref[0]):11.2e}"
            f"{_rel(got[0][ri], ref[0][ri]):11.2e}{_rel(got[0][ti], ref[0][ti]):11.2e}"
            f"{_rel(got[1], ref[1]):11.2e}{_rel(got[2], ref[2]):11.2e}"
            f"{_rel(got[3], ref[3]):11.2e}"
        )
        logger.info("  参照：bf16 输入下这套 fp32 参考的可达下限约 2e-03（一次无掩码调用 vs fp32）")

    def _long_block_c(dev, H=2, D=128, n=3000):
        scale = D**-0.5
        T, _, _, mask, item_cu, bs, be = _build([[("t", 5), ("b", n)]], dev)
        torch.manual_seed(0)
        q, k, v = (torch.randn(T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        dy = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        ref = _grads(lambda a, b, c: _ref(a, b, c, scale, mask), q, k, v, dy)
        got = _grads(lambda a, b, c: block_mask_attn_kernel(a, b, c, item_cu, bs, be, scale), q, k, v, dy)
        logger.info(f"\n单块 L={n} > 2048 (压缩模板尺寸), T={T}, H={H}, D={D}")
        logger.info(
            f"  out {_rel(got[0], ref[0]):.2e}   dq {_rel(got[1], ref[1]):.2e}   "
            f"dk {_rel(got[2], ref[2]):.2e}   dv {_rel(got[3], ref[3]):.2e}"
        )

    def _perf_c(dev, H=20, D=256, nblk=1):
        from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.npu_lse import D_LENS, P_LENS, block_mask_attn_lse

        scale = D**-0.5
        layout = []
        for p, d in zip(P_LENS, D_LENS):
            spec = [("t", p)]
            if d:
                q_, r_ = divmod(d, nblk)
                for i in range(nblk):
                    spec.append(("b", q_ + (r_ if i == nblk - 1 else 0)))
                    if i != nblk - 1:
                        spec.append(("t", 1))
            layout.append(spec)
        T, blocks, _, _, item_cu, bs, be = _build(layout, dev)
        torch.manual_seed(0)
        q, k, v = (torch.randn(T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        dy = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        cands = {
            "causal only (下限参照)": lambda: _causal(q, k, v, item_cu, scale),
            "scheme B": lambda: block_mask_attn_lse(q, k, v, item_cu, bs, be, scale),
            "scheme C": lambda: block_mask_attn_kernel(q, k, v, item_cu, bs, be, scale),
        }
        logger.info(f"\n性能  T={T}, H={H}, D={D}, {len(blocks)} blocks ({nblk}/item)")
        logger.info(f"{'实现':<24}{'fwd ms':>10}{'fwd+bwd ms':>13}{'layer':>9}")
        for name, fn in cands.items():
            try:
                tf, tfb = _time(fn, dy)
                logger.info(f"{name:<24}{tf:10.3f}{tfb:13.3f}{tf + tfb:9.2f}")
            except Exception as e:
                logger.info(f"{name:<24}  ✗ {type(e).__name__}: {str(e)[:70]}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="1,4", help="comma list: blocks per item in --perf")
    ap.add_argument("--skip-perf", action="store_true")
    a = ap.parse_args()

    import torch_npu  # noqa: F401

    dev = "npu:0"
    _accuracy(dev)
    _long_block_c(dev)
    if not a.skip_perf:
        for n in (int(s) for s in a.blocks.split(",")):
            _perf_c(dev, nblk=n)
