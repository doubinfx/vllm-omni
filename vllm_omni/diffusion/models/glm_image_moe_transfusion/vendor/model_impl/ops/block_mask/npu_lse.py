#!/usr/bin/env python3
"""Scheme B: NPU block-mask attention in TWO calls, joined by an LSE merge.

The mask (docs/block_mask_attn_kernel.md) is, per packed item,

    allowed(i, j) = (j <= i)  or  (i, j in the same bidirectional block)

with arbitrarily many blocks anywhere inside an item. The GPU answer is a
patched FA3 that turns causal into a per-query upper bound ``hi[i]``; there is
no Ascend equivalent. ``npu_prefix_lm_attn.block_mask_attn`` (scheme A) cuts the
mask into row groups whose visible keys are ONE contiguous range -- correct, but
it needs 1 + max_blocks_per_item calls and gathers/scatters the block rows.

Scheme B splits by KEY SET instead of by row, so the call count is 2 regardless
of how many blocks there are, and nothing is gathered:

    call A   C = {j <= i}              varlen causal over items, q/k/v in place
    call B   B = {same block, j > i}   the strict-future half of each block

C and B are disjoint and their union is the mask, so the two outputs combine by
the standard log-sum-exp merge of two disjoint key sets.

── call B without a gather, without wasted compute ──────────────────────────

The trick is that call B's kv partition is shifted ONE ROW RIGHT of its q
partition. For a block [a, e) of length L:

    q  rows [a,   e-1)      kv rows [a+1, e)        both length L-1, SQUARE
    local mask  c >= r   <=>   j - (a+1) >= i - a   <=>   j > i          ✓

The two rows that drop out of the shift are provably useless: the block's last
row e-1 has no strict-future key inside the block, and the block's first key a
is never a strict future of any block row. Every remaining q row that is not in
some [a, e-1) gets a segment with ``kv_len = 0`` (returns exactly 0, does no
work), and every kv row not in some [a+1, e) gets a segment with ``q_len = 0``.
Both actual_seq_* stay cumulative partitions of [0, T).

Extra attention pairs = sum over blocks of L(L-1)/2 -- exactly what the patched
FA3 kernel pays. No gather, no scatter, no index_select, no zero buffer.

── the merge, applied to every row ──────────────────────────────────────────

    m  = max(m1, m2);  w1 = exp(m1-m)*s1;  w2 = exp(m2-m)*s2
    out = (w1*o1 + w2*o2) / (w1+w2)

Rows call B does not own come back with s2 = 0, hence w2 = 0, hence out = o1 --
so the merge runs uniformly over all T rows with no index_copy and no where on
the [T, H, D] tensors. (``m2e`` below keeps that exact and NaN-free when m1 is
very negative; see the comment there.)

── backward ─────────────────────────────────────────────────────────────────

The stats are not differentiable outputs, so the merge cannot be backpropped
through. Instead, the ring-attention trick: drive BOTH backward calls with the
MERGED out / softmax_max / softmax_sum, so the p each kernel recomputes is
exp(s - lse_global) and each returns its own share of the global gradient.
dq/dk/dv are full [T, H, D] from both calls, so they combine with a plain add.

Verified that npu_fusion_attention_grad really honours substituted stats rather
than ignoring them (npu_probe_bm_grad.py（源仓的 npu_probe_* 探针，本仓未收）): with merged out + merged
stats dq/dk/dv land at 3.1e-03/1.9e-03/6.4e-05 of the autograd reference, while
own-out or own-stats controls blow up to 1.0e-01 ... 1.1e+00.

Op contract this rests on, all probed on 910B3 (npu_probe_bm_stats.py,
npu_probe_blockmask_b.py, npu_probe_bm_grad.py):
  * softmax_max = true fp32 row max, softmax_sum = sum exp(s - max), 8 identical
    lanes -- BUT under the default ``softmax_layout`` the varlen buffer is a
    CONCATENATION OF PER-SEGMENT (N, s_i, 8) blocks, not the (T, N, 8) its shape
    claims. Two calls with different partitions then disagree about where a
    given (row, head) lives, which is fatal here. ``softmax_layout="TND"``
    forces the global layout for both, forward and backward.
  * a segment with q_len > 0, kv_len = 0 is accepted: out 0, softmax_sum 0.
  * a segment with q_len = 0 is accepted as a kv gap filler.
  * sparse_mode=4 (band) + pre_tockens=0 + next_tockens=BIG + the 2048
    compressed template = local ``c >= r``, per segment, in TND.

    python3 -m model_impl.ops.block_mask.npu_lse
"""

import argparse
import bisect
import time

import torch

from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as _glm_image_base


logger = _glm_image_base.get_logger(__name__)


BIG = 2147483647
_MASK = {}


def _tri(device):
    """The 2048x2048 compressed template shared by sparse_mode 2/3/4."""
    m = _MASK.get(str(device))
    if m is None:
        m = torch.triu(torch.ones(2048, 2048, device=device), diagonal=1).bool()
        _MASK[str(device)] = m
    return m


def _cum(xs):
    out, acc = [], 0
    for x in xs:
        acc += x
        out.append(acc)
    return tuple(out)


# ── partitions ───────────────────────────────────────────────────────────────
def block_mask_meta(item_cu, blk_start, blk_end):
    """-> (cu_causal, cu_b_q, cu_b_kv), host tuples of cumulative lengths.

    Cached on item_cu keyed by the identity of the bound tensors: the model
    builds these once per pack and hands the same objects to every layer.
    Blocks must be disjoint and must not cross an item boundary.
    """
    got = getattr(item_cu, "_npu_bm_lse_meta", None)
    if got is not None and got[0] is blk_start and got[1] is blk_end:
        return got[2]

    items = [int(x) for x in item_cu.cpu().tolist()]
    T = items[-1]
    blocks = sorted(
        (int(a), int(b)) for a, b in zip(blk_start.cpu().tolist(), blk_end.cpu().tolist()) if int(b) - int(a) >= 2
    )  # L < 2 has no strict future
    for a, b in blocks:
        n = bisect.bisect_right(items, a) - 1
        if b > items[n + 1]:
            raise ValueError(f"block [{a}, {b}) crosses the item boundary {items[n + 1]}")

    q_len, kv_len, qpos, kvpos = [], [], 0, 0
    for a, b in blocks:
        if a > qpos:  # rows call B does not own: no keys
            q_len.append(a - qpos)
            kv_len.append(0)
        if a + 1 > kvpos:  # keys no q row of call B reads
            q_len.append(0)
            kv_len.append(a + 1 - kvpos)
        q_len.append(b - a - 1)  # q [a, b-1)  vs  kv [a+1, b)
        kv_len.append(b - a - 1)
        qpos, kvpos = b - 1, b
    if T > qpos:
        q_len.append(T - qpos)
        kv_len.append(0)
    if T > kvpos:
        q_len.append(0)
        kv_len.append(T - kvpos)

    out = (_cum([items[i + 1] - items[i] for i in range(len(items) - 1)]), _cum(q_len), _cum(kv_len))
    assert out[1][-1] == T and out[2][-1] == T, (out[1][-1], out[2][-1], T)
    try:
        item_cu._npu_bm_lse_meta = (blk_start, blk_end, out)
    except Exception:  # pragma: no cover
        pass
    return out


# ── the two calls, the merge, the two backwards ──────────────────────────────
def _lane(x):
    """(T, H, 8) stats, all 8 lanes equal -> (T, H, 1) fp32."""
    return x[..., :1].contiguous()


class _BlockMaskLSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_c, cu_bq, cu_bkv, scale):
        from torch_npu import npu_fusion_attention

        h = q.shape[1]
        mask = _tri(q.device)
        # softmax_layout="TND" is load-bearing, not cosmetic. By default the
        # varlen stats buffer is a CONCATENATION OF PER-SEGMENT (N, s_i, 8)
        # blocks, so two calls with different partitions -- which is exactly
        # what A and B are -- put the same (row, head) at different offsets and
        # an elementwise merge silently mixes unrelated rows. "TND" forces the
        # global (T, N, 8) layout for both, and the merge becomes valid with no
        # reindexing at all. (Measured: default read as global TND is 3.4e-01
        # off; with the flag it is 4.5e-08.)
        o1, m1, s1 = npu_fusion_attention(
            q,
            k,
            v,
            h,
            "TND",
            atten_mask=mask,
            scale=scale,
            sparse_mode=2,
            actual_seq_qlen=cu_c,
            actual_seq_kvlen=cu_c,
            softmax_layout="TND",
        )[:3]
        o2, m2, s2 = npu_fusion_attention(
            q,
            k,
            v,
            h,
            "TND",
            atten_mask=mask,
            scale=scale,
            sparse_mode=4,
            pre_tockens=0,
            next_tockens=BIG,
            actual_seq_qlen=cu_bq,
            actual_seq_kvlen=cu_bkv,
            softmax_layout="TND",
        )[:3]

        # m2e, not m2: a row call B does not own comes back with s2 = 0 AND
        # m2 = 0, and 0 can be far above a genuinely negative m1. Then
        # exp(m2 - m1) overflows to inf and inf * 0 = NaN. Pinning m2e to m1
        # there makes w2 exactly 0 and w1 exactly s1 with no exp of anything.
        m2e = torch.where(s2 > 0, m2, m1)
        m = torch.maximum(m1, m2e)
        w1 = (m1 - m).exp() * s1
        w2 = (m2e - m).exp() * s2
        den = w1 + w2

        # Weights stay fp32: promotion carries the blend into fp32 and rounds
        # once at the end. o1/o2 are already bf16 (the op has no fp32 output),
        # so that rounding is the one this scheme adds over a single pass --
        # 2.08e-03 vs 1.91e-03. No reason to make it worse for ~3% of a layer.
        l1, l2 = _lane(w1), _lane(w2)
        inv = 1.0 / (l1 + l2)
        out = (l1 * inv * o1 + l2 * inv * o2).to(q.dtype)

        ctx.save_for_backward(q, k, v, out, m, den)
        ctx.meta = (cu_c, cu_bq, cu_bkv, scale)
        return out

    @staticmethod
    def backward(ctx, dy):
        from torch_npu import npu_fusion_attention_grad

        q, k, v, out, m, den = ctx.saved_tensors
        cu_c, cu_bq, cu_bkv, scale = ctx.meta
        h = q.shape[1]
        mask = _tri(q.device)
        empty = q.new_empty(0)
        dy = dy.contiguous()

        # Both calls get the MERGED out/stats: that is what makes each kernel
        # recompute exp(s - lse_global) and hand back its share of the global
        # gradient. Verified against autograd in npu_probe_bm_grad.py.
        ga = npu_fusion_attention_grad(
            q,
            k,
            v,
            dy,
            h,
            "TND",
            atten_mask=mask,
            softmax_max=m,
            softmax_sum=den,
            softmax_in=empty,
            attention_in=out,
            scale_value=scale,
            sparse_mode=2,
            actual_seq_qlen=cu_c,
            actual_seq_kvlen=cu_c,
            softmax_layout="TND",
        )[:3]
        gb = npu_fusion_attention_grad(
            q,
            k,
            v,
            dy,
            h,
            "TND",
            atten_mask=mask,
            softmax_max=m,
            softmax_sum=den,
            softmax_in=empty,
            attention_in=out,
            scale_value=scale,
            sparse_mode=4,
            pre_tockens=0,
            next_tockens=BIG,
            actual_seq_qlen=cu_bq,
            actual_seq_kvlen=cu_bkv,
            softmax_layout="TND",
        )[:3]
        return (ga[0] + gb[0], ga[1] + gb[1], ga[2] + gb[2], None, None, None, None)


def block_mask_attn_lse(q, k, v, item_cu, blk_start, blk_end, scale):
    """causal OR same-block, for arbitrary in-item bidirectional blocks.

    q/k/v [T, H, D] -> [T, H, D]. Two npu_fusion_attention calls, whatever the
    number of blocks.
    """
    cu_c, cu_bq, cu_bkv = block_mask_meta(item_cu, blk_start, blk_end)
    return _BlockMaskLSE.apply(q, k, v, cu_c, cu_bq, cu_bkv, scale)


# ────────────────────────── self-test ──────────────────────────────────────
LAYOUT = [
    [("t", 50), ("b", 100), ("t", 30), ("b", 80), ("t", 20)],
    [("t", 17), ("b", 63), ("t", 11), ("b", 129), ("t", 7)],
    [("t", 40), ("b", 200)],
    [("b", 64), ("t", 33)],  # block at the very start of an item
    [("t", 33)],
]  # item with no block at all


def _build(layout, device):
    starts, blocks, text_rows, pos = [], [], [], 0
    for spec in layout:
        starts.append(pos)
        for kind, n in spec:
            (blocks.append((pos, pos + n)) if kind == "b" else text_rows.extend(range(pos, pos + n)))
            pos += n
    T = pos
    ends = starts[1:] + [T]
    mask = torch.zeros(T, T, device=device, dtype=torch.bool)
    for s, e in zip(starts, ends):
        mask[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, device=device, dtype=torch.bool))
    for a, b in blocks:
        mask[a:b, a:b] = True
    i32 = dict(device=device, dtype=torch.int32)
    return (
        T,
        blocks,
        text_rows,
        mask,
        torch.tensor([0] + ends, **i32),
        torch.tensor([a for a, _ in blocks], **i32),
        torch.tensor([b for _, b in blocks], **i32),
    )


def _ref(q, k, v, scale, mask):
    """fp32 reference, differentiable."""
    s = q.float().transpose(0, 1) @ k.float().transpose(0, 1).transpose(-1, -2) * scale
    p = s.masked_fill(~mask[None], -float("inf")).softmax(-1)
    return (p @ v.float().transpose(0, 1)).transpose(0, 1)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _grads(fn, q, k, v, dy):
    qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = fn(qq, kk, vv)
    o.backward(dy if o.dtype == dy.dtype else dy.float())
    return o.detach(), qq.grad, kk.grad, vv.grad


def _accuracy(dev, H=8, D=128):
    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.prefix_lm_npu import block_mask_attn

    scale = D**-0.5
    T, blocks, text_rows, mask, item_cu, bs, be = _build(LAYOUT, dev)
    torch.manual_seed(0)
    q, k, v = (torch.randn(T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
    dy = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)

    ref = _grads(lambda a, b, c: _ref(a, b, c, scale, mask), q, k, v, dy)
    runs = {
        "scheme A (1+max_m calls)": lambda a, b, c: block_mask_attn(a, b, c, item_cu, bs, be, scale),
        "scheme B (2 calls)": lambda a, b, c: block_mask_attn_lse(a, b, c, item_cu, bs, be, scale),
    }
    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.prefix_lm_npu import _blockmask_meta

    ri = torch.tensor([i for a, b in blocks for i in range(a, b)], device=dev)
    ti = torch.tensor(text_rows, device=dev)
    logger.info(f"多块布局: {len(LAYOUT)} items, {len(blocks)} blocks, T={T}, H={H}, D={D}")
    logger.info(
        f"  scheme A: 1 + {len(_blockmask_meta(item_cu, bs, be))} calls，"
        f"scheme B: 2 calls（call B 内 {len(block_mask_meta(item_cu, bs, be)[1])} 段）"
    )
    logger.info(f"\n{'实现':<34}{'out':>11}{'block 行':>11}{'text 行':>11}{'dq':>11}{'dk':>11}{'dv':>11}")
    for name, fn in runs.items():
        got = _grads(fn, q, k, v, dy)
        logger.info(
            f"{name:<34}{_rel(got[0], ref[0]):11.2e}"
            f"{_rel(got[0][ri], ref[0][ri]):11.2e}{_rel(got[0][ti], ref[0][ti]):11.2e}"
            f"{_rel(got[1], ref[1]):11.2e}{_rel(got[2], ref[2]):11.2e}"
            f"{_rel(got[3], ref[3]):11.2e}"
        )
    logger.info("  参照：bf16 输入下这套 fp32 参考的可达下限约 2e-03（一次无掩码调用 vs fp32）")


def _long_block(dev, H=2, D=128, n=3000):
    """The band template is 2048x2048; real image blocks are ~3500 rows."""
    scale = D**-0.5
    T, _, _, mask, item_cu, bs, be = _build([[("t", 5), ("b", n)]], dev)
    torch.manual_seed(0)
    q, k, v = (torch.randn(T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
    dy = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    ref = _grads(lambda a, b, c: _ref(a, b, c, scale, mask), q, k, v, dy)
    got = _grads(lambda a, b, c: block_mask_attn_lse(a, b, c, item_cu, bs, be, scale), q, k, v, dy)
    logger.info(f"\n单块 L={n} > 2048 (压缩模板尺寸), T={T}, H={H}, D={D}")
    logger.info(
        f"  out {_rel(got[0], ref[0]):.2e}   dq {_rel(got[1], ref[1]):.2e}   "
        f"dk {_rel(got[2], ref[2]):.2e}   dv {_rel(got[3], ref[3]):.2e}"
    )


P_LENS = [179, 222, 1, 161, 429, 344]
D_LENS = [3520, 2992, 3600, 2604, 0, 0]


def _perf(dev, H=20, D=256, nblk=1):
    """Real pack shape. nblk = how many bidirectional blocks per item."""
    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.prefix_lm_npu import block_mask_attn

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
        "scheme A": lambda: block_mask_attn(q, k, v, item_cu, bs, be, scale),
        "scheme B": lambda: block_mask_attn_lse(q, k, v, item_cu, bs, be, scale),
    }
    logger.info(f"\n性能  T={T}, H={H}, D={D}, {len(blocks)} blocks ({nblk}/item)")
    logger.info(f"{'实现':<24}{'fwd ms':>10}{'fwd+bwd ms':>13}{'layer':>9}")
    for name, fn in cands.items():
        try:
            tf, tfb = _time(fn, dy)
            logger.info(f"{name:<24}{tf:10.3f}{tfb:13.3f}{tf + tfb:9.2f}")
        except Exception as e:
            logger.info(f"{name:<24}  ✗ {type(e).__name__}: {str(e)[:70]}")


def _causal(q, k, v, item_cu, scale):
    from torch_npu import npu_fusion_attention

    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.prefix_lm_npu import _cu_host

    cu = _cu_host(item_cu)
    return npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        "TND",
        atten_mask=_tri(q.device),
        scale=scale,
        sparse_mode=2,
        actual_seq_qlen=cu,
        actual_seq_kvlen=cu,
    )[0]


def _time(fn, dy, warm=3, it=10):
    for _ in range(warm):
        with torch.no_grad():
            fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        with torch.no_grad():
            fn()
    torch.npu.synchronize()
    tf = (time.perf_counter() - t) / it * 1e3

    def fb():
        fn().backward(dy)

    for _ in range(warm):
        fb()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fb()
    torch.npu.synchronize()
    return tf, (time.perf_counter() - t) / it * 1e3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="1,4", help="comma list: blocks per item in --perf")
    ap.add_argument("--skip-perf", action="store_true")
    a = ap.parse_args()

    import torch_npu  # noqa: F401

    dev = "npu:0"
    _accuracy(dev)
    _long_block(dev)
    if not a.skip_perf:
        for n in (int(s) for s in a.blocks.split(",")):
            _perf(dev, nblk=n)
