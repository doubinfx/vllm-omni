"""【自动生成，勿手改】几个小模块的合集 —— 逐字节搬自 veomni，见 model_impl/tools/gen_extras.py。

为什么合成一个文件：transformers 的习惯是一个模型几个大文件，而不是十来个
几十行的小文件。合并只做"import 提顶去重 + 原样拼接"，**没动任何一行实现**。

为什么这些模块必须搬（而不是像训练路径那样占位）：它们是 `modeling_hy.py` 里
**函数内 lazy import** 的目标，其中 `transfusion_cp_shard` 和
`record_router_stats` 在 forward 里是**无条件调用**（前者单进程返回 None，
后者靠内部 `_stats_enabled` 早退）⇒ 用占位会直接炸在 eval 上。

重新生成（必须在 gen_* 之后）：bash model_impl/tools/regen.sh
"""

from __future__ import annotations
import torch
import math
import torch.distributed as dist
from typing import Any, Iterator, List, Optional, Sequence
from .base import get_parallel_state
from collections import deque
from collections import Counter
import os


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/models/transfusion/conditioning.py`。
# ──────────────────────────────────────────────────────────────────────────

def reference_grid_ratio(vit_stride, vae_stride):
    """VAE cells per merged ViT cell; separate from ViT's own merge size."""
    vit_stride, vae_stride = int(vit_stride), int(vae_stride)
    if min(vit_stride, vae_stride) < 1 or vit_stride % vae_stride:
        raise ValueError(f"ViT token stride {vit_stride} must be divisible by VAE token stride {vae_stride}")
    return vit_stride // vae_stride
def image_position_ids(grid, anchor=0, *, spatial_grid=None, temporal_offset=0, device=None):
    """Place image cells in a common spatial frame without changing token order."""
    t, h, w = (int(v) for v in grid)
    th, tw = (h, w) if spatial_grid is None else (int(v) for v in spatial_grid[-2:])
    times = torch.arange(t, device=device, dtype=torch.long) + int(anchor) + temporal_offset
    rows = torch.arange(h, device=device, dtype=torch.long) * th // h + int(anchor)
    cols = torch.arange(w, device=device, dtype=torch.long) * tw // w + int(anchor)
    return torch.stack(torch.meshgrid(times, rows, cols, indexing="ij")).reshape(3, -1)
def reference_position_ids(vae_grid, vit_grid, anchor=0, *, device=None):
    """VAE rows followed by ViT rows, with spatial alignment and distinct time IDs."""
    return torch.cat(
        (
            image_position_ids(vae_grid, anchor, device=device),
            image_position_ids(vit_grid, anchor, spatial_grid=vae_grid, temporal_offset=1, device=device),
        ),
        dim=1,
    )


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/models/transfusion/hy/freezing.py`。
# ──────────────────────────────────────────────────────────────────────────

def freeze_vision_encoder(model):
    encoder = getattr(model.model, 'vq_visual', None)
    if encoder is None:
        raise ValueError('HY freeze_vit requires model.vq_visual')
    encoder.requires_grad_(False)
    encoder.eval()
    model._hy_frozen_vit = True
    return sum(parameter.numel() for parameter in encoder.parameters())
def encode_vision(encoder, pixels, grid_thw):
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        return encoder(pixels, grid_thw=grid_thw)
    # Parent model.train() may have reset this flag. no_grad tensors remain
    # valid saved inputs for the trainable aligner; inference_mode would not.
    encoder.eval()
    with torch.no_grad():
        return encoder(pixels, grid_thw=grid_thw)


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/utils/moe_monitor.py`。
# ──────────────────────────────────────────────────────────────────────────

ROUTER_STATS_CLASSES = ("text", "cond_image", "gen_latent")
def _load_balance_metrics(counts):
    counts = counts.float()
    total = counts.sum(-1)
    active = total > 0
    experts = counts.shape[-1]
    mean = (total / experts).clamp_min(1e-12)
    p = counts / total.clamp_min(1e-12).unsqueeze(-1)
    return {
        "imbalance_max": counts.amax(-1) / mean,
        "imbalance_cv": counts.std(-1, unbiased=False) / mean,
        "dead_expert_frac": (counts == 0).float().mean(-1),
        "load_entropy": -(p * (p + 1e-12).log()).sum(-1) / math.log(experts) if experts > 1 else active.float(),
        "eff_expert_frac": total.square() / counts.square().sum(-1).clamp_min(1e-12) / experts,
    }, active
def _mean_active(value, active):
    return (value * active).sum() / active.sum().clamp_min(1)
@torch.no_grad()
def collect_moe_expert_stats(global_tokens_per_expert_list):
    if not global_tokens_per_expert_list:
        return {}
    counts = torch.stack(global_tokens_per_expert_list).cpu()
    values, active = _load_balance_metrics(counts)
    if not bool(active.any()):
        return {}
    out = {}
    for name, vector in values.items():
        layer_name = "dead_expert" if name == "dead_expert_frac" else name
        out.update({f"moe/layer_{i}/{layer_name}": v for i, v in enumerate(vector)})
        out[f"moe/{name}"] = _mean_active(vector, active)
    return out
@torch.no_grad()
def collect_moe_expert_vision_stats(global_vision_tokens_per_expert_list):
    if not global_vision_tokens_per_expert_list:
        return {}
    counts = torch.stack(global_vision_tokens_per_expert_list).cpu()
    values, active = _load_balance_metrics(counts)
    if not bool(active.any()):
        return {}
    out = {}
    for name, vector in values.items():
        out.update({f"moe/layer_{i}/vision_{name}": v for i, v in enumerate(vector)})
        out[f"moe/vision_{name}"] = _mean_active(vector, active)
    effective = values["eff_expert_frac"] * counts.shape[-1]
    out.update({f"moe/layer_{i}/vision_eff_experts": v for i, v in enumerate(effective)})
    out["moe/vision_eff_experts"] = _mean_active(effective, active)
    return out
@torch.no_grad()
def collect_moe_router_class_stats(class_stats_list, class_names=ROUTER_STATS_CLASSES):
    if not class_stats_list:
        return {}
    counts = torch.stack(class_stats_list).cpu().float()
    if counts.ndim != 3 or counts.shape[1] != len(class_names):
        raise ValueError("Routing counts must have shape [layers, classes, experts]")
    out = {}
    for i, name in enumerate(class_names):
        metrics, active = _load_balance_metrics(counts[:, i])
        if not bool(active.any()):
            continue
        out.update({f"moe/{name}/{key}": _mean_active(v, active) for key, v in metrics.items()})
        out[f"moe/{name}/token_share"] = counts[:, i].sum() / counts.sum()
    p = counts / counts.sum(-1, keepdim=True).clamp_min(1e-12)
    for a in range(len(class_names)):
        for b in range(a + 1, len(class_names)):
            active = (counts[:, a].sum(-1) > 0) & (counts[:, b].sum(-1) > 0)
            if bool(active.any()):
                overlap = torch.minimum(p[:, a], p[:, b]).sum(-1)
                out[f"moe/overlap/{class_names[a]}-{class_names[b]}"] = _mean_active(overlap, active)
    metrics, active = _load_balance_metrics(counts.sum(1))
    if bool(active.any()):
        out.update({f"moe/all_classes/{key}": _mean_active(v, active) for key, v in metrics.items()})
    return out
@torch.no_grad()
def collect_moe_class_bias_stats(class_bias_list, class_names=ROUTER_STATS_CLASSES):
    if not class_bias_list:
        return {}
    biases = torch.stack([b.detach().float().cpu() for b in class_bias_list])
    out = {}
    for i, name in enumerate(class_names[:biases.shape[1]]):
        b = biases[:, i]
        out[f"moe/class_bias/{name}/absmean"] = b.abs().mean()
        out[f"moe/class_bias/{name}/spread"] = (b.amax(-1) - b.amin(-1)).mean()
    return out
def record_router_stats(router, indices, token_mask=None):
    """Count on log steps only. Labels -1 exclude padding. No scalar readbacks.

    Recompute may multiply counts uniformly; all published metrics are ratios.
    These transient counters never enter state_dict, EMA or balancing updates.
    """
    if not getattr(router, "_stats_enabled", False):
        return
    if not getattr(router, "_offline_routing", False) and (not router.training or not torch.is_grad_enabled()):
        return
    labels = getattr(router, "_router_token_class_local", None)
    if labels is None:
        labels = getattr(router, "_router_token_class", None)
    if labels is None:
        labels = torch.zeros(indices.shape[0], dtype=torch.long, device=indices.device)
    labels = labels.reshape(-1).to(device=indices.device, dtype=torch.long)
    if labels.numel() != indices.shape[0]:
        raise ValueError("Router class labels do not match the local routed-token axis")
    experts = router.local_tokens_per_expert.numel()
    counts = getattr(router, "_routing_class_counts", None)
    if counts is None or counts.device != indices.device:
        counts = torch.zeros((len(ROUTER_STATS_CLASSES), experts), device=indices.device, dtype=torch.float32)
        router._routing_class_counts = counts
    valid = (labels >= 0) & (labels < len(ROUTER_STATS_CLASSES))
    if token_mask is None:
        token_mask = getattr(router, "_token_mask_for_stats", None)
    if token_mask is not None:
        # Routers accept either a Boolean mask or explicit row indices (MoT's
        # empty stream uses an empty index tensor to exclude its dummy row).
        if torch.is_tensor(token_mask) and token_mask.dtype == torch.bool:
            valid = valid & token_mask.reshape(-1)
        else:
            selected = torch.zeros_like(valid)
            selected[token_mask] = True
            valid = valid & selected
    keys = labels.clamp(0, len(ROUTER_STATS_CLASSES) - 1).unsqueeze(-1) * experts + indices
    weight = valid.unsqueeze(-1).expand_as(indices).to(torch.float32)
    scale = getattr(router.local_tokens_per_expert, "_hy_cp_stats_scale", 1.0)
    counts.view(-1).scatter_add_(0, keys.reshape(-1).long(), (weight * scale).reshape(-1))
class MoERoutingMonitor:
    """Same lifecycle for ordinary/HY models. interval=0 disables monitoring."""

    def __init__(self, model, interval=20):
        if interval < 0:
            raise ValueError("moe_stats_log_steps must be nonnegative")
        self.interval = interval
        self.routers = [m for m in model.modules() if hasattr(m, "local_tokens_per_expert")]
        self.active = False
        for router in self.routers:
            router._stats_enabled = False

    def begin_step(self, step):
        self.active = bool(self.interval and step % self.interval == 0)
        for router in self.routers:
            router._stats_enabled = self.active
            if (counts := getattr(router, "_routing_class_counts", None)) is not None:
                counts.zero_()
            router._router_token_class_local = None

    @torch.no_grad()
    def collect(self, group=None):
        if not self.active or not self.routers:
            return {}
        counters = []
        for router in self.routers:
            count = getattr(router, "_routing_class_counts", None)
            if count is None:
                count = torch.zeros((len(ROUTER_STATS_CLASSES), router.local_tokens_per_expert.numel()),
                                    device=router.local_tokens_per_expert.device)
            counters.append(count)
        packed = torch.cat([c.reshape(-1) for c in counters])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
        packed = packed.cpu()
        host, offset = [], 0
        for count in counters:
            n = count.numel()
            host.append(packed[offset:offset + n].reshape(count.shape))
            offset += n
        metrics = collect_moe_expert_stats([c.sum(0) for c in host])
        metrics.update(collect_moe_router_class_stats(host))
        metrics.update(collect_moe_expert_vision_stats([c[1:].sum(0) for c in host]))
        metrics.update(collect_moe_class_bias_stats([
            r.e_score_correction_bias_class for r in self.routers
            if hasattr(r, "e_score_correction_bias_class")
        ]))
        return metrics


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/distributed/sequence_parallel/transfusion_cp.py`。
# ──────────────────────────────────────────────────────────────────────────

_MANUAL = False
def enable_transfusion_cp() -> None:
    """训练脚本在 init_parallel_state 之后、建模型之前调用一次。"""
    global _MANUAL
    _MANUAL = True
def transfusion_cp_manual() -> bool:
    """本模块接管了 Ulysses（通用 flash_attention_forward 据此跳过自动切分）。"""
    return _MANUAL
def transfusion_cp_enabled() -> bool:
    return _MANUAL and get_parallel_state().ulysses_enabled
def cp_group():
    """-> (group, rank, size)；未开启时 None。"""
    if not transfusion_cp_enabled():
        return None
    ps = get_parallel_state()
    return ps.ulysses_group, ps.ulysses_rank, ps.ulysses_size
def data_group():
    """共用同一个 pack 的一组 rank -> (group, rank, size)；未开 CP 时 None。"""
    return cp_group()
def cp_src_global_rank() -> int:
    g = data_group()
    return dist.get_global_rank(g[0], 0)
class _AllGatherSum(torch.autograd.Function):
    """dim0 拼接的 all-gather；反向以 reduce-scatter SUM 汇总各 rank 的下游贡献。"""

    @staticmethod
    def forward(ctx, x, group, size):
        ctx.group, ctx.size = group, size
        x = x.contiguous()
        out = x.new_empty((x.shape[0] * size,) + tuple(x.shape[1:]))
        dist.all_gather_into_tensor(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        out = g.new_empty((g.shape[0] // ctx.size,) + tuple(g.shape[1:]))
        dist.reduce_scatter_tensor(out, g, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None, None
def _seq_to_head(x, group, size):
    """[Lc, H, D] → [Lc·size, H/size, D]：收齐序列、按头分散。本 rank 拿第 rank 组头。"""
    lc, h, d = x.shape
    x = x.reshape(lc, size, h // size, d).transpose(0, 1).contiguous()  # [size, Lc, h/size, D]
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)  # out[j] = rank j 的 token、本 rank 的头组
    return out.reshape(size * lc, h // size, d)
def _head_to_seq(y, group, size):
    """[Lc·size, h, D] → [Lc, h·size, D]：_seq_to_head 的逆。"""
    p, h, d = y.shape
    lc = p // size
    y = y.reshape(size, lc, h, d).contiguous()  # y[j] = 第 j 段 token
    out = torch.empty_like(y)
    dist.all_to_all_single(out, y, group=group)  # out[j] = 本 rank 的 token、rank j 的头组
    return out.transpose(0, 1).reshape(lc, h * size, d)
class _SeqToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size):
        ctx.group, ctx.size = group, size
        return _seq_to_head(x, group, size)

    @staticmethod
    def backward(ctx, g):
        return _head_to_seq(g.contiguous(), ctx.group, ctx.size), None, None
class _HeadToSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, group, size):
        ctx.group, ctx.size = group, size
        return _head_to_seq(y, group, size)

    @staticmethod
    def backward(ctx, g):
        return _seq_to_head(g.contiguous(), ctx.group, ctx.size), None, None
class SeqShard:
    """长度 n 的序列在 SP 组里的均匀切分。尾部补 ``pad`` 个 token 到 ``chunk·size``。"""

    def __init__(self, n: int, group, rank: int, size: int):
        self.n = int(n)
        self.group, self.rank, self.size = group, int(rank), int(size)
        self.chunk = max(1, -(-self.n // self.size))
        self.padded = self.chunk * self.size
        self.pad = self.padded - self.n
        self.start = self.rank * self.chunk
        self.end = self.start + self.chunk
        self.n_valid = max(0, min(self.end, self.n) - self.start)

    # 切片 ------------------------------------------------------------------
    def local(self, x: torch.Tensor, dim: int = 0, value=0) -> torch.Tensor:
        """完整张量（dim 维长度 n）→ 本 rank 那一段（长度 chunk，末段含补齐）。"""
        dim = dim % x.dim()
        assert x.shape[dim] == self.n, f"SeqShard.local: dim {dim} 长度 {x.shape[dim]} != n={self.n}"
        if self.pad:
            shape = list(x.shape)
            shape[dim] = self.pad
            x = torch.cat([x, x.new_full(shape, value)], dim=dim)
        return x.narrow(dim, self.start, self.chunk)

    def range_of(self, rank: int):
        s = rank * self.chunk
        return s, max(s, min(s + self.chunk, self.n))

    def valid_mask(self, device) -> torch.Tensor:
        m = torch.zeros(self.chunk, dtype=torch.bool, device=device)
        m[: self.n_valid] = True
        return m

    def pad_cu(self, cu: torch.Tensor) -> torch.Tensor:
        """cu_seqlens 末尾补一个补齐 item。"""
        if not self.pad:
            return cu
        return torch.cat([cu, cu.new_full((1,), self.padded)])

    # 通信 ------------------------------------------------------------------
    def gather(self, x_loc: torch.Tensor) -> torch.Tensor:
        """本 rank 的 [chunk, ...] → 完整 [n, ...]（去掉补齐）。反向 reduce-scatter 求和。"""
        assert x_loc.shape[0] == self.chunk
        return self.unpad_gathered(_AllGatherSum.apply(x_loc, self.group, self.size))

    def unpad_gathered(self, full: torch.Tensor) -> torch.Tensor:
        return full.narrow(0, 0, self.n)

    def attn_in(self, x: torch.Tensor) -> torch.Tensor:
        """[chunk, H, D] → [padded, H/size, D]"""
        return _SeqToHead.apply(x, self.group, self.size)

    def attn_out(self, y: torch.Tensor) -> torch.Tensor:
        """[padded, H/size, D] → [chunk, H, D]"""
        return _HeadToSeq.apply(y, self.group, self.size)

    def attn_qkv(self, q, k, v):
        """Exchange GQA Q/K/V in one collective, grouped by each KV head."""
        length, heads, width = q.shape
        kv_heads = k.shape[1]
        assert k.shape == v.shape == (length, kv_heads, width)
        assert heads % kv_heads == 0 and kv_heads % self.size == 0
        queries = heads // kv_heads
        packed = torch.cat((q.reshape(length, kv_heads, queries, width),
                            k.unsqueeze(2), v.unsqueeze(2)), dim=2)
        packed = self.attn_in(packed.reshape(length, kv_heads, (queries + 2) * width))
        packed = packed.reshape(self.padded, kv_heads // self.size, queries + 2, width)
        return (packed[:, :, :queries].reshape(self.padded, heads // self.size, width),
                packed[:, :, queries], packed[:, :, queries + 1])
def transfusion_cp_shard(n: int) -> Optional[SeqShard]:
    g = cp_group()
    if g is None:
        return None
    return SeqShard(n, *g)
def balanced_assign(costs: Sequence[float], size: int) -> List[int]:
    """确定性的贪心均衡：大的先放，放到当前最轻的 rank（平局取小 rank）。各 rank 算出同一结果。"""
    order = sorted(range(len(costs)), key=lambda i: (-float(costs[i]), i))
    load = [0.0] * size
    owner = [0] * len(costs)
    for i in order:
        r = min(range(size), key=lambda k: (load[k], k))
        owner[i] = r
        load[r] += float(costs[i])
    return owner
def gather_by_owner(
    x_loc: torch.Tensor,
    lens: Sequence[int],
    owner: Sequence[int],
    group,
    rank: int,
    size: int,
) -> torch.Tensor:
    """各 rank 按 ``owner`` 算了自己那几项、按项序号升序拼成 ``x_loc``；拼回完整的项序。

    ``lens[i]`` 是第 i 项的行数（各 rank 都知道）。前向 all-gather（补到最大长度），反向
    reduce-scatter 求和，所以下游可以只用其中任意一部分。
    """
    per_rank = [0] * size
    for i, n in enumerate(lens):
        per_rank[owner[i]] += int(n)
    assert x_loc.shape[0] == per_rank[rank], f"gather_by_owner: 本 rank {x_loc.shape[0]} 行，应为 {per_rank[rank]}"
    m = max(max(per_rank), 1)
    if x_loc.shape[0] < m:
        x_loc = torch.cat([x_loc, x_loc.new_zeros((m - x_loc.shape[0],) + tuple(x_loc.shape[1:]))], dim=0)
    buf = _AllGatherSum.apply(x_loc, group, size)  # [size·m, ...]
    off = [0] * size
    idx = []
    for i, n in enumerate(lens):
        r = owner[i]
        s = r * m + off[r]
        idx.append(torch.arange(s, s + int(n), dtype=torch.long))
        off[r] += int(n)
    if not idx:
        return buf[:0]
    return buf.index_select(0, torch.cat(idx).to(buf.device))
@torch.no_grad()
def gather_varlen_nograd(x_loc: torch.Tensor, per_rank: Sequence[int], group, rank: int, size: int) -> torch.Tensor:
    """各 rank 行数为 ``per_rank`` 的张量按 rank 序拼起来（不求导）。"""
    m = max(max(per_rank), 1)
    if x_loc.shape[0] < m:
        x_loc = torch.cat([x_loc, x_loc.new_zeros((m - x_loc.shape[0],) + tuple(x_loc.shape[1:]))], dim=0)
    buf = x_loc.new_empty((m * size,) + tuple(x_loc.shape[1:]))
    dist.all_gather_into_tensor(buf, x_loc.contiguous(), group=group)
    return torch.cat([buf[r * m : r * m + per_rank[r]] for r in range(size)], dim=0)
def _describe_cp_input(value: Any, source_tensors: List[torch.Tensor]) -> Any:
    # Keep recursion at module scope: a recursive local function references
    # itself through its closure and can retain every source image until GC.
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("cp_broadcast_inputs accepts raw data, not tensors requiring gradients")
        source_tensors.append(value)
        return ("tensor", (tuple(value.shape), value.dtype, value.device.type == "cpu"))
    if isinstance(value, dict):
        return ("dict", [(key, _describe_cp_input(item, source_tensors)) for key, item in value.items()])
    if isinstance(value, (list, tuple)):
        return (
            "list" if isinstance(value, list) else "tuple",
            [_describe_cp_input(item, source_tensors) for item in value],
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    raise TypeError(f"Unsupported CP input type: {type(value).__name__}")
def _receive_cp_input(spec: Any, source_tensors: Iterator[torch.Tensor], device, group, source, rank) -> Any:
    kind, value = spec
    if kind == "tensor":
        shape, dtype, on_cpu = value
        if rank == 0:
            tensor = next(source_tensors).contiguous().to(device)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        if tensor.numel() > 0:
            dist.broadcast(tensor, src=source, group=group)
        return tensor.cpu() if on_cpu else tensor
    if kind == "dict":
        return {
            key: _receive_cp_input(item, source_tensors, device, group, source, rank) for key, item in value
        }
    if kind == "list":
        return [_receive_cp_input(item, source_tensors, device, group, source, rank) for item in value]
    if kind == "tuple":
        return tuple(_receive_cp_input(item, source_tensors, device, group, source, rank) for item in value)
    return value
@torch.no_grad()
def cp_broadcast_inputs(inputs: Any, device: torch.device) -> Any:
    """Copy the CP leader's raw input tree, including shapes and CPU metadata.

    Run before any collective whose shape depends on the input (notably the
    parallel VAE encode). Other ranks may have different shapes, missing keys,
    or a different number of images. Only small metadata is pickled; tensors
    are broadcast in their original dtype on ``device``. CPU tensors such as
    cu_seqlens are moved back to CPU after communication.

    This is for data, never trainable embeddings/activations: an ordinary
    broadcast would disconnect their autograd graph. CP1 is an identity.
    """
    group_info = cp_group()
    if group_info is None or group_info[2] == 1:
        return inputs
    group, rank, _ = group_info
    source = dist.get_global_rank(group, 0)
    source_tensors = []

    # Report invalid leader input to every rank before tensor collectives.
    metadata = [None]
    if rank == 0:
        try:
            metadata[0] = (_describe_cp_input(inputs, source_tensors), None)
        except (TypeError, ValueError) as error:
            metadata[0] = (None, str(error))
    dist.broadcast_object_list(metadata, src=source, group=group, device=device)
    specification, error = metadata[0]
    if error is not None:
        raise ValueError(error)
    return _receive_cp_input(specification, iter(source_tensors), device, group, source, rank)
@torch.no_grad()
def cp_broadcast_(tensors: Sequence[Optional[torch.Tensor]], group=None, src: Optional[int] = None) -> None:
    """就地广播（默认：SP 组内从组 rank0）。"""
    if group is None:
        group = data_group()[0]
        src = cp_src_global_rank()
    for t in tensors:
        if t is not None and t.numel() > 0:
            dist.broadcast(t, src=src, group=group)
@torch.no_grad()
def cp_assert_same(signature: Sequence[int], device, what: str = "batch") -> None:
    """SP 组内各 rank 的整数签名必须完全一致（数据 / 控制流对齐的闸门）。"""
    g, rank, size = data_group()
    sig = torch.tensor([int(v) for v in signature], dtype=torch.int64, device=device)
    buf = sig.new_empty((size, sig.numel()))
    dist.all_gather_into_tensor(buf, sig, group=g)
    if not bool((buf == buf[0]).all()):
        raise RuntimeError(
            f"[cp] SP 组内各 rank 的 {what} 不一致（CP 要求同组拿同一个 pack）：\n{buf.tolist()}\n"
            "检查数据侧是否有按全局 rank 取种子 / 分片的地方。"
        )
@torch.no_grad()
def cp_vae_encode(
    images: Sequence[torch.Tensor],
    token_counts: Sequence[int],
    encode_one,
    feat_dim: int,
    dtype: torch.dtype,
    device,
) -> List[torch.Tensor]:
    """冻结 VAE 按图分给 SP 组各 rank，只编码自己那份，拼回原顺序。

    ``encode_one(img) -> [n_i, feat_dim]``；``token_counts[i]`` 是第 i 张图的 token 数
    （由 latent shape 给出，各 rank 都知道）。
    """
    g, rank, size = cp_group()
    costs = [float(img.shape[-1]) * float(img.shape[-2]) for img in images]
    owner = balanced_assign(costs, size)
    mine = [i for i in range(len(images)) if owner[i] == rank]
    parts = []
    for i in mine:
        z = encode_one(images[i])
        if z.shape[0] != int(token_counts[i]) or z.shape[1] != feat_dim:
            raise RuntimeError(
                f"[cp] VAE 第 {i} 张图编码出 {tuple(z.shape)}，按 latent shape 应为 ({token_counts[i]}, {feat_dim})"
            )
        parts.append(z.to(dtype))
    x_loc = torch.cat(parts, dim=0) if parts else torch.zeros((0, feat_dim), dtype=dtype, device=device)
    full = gather_by_owner(x_loc, token_counts, owner, g, rank, size)
    out, o = [], 0
    for n in token_counts:
        out.append(full[o : o + int(n)])
        o += int(n)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/optimizations/transfusion30b/inputs.py`。
# ──────────────────────────────────────────────────────────────────────────

_active = None
def scatter_token_rows(inputs, mask, source, rows=None):
    if (rows is None or rows.ndim != 1 or rows.dtype != torch.int64
            or rows.device != inputs.device or mask.dtype != torch.bool
            or mask.ndim != inputs.ndim or mask.shape[-1] != 1
            or mask.shape[:-1] != inputs.shape[:-1] or source.ndim != 2
            or source.shape[-1] != inputs.shape[-1] or source.shape[0] == 0
            or rows.numel() != source.shape[0]):
        return inputs.masked_scatter(mask, source)
    # rows are sorted unique locations of unchanged CPU input IDs. This is
    # the same row insertion and the same backward, without device nonzero.
    return inputs.reshape(-1, inputs.shape[-1]).index_copy(0, rows, source).reshape(inputs.shape)
class InputRows:
    def __init__(self, config):
        global _active
        self.config, self.pending = config, deque()
        _active = self

    def prepare(self, batches):
        if self.pending:
            raise RuntimeError("Previous pack's token row metadata was not consumed")
        for batch in batches:
            ids = batch["input_ids"]
            if ids.device.type != "cpu":
                raise ValueError("Prepare row metadata before moving input IDs to NPU")
            self.pending.append({
                "_veomni_image_rows": (ids.reshape(-1) == self.config.image_token_id).nonzero(as_tuple=True)[0],
                "_veomni_vae_rows": (ids.reshape(-1) == self.config.vae_token_id).nonzero(as_tuple=True)[0],
            })
        return batches

    def forward_pre_hook(self, model, args, kwargs):
        if self.pending:
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None:
                raise ValueError("Prepared input rows require input_ids")
            rows = self.pending.popleft()
            kwargs.update({name: value.to(ids.device, non_blocking=True) for name, value in rows.items()})
        return args, kwargs
class PreparedLoader:
    """Delegate data/RNG/checkpoint state to the original loader unchanged."""

    def __init__(self, native):
        self.native, self.iterator = native, None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __len__(self):
        return len(self.native)

    def __iter__(self):
        self.iterator = iter(self.native)
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.native)
        if _active is None:
            raise RuntimeError("The model's input metadata hook is not installed")
        return _active.prepare(next(self.iterator))


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/optimizations/hy106/entry_split.py`。
# ──────────────────────────────────────────────────────────────────────────

counts = Counter()
def entry_split(conditional_latents, cumulative_lengths, conditional_items):
    offset = conditional_latents.shape[0] if conditional_latents is not None else 0
    if os.environ.get("VEOMNI_HY106_ENTRY_SPLIT_VALIDATE") == "1":
        reference = int(cumulative_lengths[conditional_items].item())
        assert offset == reference, ("Conditional entry split changed", offset, reference)
        counts["actual_model_exact_comparisons"] += 1
    counts["host_shape_offsets"] += 1
    return offset


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/optimizations/hy106/fixed_visual_scatter.py`。
# ──────────────────────────────────────────────────────────────────────────

counts = Counter()
class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, text, rows, source):
        assert tuple(rows.shape) == tuple(text.shape[:-1])
        assert source.ndim == 2 and source.shape[1] == text.shape[-1]
        ctx.save_for_backward(rows)
        ctx.source_shape = tuple(source.shape)
        counts["forward_calls"] += 1
        return text.masked_scatter(rows.unsqueeze(-1).expand_as(text), source)

    @staticmethod
    def backward(ctx, gradient):
        (rows,) = ctx.saved_tensors
        dx = gradient.masked_fill(rows.unsqueeze(-1), 0) if ctx.needs_input_grad[0] else None
        ds = None
        if ctx.needs_input_grad[2]:
            tokens, width = ctx.source_shape
            if tokens:
                positions = torch.arange(rows.numel(), device=rows.device, dtype=torch.int64)
                keys = torch.where(rows.reshape(-1), positions, positions + rows.numel())
                indices = torch.topk(keys, tokens, largest=False, sorted=True).indices
                ds = gradient.reshape(-1, width).index_select(0, indices)
            else:
                ds = gradient.new_empty(ctx.source_shape)
            counts["fixed_size_backward_calls"] += 1
        return (dx, None, ds)
def scatter(text, rows, source):
    if torch.is_grad_enabled() and (text.requires_grad or source.requires_grad):
        return Scatter.apply(text, rows, source)
    return text.masked_scatter(rows.unsqueeze(-1).expand_as(text), source)


# ──────────────────────────────────────────────────────────────────────────
# 逐字节搬自 `veomni/optimizations/hy106/sparse_ar.py`。
# ──────────────────────────────────────────────────────────────────────────

records = []
def select(hidden, labels, indices=None):
    if indices is None:
        indices = (labels != -100).nonzero(as_tuple=True)[0]
    assert indices.numel() > 0, "The existing all-ignored branch must remain in use"
    if indices.numel() == labels.numel():
        return (hidden, labels)
    return (hidden.index_select(0, indices), labels.index_select(0, indices))
