"""luyizhuo · HY transfusion 模型的**纯文本 KV-cache 自回归解码**（910C / NPU）。

为什么要新写一条路
------------------
HY 的注意力（`GlmImageMoeTransfusionAttention`）在 `__init__` 里把 `self.forward`
**换成** `_fa3_block_fused_forward` / `_two_pass_mla_forward`，这两条都是
**varlen + blockmask** 的训练路径：`past_key_values` 形参收下了但一行都不用。
⇒ 直接拿 HY 的 forward 做 AR 解码 = 每出一个 token 重算整条序列（O(L²)），不可用。

但它的**父类** `Glm4MoeLiteAttention.forward`（`veomni/models/glm4_moe_lite/
modeling_glm4_moe_lite.py:402`）**本来就带完整的 KV-cache 实现**（naive full-K/V 的
`_forward_full` + MLA 矩阵吸收的 `_forward_absorb`，后者在 NPU 上实测慢 25×，所以默认
不用）。而 `GlmImageMoeTransfusionAttention` **没有在类上定义 `forward`** ——
它只是往实例字典里塞了一个。于是：

    del attn.__dict__["forward"]      # 实例属性一摘，attn.forward 就落回父类那份

本文件就是围绕这一条做的：**换注意力实现、不换权重、不换任何 Linear/MoE/Norm**。

三件必须对齐的事（错一件就是静默出垃圾）
----------------------------------------
① **位置编码**：HY 的 rope 是 `GlmImageTextRotaryEmbedding(position_ids[3,B,L])`
   → `cos/sin [3,B,L,64]`，再由 `_assemble_mrope(..., image_rope_mask)` 按行 gather。
   **文本行走的是 `sec_block` 表**，而纯文本的 t/h/w 位置**三轴相等**
   ⇒ 不论 gather 到哪个轴，取到的都是同一组值 ⇒ `cos3d[0]` 与 `_assemble_mrope`
   在文本行上**逐位相同**。所以这里直接用轴 0，等价且省一次 gather。
   （`rope_table_dtype=float32` ⇒ 与训练同走 `_rope_tables` 的 fp32 分支。）
② **注意力实现**：`_forward_full` 读 `self.config._attn_implementation`，而
   `fa3_block` 不在 `ALL_ATTENTION_FUNCTIONS` 里 ⇒ 必须临时换成 `eager` 或 `sdpa`。
   * `eager`（默认）：纯 matmul+softmax，NPU 上一定能跑，需要显式加性 mask；
   * `sdpa`：`attention_mask=None` 时 transformers 自己按 `q_len>1` 置 `is_causal`
     （`integrations/sdpa_attention.py`），prefill 因果、decode 全可见，正好对；
     但它会顺带传 `enable_gqa=True`，个别 torch_npu 版本不吃 ⇒ 留成可选。
   本模型 `qk_head_dim == v_head_dim == 256`，所以两条路都不需要 pad value。
③ **FSDP**：hp_fsdp2 用的是标准 `register_forward_pre_hook/forward_hook`
   （`hyper_parallel/platform/torch/fully_shard/scheduler.py:266-271`），逐层 unshard
   靠调 `layer(...)` 触发。为了让 root 的那对 hook 也正常走一遍（root 上挂着
   embed/lm_head 的 prefetch 计划），**整个生成循环放在一次 root forward 里跑**
   —— 临时把 `model.forward` 换成循环体，调一次 `model(...)`，循环结束再换回来。

并行语义
--------
FSDP 的 all-gather 只搬**参数**、与喂什么数据无关 ⇒ 同一个 shard 组里各 rank 完全可以
跑**不同的 prompt**（这就是数据并行），**唯一的硬约束是前向次数必须一样**。
而思考链长短随样本 ⇒ 每出一个 token 要在 `param_shard_group` 上 all-reduce 一个
"我还没停"的标志，组内一起跑到**组内最长**那条为止；已停的 rank 继续喂哑 token 走
前向、结果丢掉（`wasted_steps` 会记下来）。这样满世界 32 路并行，代价是组内对齐到
max 链长（实测链长 500~2500 ⇒ 约 2× 冗余），比"一组一条 prompt"（并行度 = 组数）好得多。
**不同 dp replica 之间不需要任何同步** —— 前向里没有跨 replica 的集合通信
（MoE 的 `collect_global_tokens_per_expert` 是训练循环调的，不在 forward 里）。

采样超参（默认值 = 2026-11-13 在 sglang 上做的 16 臂消融定稿）
--------------------------------------------------------------
    temperature 0.6 / top_p 0.9 / top_k 50 / repetition_penalty 1.0
    max_new_tokens 4096（训练 think+final 的 p99 是 4097）/ stop=[<boi>]

★ `repetition_penalty` 必须是 1.0，**不能**沿用旧的 1.3。
  16 臂 × 8 prompt 实测自然收尾率：rep 1.0 → 1.00、rep 1.05 → 1.00、
  rep 1.1 → 0.38、**rep 1.3 → 0.00**。机制见 `apply_sampling` 里的长注释：
  rep 是对所有出现过的 token 一刀切，链长过千后把模型逼出分布。
  旧的 0.8/1.3 来自 2026-09-15 旧仓 sft_v4 那轮 —— 那次的病是**贪心复读环**，
  rep 1.3 对症；换了 ckpt 就换了病，药方不能照抄。

★ 要压复读环请用 `frequency_penalty`（按出现次数线性累加的减法），不要调大 rep。
  rep=1.0 下仍有约 4% 的链会掉进真正的句子级复读环（实测 1/24），
  这正是 frequency_penalty 该管的事。
"""


from __future__ import annotations

import contextlib
import time
import types
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


THINK_ID = 151350      # 旧 <think> 透传位（新词表 decode 为 'ऄ'，预期外观）
THINK_END_ID = 151351  # 旧 </think>
SOP_ID = 154824        # <sop>
EOP_ID = 154825        # <eop>
BOI_ID = 154830        # <boi> = image_start_token_id
USER_ID = 154827       # <|user|>
ASSISTANT_ID = 154828  # <|assistant|>


@dataclass
class ARSamplingConfig:
    """AR（思考链）采样旋钮。默认值见模块 docstring。"""

    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    #: `logit -= frequency_penalty × 出现次数`（OpenAI/sglang 口径，只数**已生成**的部分）。
    #: 这是压「句子级复读环」的对症旋钮；别拿 repetition_penalty 去压（见 apply_sampling 注释）。
    frequency_penalty: float = 0.0
    #: `logit -= presence_penalty × (出现过 ? 1 : 0)`。本轮没用到，留着对齐 sglang 口径。
    presence_penalty: float = 0.0
    max_new_tokens: int = 4096
    seed: int = 0
    # 生成到这些 token 就停。默认只有 <boi>（= 思考链自然收敛的标志）。
    stop_token_ids: list[int] = field(default_factory=lambda: [BOI_ID])
    # 额外的早停：出现 <sop> 说明模型开始写分辨率网格，思考+final 已经写完。
    # 我们要**自己**按请求分辨率拼 `<sop>Hg Wg<eop>`，所以到这里就可以收工。
    stop_at_sop: bool = True
    # ★ final 段长度上限（0 = 不限）。>0 时：一旦出了 </think>₁₅₁₃₅₁，再生成这么多 token 就停。
    #
    #   为什么需要它（2026-11-13 step7000 实测）：**思考块会正常闭合，但 final 段学不会收尾**。
    #   实测一条人像题在 ~768 token 处正确打出 </think> 并开始写 final，然后一路漂到
    #   词汇堆砌（"moral integrity compassion kindness empathy ... peace love acceptance"）
    #   都不出 <sop>。注意这**不是复读环** —— 词都不重复，所以 repetition_penalty 咬不住它。
    #   后果：扩散拿到的"最终 caption"是一段几千 token 的流水账，比思考块被截断更糟。
    #   训练侧的 `image_generation_prompt` 是一段 T2I caption（实测量级 80~250 token），
    #   所以给个上限是**贴着训练分布**的解码约束，不是凑数。
    #   ⚠ 它会让 `stop_reason` 变成 `final_cap`：这正是要如实记到网页上的"模型成熟度"信号，
    #     别把它当成"链路有问题"。
    max_final_tokens: int = 0
    # ★ 给 final 预留的 token 数（0 = 关闭这道保险）。>0 时：生成到
    #   `max_new_tokens - final_reserve_tokens` 还没出现 </think>₁₅₁₃₅₁，就**替模型写上**。
    #
    #   为什么需要它（2026-11-14 实测，RESULTS.md §3.8 / §3.9）：定稿采样参数之后仍有
    #   约 14% 的链收不住，其中**最重的一种形态是「连 </think> 都没写出来」** ——
    #   没有 final，`trim_generated` 会把整段漂移的 think 当 context 灌给扩散。
    #   这种形态只出现在 1536² 以上（2048² 3/24、1792² 1/24、1536² 2/24，低两档 0）。
    #   实测（sglang 上 2048²×10 个高危 prompt×2 次，n=20）：3 条没合 </think>，
    #   强行合上之后 **3/3 都写出了成句 final 并自己收尾**（自发 <sop>H W<eop><boi>），
    #   prompt 词重叠中位 0.778（同轮自然收尾链是 0.861，降约 10%）、长度中位 500 字。
    #   ⚠ n=3，别读成「100% 能救」—— 但它**只在链已经坏掉时才触发**，
    #     所以对本来正常的链零影响，下界是「不更差」。
    #   ⚠ 预留量要够：实测救回的 final 是 352/354/487 token ⇒ 600 比较稳，
    #     给 400 会把一部分 final 再截断一次。
    #   实现上是**替换当步的采样结果**而不是额外插 token ⇒ 前向次数不变，
    #   组内 lockstep（各 rank 前向次数必须一致）不受影响。
    final_reserve_tokens: int = 0
    # 每多少个 token 打一次心跳（0 = 不打）
    heartbeat: int = 128

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed, "stop_token_ids": list(self.stop_token_ids),
            "stop_at_sop": self.stop_at_sop, "max_final_tokens": self.max_final_tokens,
            "final_reserve_tokens": self.final_reserve_tokens,
        }


def apply_sampling(logits: torch.Tensor, generated: list[int], prompt_ids: list[int],
                   cfg: ARSamplingConfig, generator: torch.Generator) -> int:
    """temperature → repetition_penalty → top_k → top_p → multinomial。

    口径与 sglang/vLLM 一致的几处：
      * `repetition_penalty` 作用在**已出现过的 token**（prompt + 已生成）上，
        正 logit 除以惩罚、负 logit 乘以惩罚（HF `RepetitionPenaltyLogitsProcessor`）；
      * `top_k` 先、`top_p` 后；
      * `temperature == 0` ⇒ 退化成贪心（不采样）。

    ★ 整个采样**在 CPU 上做**（`generator` 也是 CPU 的）：
      ① 一个 154880 维的 float 向量只有 0.6 MB，搬下来的代价可以忽略，而我们每步本来
         就要 `.item()` 同步一次；
      ② `multinomial` / `sort` / `cumsum` 带 generator 在 torch_npu 上的行为不值得赌，
         放 CPU 则**换硬件/换 torch 版本都逐位可复现**（横比干净）。
    """
    logits = logits.detach().float().cpu()
    if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
        seen = torch.tensor(sorted(set(prompt_ids) | set(generated)), dtype=torch.long)
        if seen.numel():
            picked = logits.index_select(0, seen)
            picked = torch.where(picked > 0, picked / cfg.repetition_penalty,
                                 picked * cfg.repetition_penalty)
            logits = logits.index_copy(0, seen, picked)
    # ★ frequency / presence penalty —— OpenAI / sglang 口径：
    #     logit[t] -= frequency_penalty × count[t] + presence_penalty × (count[t] > 0)
    #   与 repetition_penalty 的区别是**本质性的**，别混用：
    #     · repetition_penalty 对**所有出现过**的 token 一刀切（乘除法）⇒ 链长过千后
    #       已出现集合覆盖了大部分常用词，模型被逼去搜没用过的词 ⇒ 漂出分布。
    #     · frequency_penalty 是**按出现次数线性累加**的减法 ⇒ 出现一次几乎不罚，
    #       真的在复读（同一 token 反复出现十几次）才被重罚。
    #   所以要压「句子级复读环」该用 frequency_penalty，而不是把 repetition_penalty 调大。
    #   ⚠ 只统计**已生成**的部分，不含 prompt —— prompt 里的词本来就该能再用
    #     （与 sglang 的 `frequency_penalty` 一致；repetition_penalty 那边才含 prompt）。
    if (cfg.frequency_penalty or cfg.presence_penalty) and generated:
        counts: dict[int, int] = {}
        for token in generated:
            counts[token] = counts.get(token, 0) + 1
        index = torch.tensor(sorted(counts), dtype=torch.long)
        count = torch.tensor([counts[int(t)] for t in index], dtype=logits.dtype)
        delta = (float(cfg.frequency_penalty or 0.0) * count
                 + float(cfg.presence_penalty or 0.0) * (count > 0).to(logits.dtype))
        logits = logits.index_copy(0, index, logits.index_select(0, index) - delta)
    if cfg.temperature is None or cfg.temperature <= 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / float(cfg.temperature)
    if cfg.top_k and cfg.top_k > 0 and cfg.top_k < logits.numel():
        kth = torch.topk(logits, cfg.top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    if cfg.top_p and 0.0 < cfg.top_p < 1.0:
        ordered, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        # 保留第一个使累积 ≥ top_p 的位置（保证至少 1 个候选）
        drop = cumulative - ordered > cfg.top_p
        ordered = torch.where(drop, torch.zeros_like(ordered), ordered)
        probs = torch.zeros_like(probs).index_copy(0, order, ordered)
    total = probs.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return int(torch.argmax(logits).item())
    probs = probs / total
    return int(torch.multinomial(probs, 1, generator=generator).item())


class HYTextDecoder:
    """HY 模型上的纯文本增量解码器。用法：

        decoder = HYTextDecoder(model, device, shard_group=...)
        gen_ids, info = decoder.generate(prefix_ids, cfg)
    """

    def __init__(self, model, device, *, shard_group=None, attn_impl: str = "eager",
                 logger=None):
        self.model = model
        self.device = device
        self.shard_group = shard_group
        self.attn_impl = attn_impl
        self.logger = logger
        self.backbone = model.model                      # GlmImageMoeTransfusionBackbone(HY)
        self.layers = self.backbone.layers
        self.text_config = model.config.text_config
        self.n_layers = int(self.text_config.num_hidden_layers)
        if attn_impl not in ("eager", "sdpa"):
            raise ValueError(f"attn_impl 只支持 eager / sdpa，得到 {attn_impl!r}")

    # ----------------------------------------------------------- 实现切换
    @contextlib.contextmanager
    def _hf_attention(self):
        """把每层注意力从 HY 的 blockmask 实现切回父类那份带 KV-cache 的实现。"""
        saved: list[tuple[object, object]] = []
        saved_impl = getattr(self.text_config, "_attn_implementation", None)
        try:
            for layer in self.layers[: self.n_layers]:
                attn = layer.self_attn
                saved.append((attn, attn.__dict__.pop("forward", None)))
            self.text_config._attn_implementation = self.attn_impl
            yield
        finally:
            for attn, previous in saved:
                if previous is not None:
                    attn.__dict__["forward"] = previous
                else:
                    attn.__dict__.pop("forward", None)
            if saved_impl is not None:
                self.text_config._attn_implementation = saved_impl

    # --------------------------------------------------------------- 前向
    def _rope(self, start: int, length: int, reference: torch.Tensor):
        """纯文本 rope：三轴位置相等 ⇒ 取 `_assemble_mrope` 在文本行上等价的轴 0。"""
        positions = torch.arange(start, start + length, device=self.device)
        position_ids = positions.view(1, -1).expand(3, -1).unsqueeze(1).contiguous()  # [3,1,n]
        cos, sin = self.backbone.rotary_emb(
            torch.empty(1, dtype=torch.float32, device=self.device), position_ids
        )  # [3,1,n,D]
        return cos[0], sin[0], position_ids, positions  # [1,n,D]

    def _causal_mask(self, q_len: int, kv_len: int, dtype: torch.dtype):
        if self.attn_impl == "sdpa":
            return None  # transformers 自己按 q_len>1 置 is_causal
        if q_len == 1:
            return None  # decode：全可见，加性 mask 全 0，省掉它
        offset = kv_len - q_len
        row = torch.arange(q_len, device=self.device).view(-1, 1) + offset
        col = torch.arange(kv_len, device=self.device).view(1, -1)
        blocked = col > row
        mask = torch.zeros((1, 1, q_len, kv_len), dtype=dtype, device=self.device)
        mask.masked_fill_(blocked.view(1, 1, q_len, kv_len), torch.finfo(dtype).min)
        return mask

    @torch.no_grad()
    def _chunk_logits(self, ids: torch.Tensor, start: int, cache) -> torch.Tensor:
        """喂一段 token（prefill 整条前缀 / decode 一个 token），返回最后位置 logits [V]。"""
        length = int(ids.shape[1])
        embeds = self.backbone.embed_tokens(ids)
        cos, sin, position_ids, positions = self._rope(start, length, embeds)
        cos = cos.to(embeds.dtype)
        sin = sin.to(embeds.dtype)
        kv_len = start + length
        mask = self._causal_mask(length, kv_len, embeds.dtype)
        hidden = embeds
        for layer in self.layers[: self.n_layers]:
            out = layer(
                hidden_states=hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=positions,
                position_embeddings=(cos, sin),
            )
            hidden = out[0] if isinstance(out, tuple) else out
        hidden = self.backbone.norm(hidden)
        return self.model.lm_head(hidden[:, -1]).float()[0]

    # ------------------------------------------------------------- 采样环
    def _group_any_running(self, running: bool) -> bool:
        """组内还有 rank 没停吗？（在 `param_shard_group` 上 all-reduce MAX）

        为什么需要它：FSDP 的 all-gather 只搬**参数**、与喂什么数据无关 ⇒ 同一个 shard
        组里各 rank 可以跑**不同的 prompt**；但**前向次数必须一样**，否则 all-gather
        错位直接挂死。思考链长短随样本 ⇒ 每步同步一个标志，组内一起跑到最长那条为止，
        已停的 rank 继续喂哑 token 走前向、结果丢掉。
        （`group=None`，即不分片/单卡时，这个函数就是恒等。）
        """
        if self.shard_group is None or dist.get_world_size(self.shard_group) <= 1:
            return running
        buffer = torch.tensor([1 if running else 0], dtype=torch.int32, device=self.device)
        dist.all_reduce(buffer, op=dist.ReduceOp.MAX, group=self.shard_group)
        return bool(buffer.item())

    @torch.no_grad()
    def _generate_inner(self, prefix_ids: torch.Tensor, cfg: ARSamplingConfig,
                        tokenizer=None) -> tuple[list[int], dict]:
        from transformers.cache_utils import DynamicCache

        prompt_ids = prefix_ids.view(-1).tolist()
        ids = prefix_ids.view(1, -1).to(self.device)
        prefix_len = int(ids.shape[1])
        cache = DynamicCache()
        generator = torch.Generator().manual_seed(int(cfg.seed))  # CPU 采样，见 apply_sampling
        generated: list[int] = []
        stop_set = {int(t) for t in cfg.stop_token_ids}
        reason, running, wasted = "max_new_tokens", True, 0
        think_closed_at = None   # </think> 出现在 generated 的第几个位置
        forced_close_at = None   # 到预算被**强行**合上的位置（None = 模型自己写的）
        started = time.time()

        t0 = time.time()
        logits = self._chunk_logits(ids, 0, cache)
        prefill_s = time.time() - t0
        step = 0
        while True:
            if running:
                token = apply_sampling(logits, generated, prompt_ids, cfg, generator)
                # ★ 思考链预算用尽还没合 </think> ⇒ 替它合上（见模块 docstring ⑤）。
                #   这里是**替换采样结果**而不是额外插一个 token ⇒ 前向次数不变，
                #   组内 lockstep 的「各 rank 前向次数必须一样」不受影响。
                if (cfg.final_reserve_tokens and think_closed_at is None
                        and len(generated) >= int(cfg.max_new_tokens)
                        - int(cfg.final_reserve_tokens)):
                    token = THINK_END_ID
                    forced_close_at = len(generated)
                if cfg.stop_at_sop and token == SOP_ID:
                    reason, running = "sop", False
                else:
                    generated.append(token)
                    if token == THINK_END_ID and think_closed_at is None:
                        think_closed_at = len(generated)
                    if token in stop_set:
                        reason, running = f"stop:{token}", False
                    elif (cfg.max_final_tokens and think_closed_at is not None
                          and len(generated) - think_closed_at >= int(cfg.max_final_tokens)):
                        reason, running = "final_cap", False
                    elif len(generated) >= int(cfg.max_new_tokens):
                        reason, running = "max_new_tokens", False
            else:
                wasted += 1
            if not self._group_any_running(running):
                break
            feed = generated[-1] if generated else int(prompt_ids[-1])
            nxt = torch.tensor([[feed]], dtype=torch.long, device=self.device)
            logits = self._chunk_logits(nxt, prefix_len + step, cache)
            step += 1
            if cfg.heartbeat and self.logger is not None and running and step % cfg.heartbeat == 0:
                tail = tokenizer.decode(generated[-cfg.heartbeat:], skip_special_tokens=False) \
                    if tokenizer is not None else ""
                self.logger.info(
                    f"[ar] +{len(generated)} tok "
                    f"({len(generated) / max(1e-6, time.time() - started):.2f} tok/s) tail={tail[:160]!r}"
                )

        elapsed = time.time() - started
        info = {
            "prefix_len": prefix_len,
            "gen_len": len(generated),
            "stop_reason": reason,
            "natural_boi": reason in ("sop", f"stop:{BOI_ID}"),
            "group_steps": step + 1,
            "wasted_steps": wasted,
            "seconds": round(elapsed, 2),
            "prefill_seconds": round(prefill_s, 2),
            "tok_per_s": round(len(generated) / elapsed, 3) if elapsed > 0 else None,
            "has_think_open": THINK_ID in generated,
            "has_think_close": THINK_END_ID in generated,
            "forced_think_close": forced_close_at is not None,
            "forced_think_close_at": forced_close_at,
            "think_len": think_closed_at,
            "final_len": (len(generated) - think_closed_at) if think_closed_at is not None else None,
            "sampling": cfg.as_dict(),
            "attn_impl": self.attn_impl,
        }
        return generated, info

    @contextlib.contextmanager
    def _root_forward(self):
        """让整段工作跑在**一次 root forward** 里（见模块 docstring ③）。

        为什么非得这么做：`backbone.norm` 这类"不属于任何子 FSDP 单元"的参数归**root**
        单元管（`hp_fsdp2.py:402` 的 `hp_fully_shard(model, **root_kwargs)`），只有 root
        自己的 forward pre-hook 会把它 unshard。逐层调 `layer(...)` 只触发层的 hook，
        root 的参数仍是分片态 ⇒ `norm` 拿到的是 1/shard 的碎片（形状对不上或静默出错）。
        临时把 `model.forward` 换成一个闭包，调一次 `model()`，hook 正常走一遍。
        """
        box: dict = {}

        def _runner(_self, *_args, **_kwargs):
            box["value"] = box["fn"]()
            return 0  # 返回标量常量：别把 token 列表交给 FSDP 的 post-forward hook

        def _call(fn):
            box["fn"] = fn
            saved = self.model.__dict__.get("forward", None)
            try:
                self.model.forward = types.MethodType(_runner, self.model)
                self.model()
            finally:
                if saved is None:
                    self.model.__dict__.pop("forward", None)
                else:
                    self.model.__dict__["forward"] = saved
            return box["value"]

        yield _call

    @torch.no_grad()
    def generate(self, prefix_ids: torch.Tensor, cfg: ARSamplingConfig,
                 tokenizer=None) -> tuple[list[int], dict]:
        with self._hf_attention(), self._root_forward() as call:
            return call(lambda: self._generate_inner(prefix_ids, cfg, tokenizer))

    # ------------------------------------------------------------ 精度闸门
    @torch.no_grad()
    def verify_against_nocache(self, prefix_ids: torch.Tensor, n_steps: int = 8,
                               atol: float = 2.0) -> dict:
        """把 KV-cache 快路径与**无 cache 的整条重算**逐 token 对拍（贪心）。

        两条路走的是同一份注意力实现（父类 `_forward_full`），区别只在"有没有 cache"
        ⇒ 这条闸门验的是 cache / 位置 / mask 的接法。判据是 **argmax 一致**；
        `max_abs_delta` 只作参考（跨 kernel 路径 bit-identical 本就不可达）。
        """
        from transformers.cache_utils import DynamicCache

        ids = prefix_ids.view(1, -1).to(self.device)
        prefix_len = int(ids.shape[1])
        rows = []

        def _run():
            cache = DynamicCache()
            fast = self._chunk_logits(ids, 0, cache)           # prefill
            slow = self._chunk_logits(ids, 0, DynamicCache())  # 同一条，另一份 cache
            current = ids
            for step in range(n_steps):
                fast_token = int(torch.argmax(fast).item())
                slow_token = int(torch.argmax(slow).item())
                rows.append({
                    "step": step, "fast": fast_token, "slow": slow_token,
                    "max_abs_delta": round(float((fast - slow).abs().max().item()), 4),
                    "ok": fast_token == slow_token,
                })
                token = fast_token
                current = torch.cat(
                    [current, torch.tensor([[token]], dtype=torch.long, device=self.device)], dim=1
                )
                nxt = torch.tensor([[token]], dtype=torch.long, device=self.device)
                fast = self._chunk_logits(nxt, prefix_len + step, cache)
                slow = self._chunk_logits(current, 0, DynamicCache())

        with self._hf_attention(), self._root_forward() as call:
            call(_run)
        bad = [r for r in rows if not r["ok"]]
        loose = [r for r in rows if r["max_abs_delta"] > atol]
        return {"rows": rows, "argmax_mismatch": len(bad), "over_atol": len(loose), "atol": atol,
                "passed": not bad}


def trim_generated(gen_ids: list[int]) -> list[int]:
    """把模型自己写的 `<sop>Hg Wg<eop> <boi>` 尾巴切掉，只留 think + final。

    为什么要切：出图分辨率必须由**调用方**说了算（要 2048² 就一定出 2048²），
    所以 shape 段一律由 `ThinkGenInferenceTransform.build_feature` 按请求网格重拼。
    不切就会出现两段 shape + 两个 `<boi>`，整条序列的布局与训练不符。
    用**第一个** `<sop>`：CoT 正文里出现这个特殊 token 的概率可以忽略，
    而万一模型连写两段，取第一个才不会把垃圾留在上下文里。
    """
    ids = list(int(t) for t in gen_ids)
    if SOP_ID in ids:
        return ids[: ids.index(SOP_ID)]
    while ids and ids[-1] in (BOI_ID, EOP_ID):
        ids.pop()
    return ids


def split_final(gen_ids: list[int], tokenizer) -> str:
    """取 `</think>` 之后那一段的文本 —— 也就是**真正喂给扩散段的图像描述**。

    为什么要有这个函数：think 段和 final 段在产物里是拼在一起的，而判读时关心的几乎总是
    final 段（2026-11-17 消融：改 final 段 5 个 token 能让字从乱变对；删掉整个 think 段
    图几乎不变 ⇒ think 段不决定画什么）。切分口径只放这一处，免得各处脚本各切一套。

    没有 `</think>` 时（链跑飞了、被截断）返回空串 —— 调用方据此就知道这条没有 final 段。
    """
    ids = [int(t) for t in gen_ids]
    if THINK_END_ID not in ids:
        return ""
    return tokenizer.decode(ids[ids.index(THINK_END_ID) + 1:], skip_special_tokens=False)


def parse_model_grid(gen_ids: list[int], tokenizer) -> tuple[int, int] | None:
    """从生成 token 里读模型**自己**写的 `<sop>Hg Wg<eop>`（只用于对账，不参与出图）。"""
    import re

    ids = list(int(t) for t in gen_ids)
    if SOP_ID not in ids:
        return None
    start = ids.index(SOP_ID)
    try:
        end = ids.index(EOP_ID, start + 1)
    except ValueError:
        end = min(len(ids), start + 12)
    numbers = re.findall(r"\d+", tokenizer.decode(ids[start + 1:end], skip_special_tokens=False))
    if len(numbers) < 2:
        return None
    grid_h, grid_w = int(numbers[0]), int(numbers[1])
    return (grid_h, grid_w) if 1 <= grid_h <= 512 and 1 <= grid_w <= 512 else None


__all__ = ["HYTextDecoder", "ARSamplingConfig", "apply_sampling", "trim_generated",
           "split_final", "parse_model_grid", "THINK_ID", "THINK_END_ID",
           "SOP_ID", "EOP_ID", "BOI_ID"]
