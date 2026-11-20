# ══════════ 搬自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion.py ══════════
import bisect
import itertools
import math
import os
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Literal, Optional, TypedDict, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from .configuration_hy import GlmImageVisionConfig
from .modeling_hy_vision import GlmImageVQVisionModel
from .modeling_hy_vision import VQPrePoolProjector
from .modeling_hy_vision import GlmImageVisionIBQ, GlmImageVisionResidualBlock
from .configuration_hy import Glm4MoeLiteConfig
from .modeling_hy_text import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteMLP,
    Glm4MoeLiteMoE,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteTopkRouter,
)
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops import fused_block_mask_attn
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as logging
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.base import is_torch_npu_available, is_triton_available
# 下面这些 config 类已搬到 configuration_hy.py，从那里取
from .configuration_hy import GlmImageMoeTransfusionConfig


logger = logging.get_logger(__name__)


# ============================================================================
#  Config
# ============================================================================


# （已搬到 configuration_hy.py，原 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion.py:47-145）
# class GlmImageMoeTransfusionConfig: ...


# ============================================================================
#  dataclass & typedDict
# ============================================================================


@dataclass
class GlmImageMoeTransfusionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ar_loss: Optional[torch.FloatTensor] = None
    diff_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    per_token_loss: Optional[torch.FloatTensor] = None
    # Raw diffusion velocity prediction [L_diff, C]. Training reads diff_loss and
    # ignores this; inference passes diff_target=None (no MSE) and consumes this
    # to drive the flow-match denoise loop (see tasks/transfusion/infer.py).
    diff_pred: Optional[torch.FloatTensor] = None


class TwoPassFaKwargs(TypedDict):
    item_cu: torch.IntTensor
    text_cu: torch.IntTensor
    image_cu: torch.IntTensor
    text_idx: torch.LongTensor
    image_idx: torch.LongTensor
    max_item_len: int
    max_text_len: int
    max_image_len: int


class BlockFusedFaKwargs(TypedDict):
    cu_seqlens: torch.IntTensor
    block_start: torch.IntTensor
    block_end: torch.IntTensor


# ============================================================================
# function components
# ============================================================================


def adarmsn_varlen(
    x: torch.Tensor,
    norm_module: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    seqlens: torch.IntTensor,
    fp32: bool = False,
) -> torch.Tensor:
    if fp32:
        return _adaln_apply(
            norm_module,
            x,
            scale.repeat_interleave(seqlens, dim=0),
            shift.repeat_interleave(seqlens, dim=0),
            True,
        )
    x_dtype = x.dtype
    x = norm_module(x)
    scale = scale.repeat_interleave(seqlens, dim=0)
    shift = shift.repeat_interleave(seqlens, dim=0)
    x = torch.addcmul(shift, x.to(shift.dtype), 1 + scale).to(x_dtype)
    return x


def apply_rotary_emb_fused(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("apply_rotary_emb_fused should be dynamic patch depend on device")


# ============================================================================
# module Components class
# ============================================================================


class DiffTimestepEmbedder(nn.Module):
    """Fourier embedding + 2-layer MLP (diffusion standard)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [L]  float
        # returns:   [L, hidden_size]
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [L, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [L, freq_dim]
        if self.freq_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        embedding = embedding.to(self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class DiffTimestepEmbedderFp32(DiffTimestepEmbedder):
    # register for mixed precision ignore
    pass


class SequentialFp32(nn.Sequential):
    # register for mixed precision ignore
    pass

def rms_norm_fp32_no_affine(x, eps):
    if is_torch_npu_available() and x.device.type == "npu":
        import torch_npu

        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _adaln_apply(norm: nn.Module, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, fp32: bool):
    if not fp32:
        return torch.addcmul(shift, norm(x), 1 + scale)
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = norm.variance_epsilon
    normed = rms_norm_fp32_no_affine(x.float(), eps)
    return torch.addcmul(shift.float(), normed, 1 + scale.float()).to(x.dtype)


# TODO(wwx): 思考一下 cache 怎么做
class GlmImageMoeTransfusionRotaryEmbedding(nn.Module):
    def __init__(self, config: GlmImageMoeTransfusionConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.text_config.max_position_embeddings
        self.original_max_seq_len = config.text_config.max_position_embeddings

        self.config = config

        self.rope_type = config.text_config.rope_parameters["rope_type"]
        assert self.rope_type == "default", "length expolonation of LLM is not suitable for MM model"
        # rope_init_fn: Callable = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config.text_config, device)
        self.mrope_section = config.text_config.rope_parameters["mrope_section"]
        self.mrope_layout = config.diff_rope_thw_layout
        self.cast_rope_emb_to_input_dtype = config.cast_rope_emb_to_input_dtype

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)

        if self.mrope_layout == "block":
            self.mrope_indexer = nn.Buffer(
                torch.arange(0, len(self.mrope_section), device=device).repeat_interleave(
                    torch.tensor(self.mrope_section, device=device)
                ),
                persistent=False,
            )
        elif self.mrope_layout == "interleave":
            hw_delta_t = self.mrope_section[1] - self.mrope_section[0]
            mrope_indexer = [1, 2] * hw_delta_t
            mrope_indexer = mrope_indexer + ([1, 2, 0] * self.mrope_section[0])
            self.mrope_indexer = nn.Buffer(torch.tensor(mrope_indexer, device=device), persistent=False)
        else:
            raise ValueError(f"Unknown rope_layout: {self.mrope_layout}")

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Glm4MoeLiteConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reassmbled_position_ids = position_ids.index_select(0, self.mrope_indexer).permute(1, 2, 0)

        inv_freq_expanded = self.inv_freq[None, None, :].float()
        reassmbled_position_ids = reassmbled_position_ids.float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = inv_freq_expanded.float() * reassmbled_position_ids.float()
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling  # e.g., shape = torch.Size([3, 1, 389, 64])
            sin = emb.sin() * self.attention_scaling  # e.g., shape = torch.Size([3, 1, 389, 64])

        if self.cast_rope_emb_to_input_dtype:
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        else:
            return cos, sin


class GlmImageMoeTransfusionAttention(Glm4MoeLiteAttention):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx):
        super().__init__(config, layer_idx)

        attn_forward_impl_dict = {
            "flash_attention_2": self._two_pass_mla_forward,
            "flash_attention_3": self._two_pass_mla_forward,
            "eager": self._eager_forward,
            "fa3_block": self._fa3_block_fused_forward,
        }
        if self.config._attn_implementation not in attn_forward_impl_dict:
            logger.warning_once(
                f"attn_implementation {self.config._attn_implementation} is not support, fallback to eager"
            )
        self.forward = attn_forward_impl_dict.get(config._attn_implementation, self._eager_forward)

    def _two_pass_mla_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        attn_implementation: str | None = None,
        **kwargs: Unpack[TwoPassFaKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # unpack two pass kwargs

        item_cu = kwargs["item_cu"]
        text_cu = kwargs["text_cu"]
        image_cu = kwargs["image_cu"]
        text_idx = kwargs["text_idx"]
        image_idx = kwargs["image_idx"]
        max_item_len = kwargs["max_item_len"]
        max_text_len = kwargs["max_text_len"]
        max_image_len = kwargs["max_image_len"]
        #
        bsz, q_len = hidden_states.shape[:-1]
        qk_head_dim = self.qk_head_dim
        v_head_dim = self.v_head_dim
        query_shape = (bsz, q_len, -1, qk_head_dim)
        key_shape = (bsz, q_len, -1, self.qk_nope_head_dim + v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, v_head_dim], dim=-1)
        k_rot = k_rot.view(bsz, q_len, 1, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot = apply_rotary_emb_fused(q_rot, cos, sin, self.config.rope_interleave)
        k_rot = apply_rotary_emb_fused(k_rot, cos, sin, self.config.rope_interleave)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # QKV and RoPE use BLHD in both paths; only the HF attention interface needs BHLD.
        query_states = torch.cat((q_pass, q_rot), dim=-1).transpose(1, 2)
        key_states = torch.cat((k_pass, k_rot), dim=-1).transpose(1, 2)

        pad = qk_head_dim - v_head_dim
        if pad > 0:
            value_states = F.pad(value_states, [0, pad])
        value_states = value_states.transpose(1, 2)

        # ── context parallel(手动 Ulysses)，与 _fa3_block_fused_forward 同一套约定 ──────────
        # attn_in 把 [chunk,H,D] 转成 [padded,H/cp,D]：每卡覆盖**完整序列**但只算 H/cp 个头。
        # text_idx / image_idx / *_cu 由主 forward 按完整序列算好；补齐 token 在序列末尾、不属于
        # 任何 item，两遍都不会选中它，这里直接切掉，输出处补零再换回去。
        _cp = kwargs.get("cp")
        if _cp is not None:
            assert bsz == 1
            _qkv = torch.cat((query_states, key_states, value_states), dim=-1).squeeze(0).transpose(0, 1)
            _qkv = _cp.attn_in(_qkv.contiguous()).narrow(0, 0, _cp.n)  # [n, H/cp, 3D]
            query_states, key_states, value_states = (
                t.transpose(0, 1).unsqueeze(0).contiguous() for t in _qkv.split(qk_head_dim, dim=-1)
            )
        attn_len = query_states.shape[2]

        n_heads = query_states.shape[1]
        implementation = attn_implementation or self.config._attn_implementation
        if implementation == "fa3_block":
            implementation = "flash_attention_2"
        flash = ALL_ATTENTION_FUNCTIONS[implementation]
        common = dict(
            attention_mask=None,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )
        # flash output layout is [B, S, H, D]; assemble per-row then scatter back.
        attn_out = query_states.new_zeros(bsz, attn_len, n_heads, qk_head_dim)

        # ── Causality is read from ``module.is_causal`` by the HF flash interface ──
        # The ``is_causal=`` kwarg below is NOT honored on this HF attention path, so
        # we TOGGLE ``self.is_causal`` around each pass (same mechanism the ARDiff
        # GlmImageMoeDiffAttention uses — it sets ``module.is_causal=False``). Without
        # this the image pass would inherit the attn module's default ``is_causal=True``
        # and run CAUSAL → image rows would only see EARLIER image tokens instead of
        # the full item (the bug). Saved/restored so the module default is untouched.
        _saved_is_causal = self.is_causal
        try:
            # pass C — text-only CAUSAL self-attn (q/k/v all restricted to text rows).
            # index_select on the token axis (dim 2) — output size == text_idx.numel(),
            # known on host → no device→host sync (cf. boolean-mask q[:, :, is_text, :]).
            if max_text_len > 0:
                self.is_causal = True
                out_t, _ = flash(
                    self,
                    query_states.index_select(2, text_idx),
                    key_states.index_select(2, text_idx),
                    value_states.index_select(2, text_idx),
                    cu_seq_lens_q=text_cu,
                    cu_seq_lens_k=text_cu,
                    max_length_q=max_text_len,
                    max_length_k=max_text_len,
                    is_causal=True,
                    **common,
                )  # [B, P_total, H, qk_hd]
                attn_out.index_copy_(1, text_idx, out_t)

            # pass F — image rows attend the FULL item (text + image), BIDIRECTIONAL.
            # q = image rows (cu_q = image_cu); k/v = full sequence (cu_k = item_cu).
            if max_image_len > 0:
                self.is_causal = False
                out_i, _ = flash(
                    self,
                    query_states.index_select(2, image_idx),
                    key_states,
                    value_states,
                    cu_seq_lens_q=image_cu,
                    cu_seq_lens_k=item_cu,
                    max_length_q=max_image_len,
                    max_length_k=max_item_len,
                    is_causal=False,
                    **common,
                )  # [B, D_total, H, qk_hd]
                attn_out.index_copy_(1, image_idx, out_i)
        finally:
            self.is_causal = _saved_is_causal

        if _cp is not None:
            _out = attn_out.squeeze(0)  # [n, H/cp, D]
            if _cp.pad:
                _out = F.pad(_out, (0, 0, 0, 0, 0, _cp.pad))  # 补齐行的输出记 0
            attn_out = _cp.attn_out(_out.contiguous()).unsqueeze(0)  # [1, chunk, H, D]

        if pad > 0:
            attn_out = attn_out[..., :v_head_dim]
        attn_out = attn_out.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_out)
        attn_weights = None  # attn_weights output is not support in fa mode
        return attn_output, attn_weights

    def _fa3_block_fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs: Unpack[BlockFusedFaKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cp = kwargs.get("cp")
        if (
            cp is not None and cp.size > 1 and self.q_lora_rank is not None
            and os.environ.get("VEOMNI_HY_CP_LATENT", "0") == "1"
        ):
            from .base import _eval_only_placeholder as _p_lazy
            forward_cp_latent = _p_lazy('forward_cp_latent', 'veomni.optimizations.hy.cp_latent，guard: cp.size > 1 且 VEOMNI_HY_CP_LATENT=1')

            if self.layer_idx == 0 and not getattr(self, "_hy_cp_latent_logged", False):
                logger.info_rank0("HY CP MLA: exchange normalized Q/KV latents; expand only the local attention heads")
                self._hy_cp_latent_logged = True
            return forward_cp_latent(self, hidden_states, position_embeddings, attention_mask, **kwargs)
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, seq_length, 1, self.qk_rope_head_dim)

        cos, sin = position_embeddings

        q_rot = apply_rotary_emb_fused(q_rot, cos, sin, self.config.rope_interleave)
        k_rot = apply_rotary_emb_fused(k_rot, cos, sin, self.config.rope_interleave)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        pad = self.qk_head_dim - self.v_head_dim
        if pad > 0:
            value_states = F.pad(value_states, [0, pad])

        # ── context parallel(手动 Ulysses)────────────────────────────────────
        # cp 由主 forward 放进 layer_kwargs，经 DecoderLayer 的 **kwargs 直通到这里。
        # attn_in 把 [chunk,H,D] 转成 [padded,H/cp,D]：每卡覆盖**完整序列**但只算 H/cp 个头。
        # 因此 cu_seqlens / blk_start / blk_end 必须是完整序列的边界（末尾补一个补齐 item），
        # 核看到的输入与不切时逐位相同，只是头数少了。
        _cp = kwargs.get("cp")
        _q, _k, _v = query_states.squeeze(0), key_states.squeeze(0), value_states.squeeze(0)
        # cu_seqlens 由主 forward 传进来时**已经**补齐过（pad_cu 只做一次）。
        # 不能在这里逐层调 pad_cu：blockmask 核把 meta 缓存在 cu 张量对象上、靠对象身份命中，
        # 而 pad_cu 在 pack 长度不是 cp 整数倍时每次都返回新张量，会让 47 层全部缓存失效，
        # 每层多一次 host 同步加 Python 循环。
        _cu = kwargs["cu_seqlens"]
        if _cp is not None:
            # 拼成一个张量做**一次** a2a，而不是 q/k/v 各做一次。
            # 拼接要多写一次 [chunk,H,3*D]、切开后还要 contiguous 再写一次，两趟约 500MB
            # 访存；三次独立 a2a 能省掉这两趟、搬运字节数完全相同。但实测拼接版更快
            # （2637.5 vs 2658.8 ms fwd+bwd）：NPU 上集合通信下发走 AI CPU，
            # 次数从 282 涨到 846 的开销超过了省下的拷贝。故保留拼接版。
            _qkv = _cp.attn_in(torch.cat((_q, _k, _v), dim=-1))
            _q, _k, _v = (t.contiguous() for t in _qkv.split(self.qk_head_dim, dim=-1))

        attn_output = fused_block_mask_attn(
            q=_q,
            k=_k,
            v=_v,
            cu_seqlens=_cu,
            blk_start=kwargs[
                "block_start"
            ],  # TODO(wwx): 还可以修改，里头第一次调用 build meta 的时候会有一个 tolist + create 大 tensor
            blk_end=kwargs["block_end"],
            max_seqlen=kwargs["max_seqlen"],
            scale=query_states.shape[-1] ** -0.5,
        )
        if _cp is not None:
            attn_output = _cp.attn_out(attn_output.contiguous())
        if pad > 0:
            attn_output = attn_output[..., : self.v_head_dim]
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_weights = None
        return attn_output, attn_weights

    def _eager_forward(self):
        pass


class GlmImageMoeTransfusionMLP(Glm4MoeLiteMLP):
    pass


class GlmImageMoeTransfusionMoE(Glm4MoeLiteMoE):
    pass


class GlmImageMoeTransfusionVQProjector(nn.Module):
    """VQ projector used by the VLM understanding path.

    This mirrors the upstream Tokenizer-CT projector shape for checkpoints where
    ``vq_use_quant_conv`` / ``vq_use_post_quant_conv`` / ``vq_use_post_conv`` are
    disabled (the default in the 512p transfusion ckpt), while keeping the old
    conv-backed path available for configs that request it.
    """

    def __init__(
        self,
        in_channels: int,
        codebook_size: int,
        codebook_dim: int,
        num_conv_layers: int = 2,
        num_groups: int = 32,
        pre_pool_factor: int = 1,
        pre_pool_type: str = "none",
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        use_post_conv: bool = False,
        **_,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.codebook_dim = codebook_dim
        self.pre_pool_factor = int(pre_pool_factor)
        self.use_quant_conv = use_quant_conv
        self.use_post_quant_conv = use_post_quant_conv
        self.use_post_conv = use_post_conv

        if self.pre_pool_factor > 1:
            if pre_pool_type != "projection":
                raise ValueError(f"Only pre_pool_type='projection' is supported, got {pre_pool_type!r}")
            self.pre_pool = VQPrePoolProjector(in_channels, pool_factor=self.pre_pool_factor)
            pre_pool_out_channels = in_channels
        else:
            self.pre_pool = None
            pre_pool_out_channels = in_channels

        if not use_quant_conv and pre_pool_out_channels != codebook_dim:
            raise ValueError(
                f"use_quant_conv=False requires pre_pool_out_channels ({pre_pool_out_channels}) "
                f"== codebook_dim ({codebook_dim})"
            )
        self.quant_conv = nn.Conv2d(pre_pool_out_channels, codebook_dim, kernel_size=1) if use_quant_conv else None
        self.quantize = GlmImageVisionIBQ(codebook_size, codebook_dim, l2_norm=True)
        self.post_quant_conv = nn.Conv2d(codebook_dim, in_channels, kernel_size=1) if use_post_quant_conv else None
        self.post_conv = (
            nn.Sequential(*[GlmImageVisionResidualBlock(in_channels, num_groups) for _ in range(num_conv_layers)])
            if use_post_conv
            else None
        )

    def _apply_quant_conv_flat(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_conv is None:
            return x
        return F.linear(x, self.quant_conv.weight.squeeze(-1).squeeze(-1), self.quant_conv.bias)

    def encode_unified_packed(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
        do_quantize: Union[bool, torch.Tensor],
    ) -> tuple[list[torch.Tensor] | torch.Tensor | None, torch.Tensor, torch.Tensor]:
        sum_n, channels = x.shape
        device = x.device
        k = self.pre_pool_factor
        B = grid_thw.shape[0]
        post_grids: list[tuple[int, int]] = []
        cursor = 0
        if self.pre_pool is not None:
            gather_blocks = []
            ky = torch.arange(k, device=device).view(1, 1, k, 1)
            kx = torch.arange(k, device=device).view(1, 1, 1, k)
            for i in range(B):
                t, h, w = (int(v.item()) for v in grid_thw[i])  # TODO(wwx): 这里有同步点，可以做优化
                assert t == 1, "encode_unified_packed: t!=1 not supported"
                assert h % k == 0 and w % k == 0, f"grid {(h, w)} not divisible by pre_pool_factor={k}"
                ho, wo = h // k, w // k
                post_grids.append((ho, wo))
                row = k * torch.arange(ho, device=device).view(ho, 1, 1, 1) + ky
                col = k * torch.arange(wo, device=device).view(1, wo, 1, 1) + kx
                gather_blocks.append((row * w + col).reshape(ho * wo, k * k) + cursor)
                cursor += t * h * w
            assert cursor == sum_n, f"grid token sum {cursor} != packed seq len {sum_n}"
            gathered = x[torch.cat(gather_blocks, dim=0)]
            proj = self.pre_pool.proj
            w_lin = proj.weight.permute(2, 3, 1, 0).reshape(k * k * channels, -1)
            x = gathered.reshape(-1, k * k * channels) @ w_lin + proj.bias
        else:
            for i in range(B):
                t, h, w = (int(v.item()) for v in grid_thw[i])
                assert t == 1, "encode_unified_packed: t!=1 not supported"
                post_grids.append((h, w))
                cursor += t * h * w
            assert cursor == sum_n, f"grid token sum {cursor} != packed seq len {sum_n}"

        z = self._apply_quant_conv_flat(x)
        if isinstance(do_quantize, torch.Tensor):
            if bool(do_quantize.all().item()):
                do_quantize = True
            elif not bool(do_quantize.any().item()):
                do_quantize = False
        if do_quantize is True:
            z_2d = z.t().unsqueeze(0).unsqueeze(-1).contiguous()
            z_q_2d, indices_2d, commit_loss = self.quantize(z_2d, return_commit_loss=True)
            return indices_2d.reshape(-1), z_q_2d.squeeze(-1).squeeze(0).t().contiguous(), commit_loss
        z = F.normalize(z, p=2, dim=-1) if self.quantize.l2_norm else z
        zero = torch.zeros((), device=z.device, dtype=z.dtype)
        return None, z + (self.quantize.embedding.weight.sum() * 0.0).to(z.dtype), zero


class GlmImageMoeTransfusionDecoderLayer(Glm4MoeLiteDecoderLayer):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmImageMoeTransfusionAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = GlmImageMoeTransfusionMoE(config)
        else:
            self.mlp = GlmImageMoeTransfusionMLP(config)

        self.input_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected. MoE blocks return (output, aux_loss); dense blocks
        # return just output. Aux loss is propagated via the layer return so
        # it stays inside the gradient-checkpointing boundary.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            hidden_states, aux_loss = mlp_out
        else:
            hidden_states, aux_loss = mlp_out, None
        hidden_states = residual + hidden_states
        return hidden_states, aux_loss


# ============================================================================
# Pretrained base class


class GlmImageMoeTransfusionPreTrainedModel(PreTrainedModel):
    config: GlmImageMoeTransfusionConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    # TODO(wwx): 这样会有很小的通信组 DiffTimestepEmbedderFp32 和 SequentialFp32，可以考虑一下怎么处理，8卡实测没影响
    _no_split_modules = ["GlmImageMoeTransfusionDecoderLayer", "DiffTimestepEmbedderFp32", "SequentialFp32"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_attention_backend = True
    _keep_in_fp32_modules_strict = []

    def get_parallel_plan(self):
        """EP 计划: 专家权重按第 0 维(专家维)切到各 EP rank。

        Glm4MoeLiteExperts 把专家权重存成三维 nn.Parameter:
        gate_proj / up_proj [E, I, H]、down_proj [E, H, I]，
        _moe_implementation="sonic" 时是合并的 gate_up_proj [E, 2I, H]。
        与 qwen3_moe / deepseek_v3 的命名一致，切法也一致。
        shared_experts 是普通 Linear，不参与 EP，走常规 FSDP。
        GlmImageMoeTransfusion 复用同一套 FQN(model.layers.*.mlp.experts.*)。
        """
        from torch.distributed._tensor import Shard
        from .base import _eval_only_placeholder as _p_lazy
        ParallelPlan = _p_lazy('ParallelPlan', 'veomni.distributed.parallel_plan，guard: parallel_plan 属性，只给 hp_fsdp2 分片用')
        
        ep_plan = {
            "model.layers.*.mlp.experts.gate_proj": Shard(0),
            "model.layers.*.mlp.experts.up_proj": Shard(0),
            "model.layers.*.mlp.experts.down_proj": Shard(0),
            "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        }
        return ParallelPlan(ep_plan=ep_plan)


# ============================================================================
#  Main model class
# ============================================================================


# TODO(wwx): 看看需不需要补 init, veomni/models/cogview/modeling_glm_image_moe_omni.py 是有的


class GlmImageMoeTransfusionBackbone(GlmImageMoeTransfusionPreTrainedModel):
    def __init__(self, config: GlmImageMoeTransfusionConfig):
        super().__init__(config)
        self.padding_idx = config.text_config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.text_config.vocab_size, config.text_config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GlmImageMoeTransfusionDecoderLayer(config.text_config, layer_idx)
                for layer_idx in range(config.text_config.num_hidden_layers)
            ]
        )
        self.norm = Glm4MoeLiteRMSNorm(config.text_config.hidden_size, config.text_config.rms_norm_eps)
        self.rotary_emb = GlmImageMoeTransfusionRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        if config.init_vq:
            backbone_attn_implementation = config._attn_implementation
            config.vq_vision_config._attn_implementation = (
                backbone_attn_implementation if backbone_attn_implementation != "fa3_block" else "flash_attention_2"
            )  # TODO(wwx): hardcode 了 fa2，不知道换 fa3 是啥样，待测
            self.vq_visual = GlmImageVQVisionModel(config.vq_vision_config)
            # MARK: 在 transfusion 里其实 vq 都没用，就是一个 downsample 2 的 projector
            self.vq_projector = GlmImageMoeTransfusionVQProjector(
                in_channels=config.vq_vision_config.hidden_size,
                codebook_size=config.vq_vision_config.vq_codebook_size,
                codebook_dim=config.vq_vision_config.vq_codebook_dim,
                num_conv_layers=config.vq_vision_config.vq_num_conv_layers,
                pre_pool_factor=config.vq_vision_config.vq_pre_pool_factor,
                pre_pool_type=config.vq_vision_config.vq_pre_pool_type,
                use_quant_conv=config.vq_vision_config.vq_use_post_quant_conv,
                use_post_conv=config.vq_vision_config.vq_use_post_conv,
            )
        else:
            self.vq_visual = None
            self.vq_projector = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None = None,
        vae_grid_thw: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        image_grid_is_merged: bool = False,
    ) -> torch.Tensor:
        """Use one VAE/ViT spatial frame, with ViT on the next temporal plane."""
        from .modeling_hy_extras import image_position_ids
        batch_size, length = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        position_ids = torch.zeros((3, batch_size, length), device=device, dtype=dtype)
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(vae_grid_thw) if vae_grid_thw is not None else None,
        }
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.config.image_token_id] = 1
        mm_token_type_ids[input_ids == self.config.vae_token_id] = 2

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            reference = None
            for group_index, (modality_type, start_index, end_index) in enumerate(input_type_group):
                # text == 0
                if modality_type == 0:
                    text_len = end_index - start_index
                    llm_pos_ids_list.append(
                        torch.arange(current_pos, current_pos + text_len, device=device).expand(3, -1)
                    )
                    current_pos += text_len
                    reference = None
                # vit == 1
                elif modality_type == 1:
                    grid_thw = next(grid_iters[modality_type])
                    merge = 1 if image_grid_is_merged else self.config.vq_vision_config.vq_pre_pool_factor
                    vit_grid = [int(grid_thw[0]), int(grid_thw[1]) // merge, int(grid_thw[2]) // merge]
                    anchor, vae_grid = reference if reference is not None else (current_pos, None)
                    vision_pos = image_position_ids(
                        vit_grid, anchor, spatial_grid=vae_grid,
                        temporal_offset=1 if reference is not None else 0, device=device,
                    )
                    llm_pos_ids_list.append(vision_pos)
                    current_pos = anchor + max(int(v) for v in (vae_grid if vae_grid is not None else vit_grid))
                    reference = None
                # vae == 2
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_pos = image_position_ids(grid_thw, current_pos, device=device)
                    llm_pos_ids_list.append(vision_pos)
                    following = input_type_group[group_index + 1][0] if group_index + 1 < len(input_type_group) else None
                    reference = (current_pos, grid_thw) if following == 1 else None
                    current_pos += max(int(v) for v in grid_thw)

                if llm_pos_ids_list[-1].shape[-1] != end_index - start_index:
                    raise ValueError("Image token count does not match its VAE/ViT grid")

            llm_positions = torch.cat(llm_pos_ids_list, dim=-1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions

        return position_ids

    # TODO(wwx): 还是有些 tricky，找个简单的方法
    def get_position_id_func(self):
        fake_model = SimpleNamespace(config=self.config)
        return partial(GlmImageMoeTransfusionBackbone.get_rope_index, fake_model)

    def collect_global_tokens_per_expert(
        self, group=None, add_vision_tokens: bool = False, coalesced: bool = False
    ) -> list[torch.Tensor]:
        global_tokens_per_expert_list = []

        if coalesced:
            for layer in self.layers:
                if isinstance(layer.mlp, GlmImageMoeTransfusionMoE):
                    global_tokens_per_expert_list.append(layer.mlp.gate.local_tokens_per_expert.clone())
            if add_vision_tokens:
                for layer in self.layers:
                    if isinstance(layer.mlp, GlmImageMoeTransfusionMoE):
                        global_tokens_per_expert_list.append(layer.mlp.gate.local_vision_tokens_per_expert.clone())
            dist.all_reduce_coalesced(global_tokens_per_expert_list, group=group)
        else:
            for layer in self.layers:
                if isinstance(layer.mlp, GlmImageMoeTransfusionMoE):
                    global_tokens_per_expert = layer.mlp.gate.local_tokens_per_expert.clone()
                    dist.all_reduce(global_tokens_per_expert, op=dist.ReduceOp.SUM)
                    global_tokens_per_expert_list.append(global_tokens_per_expert)
            if add_vision_tokens:
                for layer in self.layers:
                    if isinstance(layer.mlp, GlmImageMoeTransfusionMoE):
                        global_tokens_per_expert = layer.mlp.gate.local_vision_tokens_per_expert.clone()
                        dist.all_reduce(global_tokens_per_expert, op=dist.ReduceOp.SUM)
                        global_tokens_per_expert_list.append(global_tokens_per_expert)

        return global_tokens_per_expert_list

    def update_router_bias(self, global_tokens_per_expert_list: list[torch.Tensor]) -> None:
        sparse_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer.mlp, GlmImageMoeTransfusionMoE):
                global_tokens_per_expert_list[sparse_layer_idx]
                layer.mlp.gate.update_router_bias(global_tokens_per_expert_list[sparse_layer_idx])
                sparse_layer_idx += 1

    def zero_local_tokens_per_expert(self) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, GlmImageMoeTransfusionMoE):
                layer.mlp.gate.zero_local_tokens_per_expert()


class GlmImageMoeTransfusionForMMGeneration(GlmImageMoeTransfusionPreTrainedModel, GenerationMixin):
    """Transfusion implement model on the MoE backbone."""

    config_class = GlmImageMoeTransfusionConfig
    backbone_class = GlmImageMoeTransfusionBackbone
    lm_head_class = nn.Linear
    _checkpoint_conversion_mapping = {
        r"^model\.language_model\.": "model.",
        r"^visual\.": "model.vq_visual.",
        r"^model\.vq_visual\.": "model.vq_visual.",
        r"^model\.vq_projector\.": "model.vq_projector.",
    }

    def get_ignore_modules_in_mixed_precision(self) -> tuple[nn.Module, ...]:
        """Module classes that must NOT inherit FSDP2 mp_policy(param_dtype=bf16).

        torch_parallelize wraps each returned class in its own ``fully_shard``
        call without mp_policy, so params stay fp32 for storage AND the forward
        all-gather. Without this the MoE router's ``self.weight`` is served as
        bf16 at forward time and its internal ``.type(torch.float32)`` only
        upcasts already-rounded values — routing would effectively run at bf16
        weight precision.

        adaln_fp32 also keeps timestep/adaLN forward weights in FP32, extending
        the existing policy to the projections in additional diffusion blocks.
        """
        classes = (Glm4MoeLiteTopkRouter,)
        if self.config.adaln_fp32:
            classes += (SequentialFp32, DiffTimestepEmbedderFp32)
        return classes

    def __init__(self, config: GlmImageMoeTransfusionConfig):
        super().__init__(config)
        # 备选是用 self.language_model, 但是 follow qwen25vl & hy image3 的实现，叫 model，方便组合权重和 load from scratch
        self.model = self.backbone_class(config)
        self.lm_head = self.lm_head_class(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        if config.init_vq:
            self.vq_to_llm_proj = nn.Linear(
                config.vq_vision_config.vq_codebook_dim, config.text_config.hidden_size, bias=True
            )
        else:
            self.vq_to_llm_proj = None

        diff_in_out = config.diff_vae_latent_dim * config.diff_patch_size**2
        self.diff_input_proj = nn.Linear(diff_in_out, config.text_config.hidden_size, bias=True)
        diff_time_embed_cls = DiffTimestepEmbedderFp32 if config.adaln_fp32 else DiffTimestepEmbedder
        self.diff_time_embed = diff_time_embed_cls(
            hidden_size=config.diff_time_emb_dim, freq_dim=config.diff_timestep_embed_dim
        )

        # adaLN mode only modulates the diff ENTRY (after diff_input_proj) and
        # the diff EXIT (before diff_output_proj); the shared AR backbone is
        # untouched. In adaLN the head/entry norms carry no learnable affine —
        # scale/shift come from the timestep adaLN
        if config.diff_time_embed_mode == "adaln":
            self.diff_norm = Glm4MoeLiteRMSNorm(
                config.text_config.hidden_size,
                eps=config.text_config.rms_norm_eps,
                elementwise_affine=False,
            )
            self.diff_input_norm = Glm4MoeLiteRMSNorm(
                config.text_config.hidden_size,
                eps=config.text_config.rms_norm_eps,
                elementwise_affine=False,
            )
            # TODO(wwx): 这里可以做一下 silu fuse 到 time embed 的 output
            sequentail_cls = SequentialFp32 if config.adaln_fp32 else nn.Sequential
            self.diff_adaln_in = sequentail_cls(
                nn.SiLU(), nn.Linear(config.diff_time_emb_dim, 2 * config.text_config.hidden_size, bias=True)
            )
            self.diff_adaln_out = sequentail_cls(
                nn.SiLU(), nn.Linear(config.diff_time_emb_dim, 2 * config.text_config.hidden_size, bias=True)
            )
            # TODO(wwx): 移动到 init 里
            nn.init.zeros_(self.diff_adaln_in[1].weight)
            nn.init.zeros_(self.diff_adaln_in[1].bias)
            nn.init.zeros_(self.diff_adaln_out[1].weight)
            nn.init.zeros_(self.diff_adaln_out[1].bias)
        else:
            self.diff_norm = Glm4MoeLiteRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
            self.diff_input_norm = None
            self.diff_adaln_in = None
            self.diff_adaln_out = None
        self.diff_output_proj = nn.Linear(config.text_config.hidden_size, diff_in_out, bias=True)

        # TODO(wwx): 移动到 init 里
        if getattr(self, "diff_enabled", False) and getattr(self, "_use_adaln", False):
            nn.init.zeros_(self.diff_adaln_in[1].weight)
            nn.init.zeros_(self.diff_adaln_in[1].bias)
            nn.init.zeros_(self.diff_adaln_out[1].weight)
            nn.init.zeros_(self.diff_adaln_out[1].bias)

        self.gradient_checkpointing = False
        self.post_init()

    # TODO(wwx)
    # def _init_weights(self, module) -> None:
    #     pass

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def vq_encode_to_codebook(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        do_quantize: Union[bool, torch.Tensor],
    ) -> tuple[list[torch.Tensor] | torch.Tensor | None, torch.FloatTensor, torch.Tensor]:
        """Run the Tokenizer-CT ViT + VQ projector and return codebook-space features."""
        assert self.model.vq_visual is not None, "vq_visual encoder not initialized"
        assert self.model.vq_projector is not None, "vq_projector not initialized"
        encoder = self.model.vq_visual
        projector = self.model.vq_projector

        target_dtype = torch.bfloat16 if encoder.dtype == torch.float32 else encoder.dtype
        image_features = encoder(pixel_values.to(target_dtype), grid_thw=image_grid_thw)
        return projector.encode_unified_packed(image_features, image_grid_thw, do_quantize=do_quantize)

    def vq_visual_to_inputs_embeds(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        do_quantize: Union[bool, torch.Tensor],
    ) -> tuple[list[torch.Tensor] | torch.Tensor | None, torch.FloatTensor, torch.Tensor]:
        assert self.vq_to_llm_proj is not None, "vq_to_llm_proj not initialized (init_vq=False?)"
        indices, z_q_packed, commit_loss = self.vq_encode_to_codebook(pixel_values, image_grid_thw, do_quantize)
        embeds = self.vq_to_llm_proj(z_q_packed.to(self.vq_to_llm_proj.weight.dtype))
        return indices, embeds, commit_loss

    def vq_visual_to_inputs_embeds_cp_local(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """``vq_visual_to_inputs_embeds(do_quantize=False)`` 的 CP 版：只编码与本 rank 序列分片相交的图。

        编码发生在 ``transfusion_cp_shard`` 之前，不切的话 cp 个 rank 编码同一批图。而编码结果
        masked_scatter 进完整 pack 后几行就被切到本 rank 那一段，落在别的 rank 分片里的视觉行写
        什么都不影响前向；反向时本 rank 也只对自己分片内的视觉 token 产生梯度。所以按**序列位置**
        定属主、其余行填零，与全部编码等价且**不需要任何集合通信**：各 rank 的 ViT 偏梯度由 FSDP
        本来就要走的 reduce-scatter 求和。跨分片边界的图被相邻两个 rank 各编一次，各自只贡献自己
        那部分行的梯度，求和仍然精确。按图 balanced_assign + gather 收回的方案实测只有 −0.33%，
        省下的算力被集合通信吃掉了；这个零通信版在 ViT 参与训练时实测 −2.82%。
        """
        from .modeling_hy_extras import transfusion_cp_shard

        _cp = transfusion_cp_shard(image_mask.shape[1])
        if _cp is None or _cp.size == 1 or image_grid_thw.shape[0] < 2:
            return self.vq_visual_to_inputs_embeds(pixel_values, image_grid_thw, do_quantize=False)[1]

        projector = self.model.vq_projector
        k = projector.pre_pool_factor if projector.pre_pool is not None else 1
        in_rows = [t * h * w for t, h, w in image_grid_thw.tolist()]
        in_cu = [0, *itertools.accumulate(in_rows)]
        out_cu = [0, *itertools.accumulate(r // (k * k) for r in in_rows)]
        # 本 rank 分片 [start, end) 之前 / 之内的视觉 token 数 = 它在 visual_embed 里的行区间 [r0, r1)。
        m = image_mask.view(-1)
        r0, r1, n_vis = torch.stack((m[: _cp.start].sum(), m[: _cp.end].sum(), m.sum())).tolist()
        if n_vis != out_cu[-1]:  # 占位 token 数与编码行数对不上：不猜，走原路径
            return self.vq_visual_to_inputs_embeds(pixel_values, image_grid_thw, do_quantize=False)[1]

        if r1 > r0:
            # masked_scatter 按序填，visual_embed 的行与序列位置同序，所以相交的图是连续的一段 [i0, i1)。
            i0 = bisect.bisect_right(out_cu, r0) - 1
            i1 = bisect.bisect_left(out_cu, r1)
            lo, hi = out_cu[i0], out_cu[i1]
        else:
            # 分片里没有视觉 token：仍编码最小的一张图再丢掉，保持组内 FSDP all-gather / reduce-scatter 一致。
            i0 = min(range(len(in_rows)), key=in_rows.__getitem__)
            i1 = i0 + 1
            lo = hi = 0
        _, embeds, _ = self.vq_visual_to_inputs_embeds(
            pixel_values[in_cu[i0] : in_cu[i1]], image_grid_thw[i0:i1], do_quantize=False
        )
        # 区间外补零行：它们只会落进别的 rank 的分片，随即被 transfusion_cp_shard 切掉。
        return F.pad(embeds[: hi - lo], (0, 0, lo, out_cu[-1] - hi))

    def prepare_two_pass_mla_kwargs(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        block_range: torch.IntTensor | None,
    ) -> TwoPassFaKwargs:
        is_image = (input_ids == self.config.vae_token_id).squeeze(0)
        is_text = ~is_image
        text_idx: torch.LongTensor = is_text.nonzero(as_tuple=True)[0]
        image_idx: torch.LongTensor = is_image.nonzero(as_tuple=True)[0]

        item_cu = cu_seqlens
        n_item = item_cu.shape[0] - 1
        n_image = block_range.shape[0] if block_range is not None else 0
        assert n_image <= n_item, "two pass call is only support, t2i & edit call, not support multiple image input"
        item_lens = item_cu[1:] - item_cu[:-1]
        if block_range is not None:
            image_lens_no_pad = block_range[:, 1] - block_range[:, 0]
            image_item_idx = torch.searchsorted(
                cu_seqlens[1:],
                block_range[:, 0].contiguous(),
                right=True,
            )
            image_lens = torch.zeros(
                n_item,
                dtype=image_lens_no_pad.dtype,
                device=image_lens_no_pad.device,
            )
            image_lens.scatter_(
                0,
                image_item_idx,
                image_lens_no_pad,
            )
        else:
            if is_image.any():
                starts = is_image & ~F.pad(is_image[:-1], (1, 0))
                ends = is_image & ~F.pad(is_image[1:], (0, 1))
                start_idx = starts.nonzero(as_tuple=True)[0]
                end_idx = ends.nonzero(as_tuple=True)[0]
                image_lens = end_idx - start_idx + 1
            else:
                image_lens = torch.empty(0, dtype=torch.long, device=is_image.device)

        text_lens = item_lens - image_lens

        def _to_cu(lens) -> torch.IntTensor:
            N = lens.shape[-1]
            device = lens.device
            c = torch.zeros(N + 1, dtype=torch.int32, device=device)
            c[1:] = lens.cumsum(0).to(dtype=torch.int32)
            return c

        text_cu = _to_cu(text_lens)
        image_cu = _to_cu(image_lens)
        max_item_len: int = item_lens.max().item()
        max_image_len: int = image_lens.max().item()
        max_text_len: int = text_lens.max().item()

        return {
            "item_cu": item_cu,
            "text_idx": text_idx,
            "image_idx": image_idx,
            "max_item_len": max_item_len,
            "text_cu": text_cu,
            "image_cu": image_cu,
            "max_text_len": max_text_len,
            "max_image_len": max_image_len,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # === vit inputs ==
        pixel_values: Optional[torch.FloatTensor] = None,  # [L_patch, 3 * patch * patch]
        image_grid_thw: Optional[torch.LongTensor] = None,  # [N_vit, 3]
        # === diff inputs ===
        vae_noised_latents: Optional[torch.Tensor] = None,  # [L_diff, C] = x_t
        diff_timesteps: Optional[torch.Tensor] = None,  # [N_diff]
        diff_target: Optional[torch.Tensor] = None,  # [L_diff, C] v-pred target / x0(pixel)
        diff_flow_sigmas: Optional[torch.Tensor] = None,  # [N_diff] flow σ∈[0,1](pixel x0-pred v-loss)
        diff_grid_thw: Optional[torch.LongTensor] = None,  # [N_diff, 3]
        is_ref_vae: Optional[torch.BoolTensor] = None,  # [N_diff]
        # === aux_packing parameters
        cu_seqlens: torch.IntTensor | None = None,  # [N_items + 1]
        block_range: torch.IntTensor | None = None,  # [N_block, 2]
        max_seqlen: int | None = None,
        # ==== context parallel
        vit_cp_shard: bool = False,  # 开 CP 时 ViT 只编码与本 rank 分片相交的图，见 vq_visual_to_inputs_embeds_cp_local
        # ==== loss parameters
        return_per_token_loss: bool = False,
        **kwargs,
    ) -> GlmImageMoeTransfusionOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")

        # -- 1. AR text prep  ---------------------------------
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)  # [1, L, H]
        else:
            inputs_embeds = inputs_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention")
            )
            position_ids = self.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                vae_grid_thw=diff_grid_thw,
                attention_mask=attention_mask_tensor,
            )  # type: torch.LongTensor
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            logger.warning_once("use legacy position_ids with modality type, check your transform")
            position_ids = position_ids[:3]  # drop text_position_ids row

        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)

        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, input_ids.shape[1]], device=self.device, dtype=torch.int32)

        # -- 2. VIT input prep ----------------------------------
        if pixel_values is not None and pixel_values.shape[0] > 0:
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            if vit_cp_shard:
                visual_embed = self.vq_visual_to_inputs_embeds_cp_local(pixel_values, image_grid_thw, image_mask)
            else:
                _, visual_embed, _ = self.vq_visual_to_inputs_embeds(
                    pixel_values,
                    image_grid_thw,
                    do_quantize=False,
                )
            # assert image_mask.sum() == visual_embed.shape[0], f"{image_mask.sum()=} != {visual_embed.shape=}"  # MARK: 这个 assert 有同步点，debug 的时候开
            image_rows = kwargs.get("_veomni_image_rows")
            if image_rows is None:
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, visual_embed.to(inputs_embeds.dtype))
            else:
                from .modeling_hy_extras import scatter_token_rows

                inputs_embeds = scatter_token_rows(
                    inputs_embeds, image_mask, visual_embed.to(inputs_embeds.dtype), image_rows
                )
        elif self.training:  # MARK: 或者 parallel_state.fsdp_enabled?
            # dummy forward # TODO(wwx): dummy data 可以持久化
            fake_pixel_values = torch.randn(
                (4, 3 * (self.config.vq_vision_config.patch_size**2)), dtype=self.dtype, device=self.device
            )
            fake_grid_thw = torch.tensor([[1, 2, 2]], device=self.device, dtype=torch.int32)
            _, visual_embed, _ = self.vq_visual_to_inputs_embeds(
                fake_pixel_values,
                fake_grid_thw,
                do_quantize=False,
            )
            inputs_embeds = inputs_embeds + visual_embed.mean() * 0.0

        # -- 3. Diff input prep ----------------------

        if vae_noised_latents is not None and vae_noised_latents.shape[0] > 0:
            vae_proj = self.diff_input_proj(vae_noised_latents.to(self.diff_input_proj.weight.dtype))  # [L_diff, H]
            diff_seq_lens = diff_grid_thw.prod(dim=-1)

            if self.config.diff_time_embed_mode == "adaln":
                t_emb_item = self.diff_time_embed(diff_timesteps)  # [N_diff, H]
                sc_in, sh_in = self.diff_adaln_in(t_emb_item).chunk(2, dim=-1)
                vae_proj = adarmsn_varlen(
                    vae_proj, self.diff_input_norm, scale=sc_in, shift=sh_in,
                    seqlens=diff_seq_lens, fp32=self.config.adaln_fp32,
                )
            elif self.config.diff_time_embed_mode == "add":  # add
                ts_per_tok = diff_timesteps.repeat_interleave(diff_seq_lens, dim=0)
                t_embed = self.diff_time_embed(ts_per_tok).to(vae_proj.dtype)
                vae_proj = vae_proj + t_embed  # [L_diff, H]
            else:
                raise ValueError(f"Unknown diff_time_embed_mode: {self.config.diff_time_embed_mode}")

            vae_mask = (input_ids == self.config.vae_token_id).unsqueeze(-1)
            # assert vae_mask.sum() == vae_proj.shape[0], f"{vae_mask.sum()=} != {vae_proj.shape=}"  # MARK: 这个 assert 有同步点，debug 的时候开
            vae_rows = kwargs.get("_veomni_vae_rows")
            if vae_rows is None:
                inputs_embeds = inputs_embeds.masked_scatter(vae_mask, vae_proj.to(inputs_embeds.dtype))
            else:
                from .modeling_hy_extras import scatter_token_rows

                inputs_embeds = scatter_token_rows(inputs_embeds, vae_mask, vae_proj.to(inputs_embeds.dtype), vae_rows)

        elif self.training:  # MARK: 或者使用 parallel_state.fsdp_enabled
            # dummy forward # TODO(wwx): dummy data 可以持久化
            fake_vae_noised_latent = torch.randn(
                (1, self.config.diff_vae_latent_dim * (self.config.diff_patch_size**2)),
                device=self.device,
                dtype=self.dtype,
            )
            fake_diff_timesteps = torch.rand((1,), device=self.device)
            vae_proj = self.diff_input_proj(fake_vae_noised_latent.to(self.diff_input_proj.weight.dtype))
            if self.config.diff_time_embed_mode == "adaln":
                t_emb_item = self.diff_time_embed(fake_diff_timesteps)
                sc_in, sh_in = self.diff_adaln_in(t_emb_item).chunk(2, dim=-1)
                vae_proj = _adaln_apply(
                    self.diff_input_norm, vae_proj, sc_in, sh_in, self.config.adaln_fp32
                )
            elif self.config.diff_time_embed_mode == "add":
                fake_vae_noised_latent = torch.randn(
                    (1, self.config.diff_vae_latent_dim * (self.config.diff_patch_size**2)),
                    device=self.device,
                    dtype=self.dtype,
                )
                fake_diff_timesteps = torch.rand((1,), device=self.device)
                t_embed = self.diff_time_embed(fake_diff_timesteps).to(vae_proj.dtype)
                vae_proj = (vae_proj + t_embed).unsqueeze(0)
            inputs_embeds = inputs_embeds + 0.0 * vae_proj.mean()

        is_vae = (input_ids == self.config.vae_token_id).squeeze(0)
        vae_idx = is_vae.nonzero(as_tuple=True)[0]

        # ── Context parallel：把骨干的序列切到 SP 组各 rank ──────────────────
        # 只切骨干（几乎全部算力都在这里）。头部与 loss 见后面第 5 节。
        # 关键不变量：注意力用的 cu_seqlens / blk_start / blk_end 始终是**完整序列**的
        # 边界，切分只体现在 hidden_states 与 rope 上；每卡在注意力里覆盖全序列但只算
        # H/cp 个头（见 _fa3_block_fused_forward / _two_pass_mla_forward 里的 attn_in / attn_out）。
        from .modeling_hy_extras import transfusion_cp_shard

        _cp = transfusion_cp_shard(inputs_embeds.shape[1])
        _vision_mask_stats = is_vae
        _token_mask_stats = None
        if _cp is not None:
            if self.config._attn_implementation not in ("fa3_block", "flash_attention_2", "flash_attention_3"):
                raise NotImplementedError(
                    "transfusion CP 只实现了 attn_implementation='fa3_block'（blockmask 核）"
                    f"和 'flash_attention_2'/'flash_attention_3'（两遍法），"
                    f"当前是 {self.config._attn_implementation!r}。"
                )
            # 数据闸门：同一 SP 组必须共用同一个 pack。若数据分片不是按 dp_rank 走，
            # 组内各 rank 的序列长度会不同，attn_in 的 all-to-all 形状随即对不上，
            # 表现为难以定位的 HCCL 挂起/超时。这里用 MIN/MAX 提前拦成明确报错。
            _n_probe = torch.tensor(
                [inputs_embeds.shape[1], -inputs_embeds.shape[1]],
                device=inputs_embeds.device,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(_n_probe, op=torch.distributed.ReduceOp.MAX, group=_cp.group)
            if int(_n_probe[0]) != -int(_n_probe[1]):
                raise RuntimeError(
                    f"CP 数据闸门失败：同一 SP 组内 pack 长度不一致（max={int(_n_probe[0])}, "
                    f"min={-int(_n_probe[1])}）。同组各 rank 必须拿到同一个 pack —— "
                    f"请检查 dataloader 是否按 parallel_state.dp_rank/dp_size 分片。"
                )
            inputs_embeds = _cp.local(inputs_embeds, dim=1)
            _cos, _sin = position_embeddings
            # rope 的 cos/sin 最后一维是 head_dim，序列在**倒数第二维**；
            # 用 -2 对 [1, L, D] 和 [3, 1, L, D] 两种布局都成立。
            position_embeddings = (_cp.local(_cos, dim=-2), _cp.local(_sin, dim=-2))
            _vision_mask_stats = _cp.local(is_vae, dim=0, value=False)
            _token_mask_stats = _cp.valid_mask(inputs_embeds.device)

        # Classification belongs to the model's token layout; consumers are shared.
        _router_classes = None
        if any(getattr(getattr(layer.mlp, "gate", None), "_stats_enabled", False) for layer in self.model.layers):
            _router_classes = torch.zeros_like(input_ids.reshape(-1), dtype=torch.long)
            _router_classes.masked_fill_(input_ids.reshape(-1) == self.config.image_token_id, 1)
            _router_classes.masked_fill_(is_vae, 2)
            if is_ref_vae is not None and diff_grid_thw is not None:
                _refs = torch.repeat_interleave(is_ref_vae.to(torch.bool), diff_grid_thw.prod(-1),
                                               output_size=vae_idx.numel())
                _router_classes.index_copy_(0, vae_idx, torch.where(_refs, 1, 2).to(torch.long))
            if _cp is not None:
                _router_classes = _cp.local(_router_classes, dim=0, value=-1)

        # ── Vision-token routing monitor / 路由统计排除 CP 补齐 token ───────
        for _lyr in self.model.layers:
            _mlp = getattr(_lyr, "mlp", None)
            if _mlp is not None and hasattr(_mlp, "gate"):  # MoE block only
                _mlp.gate._vision_mask_for_stats = _vision_mask_stats
                _mlp.gate._token_mask_for_stats = _token_mask_stats
                _mlp.gate._router_token_class_local = _router_classes

        # -- 4. Backbone — one call per layer over the unified sequence -----
        hidden_states = inputs_embeds

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3"):
            # 与 fa3_block 同一个不变量：索引 / cu 都按**完整序列**算，切分只体现在 hidden_states 与 rope 上。
            layer_kwargs = self.prepare_two_pass_mla_kwargs(
                input_ids,
                cu_seqlens,
                block_range,
            )
            layer_kwargs["cp"] = _cp
        elif self.config._attn_implementation == "fa3_block":
            layer_kwargs = {
                "block_start": block_range[:, 0].clone(),
                "block_end": block_range[:, 1].clone(),
                # CP 下补齐 item 只在这里加一次，保证跨层张量身份稳定 → 核的 meta 缓存能命中
                "cu_seqlens": cu_seqlens if _cp is None else _cp.pad_cu(cu_seqlens),
                "max_seqlen": max_seqlen,
                "cp": _cp,
            }
        else:
            raise NotImplementedError(f"no supported calling method {self.config._attn_implementation}")
        for decoder_layer in self.model.layers:
            hidden_states, aux_loss_layerwise = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                # use_cache=use_cache,
                **layer_kwargs,
            )

        # -- 5. Heads + losses ---------------------------------------------
        # AR loss uses local tokens and the global valid-token denominator.
        # Diffusion heads run on the full gathered sequence on every CP rank.
        # Each replica uses the same global mean loss, without multiplying by CP.
        # Gather backward sums head adjoints; FSDP averages across the CP ranks.
        _hidden_for_heads = hidden_states
        if _cp is not None:
            _hidden_for_heads = _cp.gather(hidden_states.squeeze(0)).unsqueeze(0)

        text_hidden = self.model.norm(hidden_states.squeeze(0))  # [L_text, H]（CP 下是本段）

        _ce_scale = None
        _labels_local = labels
        if _cp is not None and labels is not None:
            _lbl_full = labels.reshape(-1)
            _n_valid_global = int((_lbl_full != -100).sum().item())
            _labels_local = _cp.local(_lbl_full, dim=0, value=-100).reshape(1, -1)
            _n_valid_loc = int((_labels_local != -100).sum().item())
            if _n_valid_global > 0 and _n_valid_loc == 0:
                # 本段没有监督 token。全局有、本地无时若跳过 CE，反向图就与同组其它 rank
                # 不同构，FSDP 的 reduce-scatter bucket 对不上会确定性死锁。
                # 办法是喂 1 行占位 label 把图建出来，再用 scale=0 抹掉它的贡献。
                _labels_local = _labels_local.clone()
                _labels_local[0, 0] = 0
                _ce_scale = 0.0
            else:
                _ce_scale = float(_n_valid_loc) * _cp.size / float(max(_n_valid_global, 1))

        ar_loss = None
        ar_logits = None
        per_token_loss = None
        # T2I training sets all loss_masks to 0 → labels all -100 → no AR
        # supervision (the AR seq is the diffusion condition, not a target).
        # Skip CE entirely in that case: fused-CE over all-ignore is 0/0 = NaN,
        # and there's nothing to train. ``(labels != -100).any()`` keeps this
        # correct for mixed batches too (some samples may carry text labels).
        if labels is None:
            ar_logits = self.lm_head(text_hidden)
        elif bool((labels != -100).any()):  # 判据用全局 labels，保证组内各 rank 走同一分支
            from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.loss import (
                fused_linear_cross_entropy as _fused,
            )
            from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.loss import (
                fused_linear_cross_entropy_none as _fused_none,
            )
            from .base import is_seed_kernels_available

            lbl_flat = _labels_local.reshape(-1)
            if _fused is not None:
                if is_seed_kernels_available():
                    ar_loss = _fused(text_hidden, self.lm_head.weight, lbl_flat, ignore_index=-100)
                else:
                    ar_loss = _fused(self.lm_head.weight, text_hidden, lbl_flat)
            else:
                logger.warning_once(
                    "fused_linear_cross_entropy unavailable (liger/seed kernels missing) — "
                    f"falling back to full-logits CE over {text_hidden.shape[0]}x"
                    f"{self.lm_head.weight.shape[0]} logits. Expect OOM at train seqlen."
                )
                logits_flat = F.linear(text_hidden, self.lm_head.weight).float()
                ar_loss = F.cross_entropy(logits_flat, lbl_flat, ignore_index=-100)
            if _ce_scale is not None:
                ar_loss = ar_loss * _ce_scale
            if return_per_token_loss and _fused_none is not None:
                with torch.no_grad():
                    per_token_loss = _fused_none(self.lm_head.weight.detach(), text_hidden.detach(), lbl_flat)

        diff_loss = None
        diff_pred = None

        image_hidden_raw = _hidden_for_heads.squeeze(0).index_select(0, vae_idx)  # [L_diff, H]

        # Every CP rank consumes the complete gathered image-token sequence.
        # Keep all per-image metadata and targets in their original order.
        if vae_noised_latents is not None and vae_noised_latents.shape[0] > 0:
            if self.config.diff_time_embed_mode == "adaln":
                sc_out, sh_out = self.diff_adaln_out(t_emb_item).chunk(2, dim=-1)
                image_hidden = adarmsn_varlen(
                    image_hidden_raw,
                    self.diff_norm,
                    scale=sc_out,
                    shift=sh_out,
                    seqlens=diff_seq_lens,
                    fp32=self.config.adaln_fp32,
                )
            else:
                image_hidden = self.diff_norm(image_hidden_raw)  # [L_diff, H]

            diff_pred = self.diff_output_proj(image_hidden)  # [L_diff, C]

            if diff_target is not None:
                vae_seq_lens = diff_grid_thw.prod(dim=-1)
                is_target_vae_idx = (~(is_ref_vae).repeat_interleave(vae_seq_lens)).nonzero(as_tuple=True)[0]
                diff_target = diff_target.index_select(0, is_target_vae_idx)
                diff_pred = diff_pred.index_select(0, is_target_vae_idx)
                if self.config.diff_pred_type == "x0" and diff_flow_sigmas is not None:
                    _sig = (
                        diff_flow_sigmas.repeat_interleave(diff_seq_lens, dim=0)
                        .clamp_min(self.config.diff_t_min)
                        .unsqueeze(-1)
                        .float()
                    )
                    _sig = _sig.index_select(0, is_target_vae_idx)
                    _err2 = F.mse_loss(diff_pred.float(), diff_target.float(), reduction="none") / _sig.square()
                else:
                    # v-pred: diff_pred = v_hat, diff_target = noise - x0, plain MSE.
                    _err2 = F.mse_loss(diff_pred.float(), diff_target.float(), reduction="none")

                # Replicated head: global target-token mean, with no extra CP factor.
                diff_loss = _err2.mean()
        elif self.training:
            # TODO(wwx): 这里很多 dummy forward 都怪怪的，做成函数和 fake data 吧
            # dummy forward for diff head
            fake_image_hidden = _hidden_for_heads.squeeze(0)[:1, ...]
            if self.config.diff_time_embed_mode == "adaln":
                sc_out, sh_out = self.diff_adaln_out(t_emb_item).chunk(2, dim=-1)
                if is_triton_available():
                    fake_image_hidden = fused_adarmsn_scale_shift_single_modality(
                        fake_image_hidden,
                        (1 + sc_out).contiguous(),
                        sh_out.contiguous(),
                        torch.tensor([1], device=fake_image_hidden.device, dtype=torch.int32),
                        self.diff_norm.variance_epsilon,
                    )
                else:
                    fake_image_hidden = _adaln_apply(
                        self.diff_norm, fake_image_hidden, sc_out, sh_out, self.config.adaln_fp32
                    )
            else:
                fake_image_hidden = self.diff_norm(fake_image_hidden)  # [L_diff, H]

            fake_diff_pred = self.diff_output_proj(fake_image_hidden)  # [L_diff, C]
            diff_loss = fake_diff_pred.mean() * 0.0

        loss = None
        ar_weight = self.config.ar_loss_weight
        diff_weight = self.config.diff_loss_weight
        if ar_loss is not None and diff_loss is not None:
            loss = ar_weight * ar_loss + diff_weight * diff_loss
        elif ar_loss is not None:
            loss = ar_loss
        elif diff_loss is not None:
            loss = diff_loss

        if ar_loss is None and loss is not None:
            # TODO(wwx): dummy forward for ar head, but is it work?
            loss = loss + 0.0 * (self.lm_head.weight.sum() + self.model.norm.weight.sum())

        return GlmImageMoeTransfusionOutput(
            loss=loss,
            ar_loss=ar_loss,
            diff_loss=diff_loss,
            logits=ar_logits,
            aux_loss=None,
            per_token_loss=per_token_loss,
            diff_pred=diff_pred,
        )


if is_triton_available():
    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.fused_adarmsn import fused_adarmsn_scale_shift_single_modality
    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.rotary import apply_rotary_emb as apply_rotary_emb_triton

    def adarmsn_varlen(
        x: torch.Tensor,
        norm_module: nn.RMSNorm,
        scale: torch.Tensor,
        shift: torch.Tensor,
        seqlens: torch.IntTensor,
        fp32: bool = False,
    ) -> torch.Tensor:
        # This fused kernel already normalizes and modulates in FP32, then
        # casts once at its output. Keep its existing arithmetic for both flags.
        if hasattr(seqlens, "_cu_seqlens"):
            cu_seqlens = seqlens._cu_seqlens
        else:
            cu_seqlens = torch.cumsum(seqlens, 0, dtype=torch.int32)
            seqlens._cu_seqlens = cu_seqlens

        return fused_adarmsn_scale_shift_single_modality(
            x.contiguous(), (1 + scale).contiguous(), shift.contiguous(), cu_seqlens, norm_module.variance_epsilon
        )

    def apply_rotary_emb_fused(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
    ) -> torch.Tensor:
        cos = cos[..., : cos.shape[-1] // 2].squeeze(0)
        sin = sin[..., : sin.shape[-1] // 2].squeeze(0)
        x = apply_rotary_emb_triton(x, cos, sin, interleaved=interleaved)
        return x

    logger.info_rank0("apply triton kernel to GlmImageMoeTransfusion: \n adarmsn_varlen \n rope")

if is_torch_npu_available():
    import torch_npu

    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.npu_kernel import block_mask_attn_kernel as fused_block_mask_attn

    # MARK: 目前 triton 的 rope 在 npu 上是负加速，所以二次覆盖成用 torch_npu.npu_rotary_mul 的版本，至少是正速度收益
    def apply_rotary_emb_fused(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
    ) -> torch.Tensor:
        rotary_mode = "interleave" if interleaved else "half"
        if interleaved:
            half = cos.shape[-1] // 2
            cos = cos[..., :half].repeat_interleave(2, dim=-1)
            sin = sin[..., :half].repeat_interleave(2, dim=-1)
            cos = cos.unsqueeze(dim=-2)
            sin = sin.unsqueeze(dim=-2)
            x_dtype = x.dtype
            x = torch_npu.npu_rotary_mul(x.to(cos.dtype), cos, sin, rotary_mode=rotary_mode).to(x_dtype)
        else:
            x_dtype = x.dtype
            x = torch_npu.npu_rotary_mul(
                x.to(cos.dtype), cos.unsqueeze(dim=-2), sin.unsqueeze(dim=-2), rotary_mode=rotary_mode
            ).to(x_dtype)
        return x

    logger.info_rank0("apply npu patch to GlmImageMoeTransfusion rope, fused block attention")

__all__ = [
    "GlmImageMoeTransfusionConfig",
    "GlmImageMoeTransfusionForMMGeneration",
    "GlmImageMoeTransfusionOutput",
]

# ══════════ 搬自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py ══════════
"""HY/SelfFlow additions to the shared MoE30B transfusion model.

The existing model creates the decoder/attention/MoE/ViT and diffusion entry/exit.
This extension adds dual conditioning, gated DiT blocks and SelfFlow, keeping the
native model.layers parameter layout and checkpoint-name conversion.
"""

import copy
import os as _os_lmh
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from .configuration_hy import Glm4vVisionConfig
from .modeling_hy_vision import GlmImageMoePreTrainedModel
from .modeling_hy_vision import (
    Glm4vVisionPatchMerger,
    GlmImageModel,
    GlmImageTextRotaryEmbedding,
    GlmImageVQVisionEmbeddings,
)
from .modeling_hy_vision import _DiffNoAffineRMSNorm
from .modeling_hy_text import (
    Glm4MoeLiteExperts,
    Glm4MoeLiteMLP,
    Glm4MoeLiteMoE,
)
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import SequentialFp32, _adaln_apply）
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionAttention as NativeAttention）
NativeAttention = GlmImageMoeTransfusionAttention
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionBackbone as NativeBackbone）
NativeBackbone = GlmImageMoeTransfusionBackbone
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionConfig as NativeConfig）
NativeConfig = GlmImageMoeTransfusionConfig
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionForMMGeneration as NativeModel）
NativeModel = GlmImageMoeTransfusionForMMGeneration
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionMLP as NativeMLP）
NativeMLP = GlmImageMoeTransfusionMLP
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionMoE as NativeMoE）
NativeMoE = GlmImageMoeTransfusionMoE
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite_glm_image.modular_glm_image_moe_transfusion import GlmImageMoeTransfusionOutput as NativeOutput）
NativeOutput = GlmImageMoeTransfusionOutput
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as logging
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.base import is_torch_npu_available
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.base import attention_checkpoint_kwargs
# 下面这些 config 类已搬到 configuration_hy.py，从那里取
from .configuration_hy import GlmImageMoeTransfusionSelfFlowBaseConfig, GlmImageMoeTransfusionSelfFlowConfig, _TransfusionConditioningConfig, _TransfusionImageConfig


logger = logging.get_logger(__name__)
_ROPE_TABLE_DTYPE_DEFAULT = torch.float64
_DIFF_BLOCK_STACKS = ("diff_head_blocks", "diff_input_process_blocks")


# （已搬到 configuration_hy.py，原 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:67-123）
# class _TransfusionImageConfig: ...


def install_unpermute():
    import torch_npu

    native = torch_npu.npu_moe_token_unpermute
    if getattr(native, "_hy_fp32_probability_backward", False):
        return

    class FP32Probability(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, inverse, probability):
            ctx.save_for_backward(x, inverse, probability)
            return native(x, inverse, probs=probability)

        @staticmethod
        def backward(ctx, dy):
            x, inverse, probability = ctx.saved_tensors
            dx, dp = torch.ops.npu.npu_moe_token_unpermute_grad(x.float(), dy.float(), inverse, probability)
            return dx.to(x.dtype), None, dp

    def unpermute(x, inverse, probs=None, **kwargs):
        if probs is not None and x.dtype == torch.bfloat16 and probs.dtype == torch.float32 and not kwargs:
            return FP32Probability.apply(x, inverse, probs)
        return native(x, inverse, probs=probs, **kwargs)

    unpermute._hy_fp32_probability_backward = True
    torch_npu.npu_moe_token_unpermute = unpermute


def block_mask_kernel(device_type):
    if device_type == "npu":
        from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask.npu_kernel import block_mask_attn_kernel

        def attention(q, k, v, item_cu, starts, ends, max_len, scale=None, check_meta=False):
            return block_mask_attn_kernel(q, k, v, item_cu, starts, ends, scale=scale)

        return attention
    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.block_mask_fa3 import fused_block_mask_attn

    return fused_block_mask_attn


def _adaln_gate_residual(x: torch.Tensor, gate: torch.Tensor, sublayer_out: torch.Tensor, fp32: bool):
    if not fp32:
        return x + gate.to(sublayer_out.dtype) * sublayer_out
    return torch.addcmul(x, gate.float(), sublayer_out.float()).to(x.dtype)



class HYMLP(NativeMLP):
    def forward(self, x):
        if x.device.type == "npu":
            from liger_kernel.ops import LigerSiLUMulFunction

            return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
        return super().forward(x)


class HYVisionEmbeddings(GlmImageVQVisionEmbeddings):
    def forward(self, hidden_states, lengths, grid_thw, h_coords, w_coords):
        # The source recipe uses the stage3 ViT's bilinear, antialiased resize.
        # Keep this choice local to HY; ordinary model checkpoints retain their
        # existing position interpolation.
        embeddings = self.patch_embedding(hidden_states.to(self.patch_embedding.weight.dtype))
        size = int(self.num_positions**0.5)
        positions = self.position_embedding.weight.view(size, size, self.embed_dim)
        positions = positions.permute(2, 0, 1).unsqueeze(0).float()
        if grid_thw.shape[0] == 0:
            return embeddings
        parts = []
        for height, width in grid_thw[:, 1:3].tolist():
            height, width = int(height), int(width)
            resized = torch.nn.functional.interpolate(
                positions, size=(height, width), mode="bilinear", align_corners=False, antialias=True
            )
            parts.append(resized.squeeze(0).permute(1, 2, 0).reshape(height * width, self.embed_dim))
        positions = torch.cat(parts, dim=0)
        return embeddings + positions.to(embeddings.dtype).to(embeddings.device)


class HYExperts(Glm4MoeLiteExperts):
    def fused_forward(self, hidden_states, top_k_index, top_k_weights):
        from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops import fused_moe

        # The common front door imposes the GPU Triton probability dtype rule.
        # The existing NPU backend accepts FP32 probabilities directly.
        return fused_moe._fused_moe_forward(
            self,
            self.num_experts,
            top_k_weights,
            top_k_index,
            hidden_states,
            self.gate_up_proj,
            None,
            self.down_proj,
        )


class HYMoE(NativeMoE):
    def forward(self, hidden_states):
        batch, length, hidden = hidden_states.shape
        logits, weights, indices = self.gate(hidden_states)
        if not self.router_weight_fp32:
            weights = weights.to(hidden_states.dtype)
        routed = self.experts(hidden_states.reshape(-1, hidden), indices, weights)
        routed = routed.view(batch, length, hidden)
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states)
        return routed, None


def adapt_backbone(model):
    from .rms_norm_hy import configure_hy_rmsnorm

    configure_hy_rmsnorm(model, npu=is_torch_npu_available())
    if is_torch_npu_available():
        install_unpermute()
    for module in list(model.modules()):
        if isinstance(module, GlmImageVQVisionEmbeddings):
            module.__class__ = HYVisionEmbeddings
        if isinstance(module, Glm4MoeLiteMoE):
            if getattr(module.config, "router_aux_loss_coef", 0.0):
                raise ValueError("HY reference recipe requires router_aux_loss_coef=0")
            if getattr(module.config, "router_class_bias_update_speed", 0.0):
                raise ValueError("HY reference recipe requires router_class_bias_update_speed=0")
            module.__class__ = HYMoE
            module.router_weight_fp32 = bool(getattr(module.config, "moe_router_weight_fp32", True))
        if isinstance(module, Glm4MoeLiteMLP):
            module.__class__ = HYMLP
        if isinstance(module, Glm4MoeLiteExperts) and is_torch_npu_available():
            if getattr(module.forward, "__name__", "") == "fused_forward":
                if not hasattr(module, "gate_up_proj"):
                    module.gate_up_proj = nn.Parameter(
                        torch.cat([module.gate_proj, module.up_proj], dim=1),
                        requires_grad=module.gate_proj.requires_grad,
                    )
                    del module.gate_proj
                    del module.up_proj
                    module._parameters = {
                        "gate_up_proj": module.gate_up_proj,
                        "down_proj": module.down_proj,
                    }
                module.__class__ = HYExperts
                module.forward = module.fused_forward
    return model


def _rope_tables(rot_emb, position_ids, ref, dtype: torch.dtype = _ROPE_TABLE_DTYPE_DEFAULT):
    if dtype == torch.float64:
        return _rope_tables_fp64(rot_emb, position_ids)
    if dtype == torch.float32:
        return rot_emb(torch.empty(1, dtype=torch.float32, device=ref.device), position_ids)
    raise ValueError(f"_rope_tables: dtype must be torch.float32 or torch.float64, got {dtype}")


def _rope_tables_fp64(rot_emb, position_ids):
    cfg = rot_emb.config.rope_parameters
    if cfg.get("rope_type", "default") != "default":
        raise RuntimeError(
            f"dtype=torch.float64 only handles rope_type='default', got {cfg.get('rope_type')!r}; the scaled variants build inv_freq in ROPE_INIT_FUNCTIONS and would need their own fp64 path."
        )
    base = float(cfg["rope_theta"])
    head_dim = (
        getattr(rot_emb.config, "head_dim", None) or rot_emb.config.hidden_size // rot_emb.config.num_attention_heads
    )
    dim = int(head_dim * cfg.get("partial_rotary_factor", 1.0))
    dev = position_ids.device
    inv = 1.0 / torch.tensor(base, dtype=torch.float64, device=dev) ** (
        torch.arange(0, dim, 2, dtype=torch.int64, device=dev).double() / dim
    )
    ref_buf = rot_emb.inv_freq.to(dev)
    ref_inv = ref_buf.double()
    _tol = max(8.0 * float(torch.finfo(ref_buf.dtype).eps), 1e-06)
    if not torch.allclose(inv, ref_inv, rtol=_tol, atol=0.0):
        raise RuntimeError(
            f"dtype=torch.float64: recomputed inv_freq disagrees with rot_emb.inv_freq (max rel {((inv - ref_inv).abs() / ref_inv).max():.3e} > tol {_tol:.3e} for buffer dtype {ref_buf.dtype}); the module is not using the plain base**(-2i/d) schedule."
        )
    ang = position_ids.double()[..., None] * inv
    scaling = float(getattr(rot_emb, "attention_scaling", 1.0))
    cos, sin = (ang.cos() * scaling).float(), (ang.sin() * scaling).float()
    return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


def _mla_two_pass_attention(
    attn,
    hidden_states,
    position_embeddings,
    item_cu,
    text_idx,
    image_idx,
    max_item_len,
    text_cu,
    image_cu,
    max_text_len,
    max_image_len,
):
    out, _ = attn._two_pass_mla_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        item_cu=item_cu,
        text_cu=text_cu,
        image_cu=image_cu,
        text_idx=text_idx,
        image_idx=image_idx,
        max_item_len=max_item_len,
        max_text_len=max_text_len,
        max_image_len=max_image_len,
        attn_implementation="flash_attention_2",
    )
    return out


class DiffHeadBlock(nn.Module):
    def __init__(self, text_cfg, config, layer_idx: int = 0):
        super().__init__()
        self.self_attn = NativeAttention(text_cfg, layer_idx=layer_idx)
        self.self_attn.attention_dropout = float(config.diff_head_attention_dropout)
        self.mlp = NativeMLP(text_cfg)
        hidden = text_cfg.hidden_size
        eps = text_cfg.rms_norm_eps
        self.input_layernorm = _DiffNoAffineRMSNorm(hidden, eps=eps)
        self.post_attn_layernorm = _DiffNoAffineRMSNorm(hidden, eps=eps)
        self._adaln_fp32 = config.adaln_fp32
        _seq_cls = SequentialFp32 if self._adaln_fp32 else nn.Sequential
        time_dim = getattr(config, "diff_time_emb_dim", hidden)
        self.adaln_attn = _seq_cls(nn.SiLU(), nn.Linear(time_dim, 2 * hidden, bias=True))
        self.adaln_mlp = _seq_cls(nn.SiLU(), nn.Linear(time_dim, 2 * hidden, bias=True))
        nn.init.zeros_(self.adaln_attn[1].weight)
        nn.init.zeros_(self.adaln_attn[1].bias)
        nn.init.zeros_(self.adaln_mlp[1].weight)
        nn.init.zeros_(self.adaln_mlp[1].bias)

    def _attn_diff_only(self, hidden_states, cos_pre, sin_pre, diff_cu_seqlens_i32, max_diff_item_len):
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        return _mla_two_pass_attention(
            self.self_attn,
            hidden_states,
            (cos_pre, sin_pre),
            diff_cu_seqlens_i32,
            indices[:0],
            indices,
            max_diff_item_len,
            torch.zeros_like(diff_cu_seqlens_i32),
            diff_cu_seqlens_i32,
            0,
            max_diff_item_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb_item: torch.Tensor,
        diff_item_ids: torch.Tensor,
        cos_pre: torch.Tensor,
        sin_pre: torch.Tensor,
        diff_cu_seqlens_i32: torch.Tensor,
        max_diff_item_len: int,
    ) -> torch.Tensor:
        sc_a, sh_a = self.adaln_attn(t_emb_item).index_select(0, diff_item_ids).chunk(2, dim=-1)
        modulated = _adaln_apply(self.input_layernorm, x, sc_a.unsqueeze(0), sh_a.unsqueeze(0), self._adaln_fp32)
        attn_out = self._attn_diff_only(modulated, cos_pre, sin_pre, diff_cu_seqlens_i32, max_diff_item_len)
        x = x + attn_out
        sc_m, sh_m = self.adaln_mlp(t_emb_item).index_select(0, diff_item_ids).chunk(2, dim=-1)
        modulated = _adaln_apply(self.post_attn_layernorm, x, sc_m.unsqueeze(0), sh_m.unsqueeze(0), self._adaln_fp32)
        mlp_out = self.mlp(modulated)
        x = x + mlp_out
        return x


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


@torch.no_grad()
def reset_diff_block_adaln_zero(model: nn.Module) -> int:
    n = 0
    for blocks in (getattr(model, "diff_head_blocks", None), getattr(model, "diff_input_process_blocks", None)):
        if blocks is None:
            continue
        for block in blocks:
            for mod in (block.adaln_attn[1], block.adaln_mlp[1]):
                _local(mod.weight).zero_()
                if mod.bias is not None:
                    _local(mod.bias).zero_()
            n += 1
    return n


def diff_block_adaln_in_weights(model_path: Optional[str]) -> bool:
    import json
    import os
    import re

    if not model_path or not os.path.isdir(model_path):
        return False
    pat = re.compile("^(?:model\\.)?diff_(?:head|input_process)_blocks\\.\\d+\\.adaln_(?:attn|mlp)\\.1\\.weight$")
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            keys = json.load(f).get("weight_map", {}).keys()
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(single):
            return False
        from safetensors import safe_open

        with safe_open(single, "pt") as f:
            keys = list(f.keys())
    return any((pat.match(k) for k in keys))


def _global_absmax(tensors: list) -> list:
    import torch.distributed as dist

    if not tensors:
        return []
    dev = _local(tensors[0]).device
    vals = torch.stack(
        [_local(t).detach().abs().max().float() if _local(t).numel() else torch.zeros((), device=dev) for t in tensors]
    ).to(dev)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vals, op=dist.ReduceOp.MAX)
    return vals.tolist()


@torch.no_grad()
def repair_diff_block_zero_rmsnorm(model: nn.Module) -> list:
    names, weights = ([], [])
    for attr in _DIFF_BLOCK_STACKS:
        for bi, blk in enumerate(getattr(model, attr, None) or []):
            for mname, mod in blk.self_attn.named_modules():
                w = getattr(mod, "weight", None)
                if w is None or w.dim() != 1 or (not type(mod).__name__.endswith("RMSNorm")):
                    continue
                names.append(f"{attr}.{bi}.self_attn.{mname}.weight")
                weights.append(w)
    fixed = []
    for name, w, m in zip(names, weights, _global_absmax(weights)):
        if m == 0.0:
            _local(w).fill_(1.0)
            fixed.append(name)
    return fixed


@torch.no_grad()
def assert_no_unexpected_zero_in_diff_blocks(model: nn.Module) -> None:
    names, params = ([], [])
    for attr in _DIFF_BLOCK_STACKS:
        for bi, blk in enumerate(getattr(model, attr, None) or []):
            for pname, p in blk.named_parameters():
                if "adaln_" in pname:
                    continue
                names.append(f"{attr}.{bi}.{pname}")
                params.append(p)
    bad = [n for n, m in zip(names, _global_absmax(params)) if m == 0.0]
    if bad:
        raise RuntimeError(
            f"diff block 里这些参数整体为零：{bad[:8]}{(' ...' if len(bad) > 8 else '')}。零权重会让所在支路输出恒 0、adaLN-Zero 的 gate 永远拿不到梯度 —— 支路永久失效且不报错。不要启动。"
        )


class _TransfusionImageEntryModel(NativeModel):
    _tied_weights_keys = {}
    accepts_loss_kwargs = False
    rotary_embedding_class = GlmImageTextRotaryEmbedding
    # FSDP handles SequentialFp32 through its mixed-precision exemptions,
    # including containers nested in DiT blocks. Do not register them again
    # as standalone decoder blocks after their parent has already sharded them.
    _no_split_modules = [
        name for name in NativeModel._no_split_modules if name != "SequentialFp32"
    ] + ["DiffHeadBlock", "GatedDiffDiTBlock"]

    def get_fsdp_prefetch_order(self):
        """The HY block execution order used by zwd's optional HP prefetch plan."""
        return [
            *(self.diff_input_process_blocks or []),
            *self.model.layers,
            *(self.diff_head_blocks or []),
        ]

    def __init__(self, config):
        # Reuse native construction without requiring a backbone factory hook.
        # The temporary native RoPE is replaced before weights or inputs load.
        native_config = copy.copy(config)
        native_config.diff_rope_thw_layout = "block"
        super().__init__(native_config)
        self.config = config
        self.model.config = config
        self.model.rotary_emb = self.rotary_embedding_class(config.text_config)
        self.model.vq_projector = None
        self._rope_table_dtype = {"float64": torch.float64, "float32": torch.float32}[config.rope_table_dtype]
        self.diff_enabled = config.diff_enabled
        self._use_adaln = config.diff_time_embed_mode == "adaln"
        self._adaln_fp32 = config.adaln_fp32
        self._diff_head_num_layers = config.diff_head_num_layers
        self._diff_input_process_num_layers = config.diff_input_process_num_layers
        self._diff_input_process_attention_dropout = config.diff_input_process_attention_dropout
        hidden = config.text_config.hidden_size
        if self._use_adaln:
            self.diff_norm = _DiffNoAffineRMSNorm(hidden, config.text_config.rms_norm_eps)
            self.diff_input_norm = _DiffNoAffineRMSNorm(hidden, config.text_config.rms_norm_eps)
        if not self.diff_enabled:
            for name in (
                "diff_input_proj",
                "diff_time_embed",
                "diff_norm",
                "diff_output_proj",
                "diff_input_norm",
                "diff_adaln_in",
                "diff_adaln_out",
            ):
                setattr(self, name, None)
        self.diff_head_blocks = (
            nn.ModuleList([DiffHeadBlock(config.text_config, config) for _ in range(self._diff_head_num_layers)])
            if self.diff_enabled and self._diff_head_num_layers
            else None
        )
        self.diff_input_process_blocks = (
            nn.ModuleList(
                [DiffHeadBlock(config.text_config, config) for _ in range(self._diff_input_process_num_layers)]
            )
            if self.diff_enabled and self._diff_input_process_num_layers
            else None
        )
        self.post_init()

    def strict_load_allow_missing_defaults(self):
        return (
            [r"\.gate\.e_score_correction_bias_class$"]
            if getattr(self.config.text_config, "router_class_bias_update_speed", 0.0)
            else []
        )

    def _init_weights(self, module):
        GlmImageMoePreTrainedModel._init_weights(self, module)
        if "RMSNorm" in type(module).__name__ and getattr(module, "weight", None) is not None:
            module.weight.data.fill_(1.0)
        if isinstance(module, DiffHeadBlock):
            for norm in (module.adaln_attn, module.adaln_mlp):
                nn.init.zeros_(norm[1].weight)
                nn.init.zeros_(norm[1].bias)

    def _build_diff_position_ids(
        self,
        diff_grid_thw: torch.Tensor,
        diff_cu_seqlens: torch.Tensor,
        device: torch.device,
        diff_pos_anchors: Optional[torch.Tensor] = None,
        diff_vit_grids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_list, h_list, w_list = ([], [], [])
        N_items = diff_grid_thw.shape[0]
        grids = diff_grid_thw.tolist()
        anchors = diff_pos_anchors.reshape(-1).tolist() if diff_pos_anchors is not None else None
        vit_grids = diff_vit_grids.tolist() if diff_vit_grids is not None else None
        for i in range(N_items):
            T, Hg, Wg = (int(grids[i][0]), int(grids[i][1]), int(grids[i][2]))
            assert T == 1, f"diff expects T=1, got T={T} for item {i}"
            anchor = float(anchors[i]) if anchors is not None else 0.0
            if vit_grids is not None:
                Hv = int(vit_grids[i][1])
                Wv = int(vit_grids[i][2])
            else:
                Hv, Wv = (Hg, Wg)
            h_coord = torch.arange(Hg, device=device, dtype=torch.float) * (Hv / Hg)
            w_coord = torch.arange(Wg, device=device, dtype=torch.float) * (Wv / Wg)
            h_idx = (anchor + h_coord).view(-1, 1).expand(Hg, Wg).reshape(-1)
            w_idx = (anchor + w_coord).view(1, -1).expand(Hg, Wg).reshape(-1)
            t_idx = torch.full((Hg * Wg,), anchor, device=device, dtype=torch.float)
            t_list.append(t_idx)
            h_list.append(h_idx)
            w_list.append(w_idx)
        pos_t = torch.cat(t_list, dim=0)
        pos_h = torch.cat(h_list, dim=0)
        pos_w = torch.cat(w_list, dim=0)
        return torch.stack([pos_t, pos_h, pos_w], dim=0).unsqueeze(1)

    @staticmethod
    def _timesteps_per_token(timesteps: torch.Tensor, diff_cu_seqlens: torch.Tensor) -> torch.Tensor:
        per_item_lens = diff_cu_seqlens[1:] - diff_cu_seqlens[:-1]
        return timesteps.repeat_interleave(per_item_lens.to(torch.long))

    @staticmethod
    def _interleave(
        text_embeds: torch.Tensor,
        text_pos: torch.Tensor,
        diff_h_seq: Optional[torch.Tensor],
        diff_pos: Optional[torch.Tensor],
        ar_prefix_ranges: Optional[torch.Tensor],
        diff_cu_eff: Optional[torch.Tensor],
        diff_t_token_mask: Optional[torch.Tensor],
        device: torch.device,
        has_diff_per_item: Optional[torch.Tensor] = None,
    ):
        if ar_prefix_ranges is None:
            L = text_embeds.shape[1]
            is_image = torch.zeros(L, dtype=torch.bool, device=device)
            item_cu = torch.tensor([0, L], dtype=torch.int32, device=device)
            image_cu = torch.zeros(item_cu.shape[0], dtype=torch.int32, device=device)
            return (text_embeds, text_pos.unsqueeze(1), is_image, is_image, item_cu, L, item_cu, image_cu, L, 0)
        embeds, poss, is_img_parts, t_parts = ([], [], [], [])
        item_lens, text_lens, image_lens = ([], [], [])
        N = ar_prefix_ranges.shape[0]
        ranges = ar_prefix_ranges.tolist()
        cu_eff = diff_cu_eff.tolist() if diff_cu_eff is not None else None
        if has_diff_per_item is None:
            has_diff = [True] * N
        else:
            has_diff = has_diff_per_item.to(torch.bool).tolist()
        diff_idx = 0
        for i in range(N):
            s = int(ranges[i][0])
            e = int(ranges[i][1])
            p_i = e - s
            embeds.append(text_embeds[0, s:e])
            poss.append(text_pos[:, s:e])
            is_img_parts.append(torch.zeros(p_i, dtype=torch.bool, device=device))
            t_parts.append(torch.zeros(p_i, dtype=torch.bool, device=device))
            if has_diff[i]:
                ds = int(cu_eff[diff_idx])
                de = int(cu_eff[diff_idx + 1])
                d_i = de - ds
                embeds.append(diff_h_seq[0, ds:de])
                poss.append(diff_pos[:, ds:de])
                is_img_parts.append(torch.ones(d_i, dtype=torch.bool, device=device))
                if diff_t_token_mask is not None:
                    t_parts.append(diff_t_token_mask[ds:de])
                else:
                    t_parts.append(torch.zeros(d_i, dtype=torch.bool, device=device))
                diff_idx += 1
            else:
                d_i = 0
            item_lens.append(p_i + d_i)
            text_lens.append(p_i)
            image_lens.append(d_i)
        inputs_embeds = torch.cat(embeds, dim=0).unsqueeze(0)
        position_ids = torch.cat(poss, dim=1).unsqueeze(1)
        is_image = torch.cat(is_img_parts, dim=0)
        t_token_in_unified = torch.cat(t_parts, dim=0)

        def _to_cu(lens):
            c = torch.zeros(N + 1, dtype=torch.int32, device=device)
            c[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
            return c

        item_cu = _to_cu(item_lens)
        text_cu = _to_cu(text_lens)
        image_cu = _to_cu(image_lens)
        return (
            inputs_embeds,
            position_ids,
            is_image,
            t_token_in_unified,
            item_cu,
            max(item_lens),
            text_cu,
            image_cu,
            max(text_lens),
            max(image_lens),
        )

    def _run_layer(
        self,
        decoder_layer,
        hidden_states,
        position_embeddings,
        item_cu,
        text_idx,
        image_idx,
        max_item_len,
        text_cu,
        image_cu,
        max_text_len,
        max_image_len,
    ):
        attn = decoder_layer.self_attn

        def _layer_fn(hidden_states):
            orig_forward = attn.forward

            def _patched(hidden_states, position_embeddings=None, attention_mask=None, **_kw):
                out = _mla_two_pass_attention(
                    attn,
                    hidden_states,
                    position_embeddings,
                    item_cu,
                    text_idx,
                    image_idx,
                    text_cu,
                    image_cu,
                    max_text_len,
                    max_item_len,
                    max_image_len,
                )
                return (out, None)

            _saved_gc = decoder_layer.gradient_checkpointing
            decoder_layer.gradient_checkpointing = False
            attn.forward = _patched
            try:
                out = decoder_layer(hidden_states, position_embeddings=position_embeddings)
                return out[0] if isinstance(out, tuple) else out
            finally:
                attn.forward = orig_forward
                decoder_layer.gradient_checkpointing = _saved_gc

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                _layer_fn, hidden_states, use_reentrant=False, **attention_checkpoint_kwargs()
            )
        return _layer_fn(hidden_states)

    def _assemble_mrope(self, cos, sin, image_rope_mask):
        """Select the configured T/H/W layout for all image rows, including conditions.

        image_rope_mask is independent of the generation-only mask used by
        diffusion losses, timestep assignment and SelfFlow feature selection.
        """
        D = cos.shape[-1]
        device = cos.device
        cache = getattr(self, "_mrope_sec_cache", None)
        if cache is None or cache[0].device != device or cache[0].shape[-1] != D:
            mrope_section = self.config.text_config.rope_parameters["mrope_section"]
            D_pairs = sum(mrope_section)
            D_full = 2 * D_pairs
            assert D in (D_pairs, D_full), f"rope table width {D}, expected {D_pairs} or {D_full}"

            def _fit(t):
                return t[:D]

            sec_block_list = []
            for i, seg_len in enumerate(mrope_section * 2):
                sec_block_list.extend([i % 3] * seg_len)
            sec_block = _fit(torch.tensor(sec_block_list, dtype=torch.long, device=device))
            half = D_pairs
            n_high = min(8, half)
            _LOW_INTERLEAVE = (1, 2, 0)
            sec_half_interleave = [
                (1 if k % 2 == 0 else 2) if k < n_high else _LOW_INTERLEAVE[(k - n_high) % 3] for k in range(half)
            ]
            sec_interleave = _fit(
                torch.tensor(sec_half_interleave + sec_half_interleave, dtype=torch.long, device=device)
            )
            t_n, h_n, w_n = mrope_section
            assert h_n >= t_n and w_n >= t_n, f"hw_thw sec requires h_n>=t_n and w_n>=t_n; got {mrope_section}"
            total_pairs = t_n + h_n + w_n
            axis_per_pair = [0] * total_pairs
            h_remain, w_remain = (h_n - t_n, w_n - t_n)
            high_n = h_remain + w_remain
            h_left, w_left = (h_remain, w_remain)
            for i in range(high_n):
                want_h = i % 2 == 0
                if want_h and h_left > 0:
                    axis_per_pair[i] = 1
                    h_left -= 1
                elif not want_h and w_left > 0:
                    axis_per_pair[i] = 2
                    w_left -= 1
                elif h_left > 0:
                    axis_per_pair[i] = 1
                    h_left -= 1
                else:
                    axis_per_pair[i] = 2
                    w_left -= 1
            for j in range(3 * t_n):
                axis_per_pair[high_n + j] = [0, 1, 2][j % 3]
            sec_hw_thw = _fit(torch.tensor(axis_per_pair + axis_per_pair, dtype=torch.long, device=device))
            axis_q = axis_per_pair[high_n:] + axis_per_pair[:high_n]
            sec_qwen3 = _fit(torch.tensor(axis_q + axis_q, dtype=torch.long, device=device))
            self._mrope_sec_cache = (sec_block, sec_interleave, sec_hw_thw, sec_qwen3)
            cache = self._mrope_sec_cache
        sec_block, sec_interleave, sec_hw_thw, sec_qwen3 = cache
        layout = self.config.diff_rope_thw_layout
        if layout == "hw_thw":
            sec_per_row = torch.where(image_rope_mask[:, None], sec_hw_thw[None, :], sec_block[None, :])
        elif layout == "qwen3":
            sec_per_row = torch.where(image_rope_mask[:, None], sec_qwen3[None, :], sec_block[None, :])
        elif layout == "interleave":
            sec_per_row = torch.where(image_rope_mask[:, None], sec_interleave[None, :], sec_block[None, :])
        else:
            raise ValueError(
                f"diff_rope_thw_layout={layout!r} is not selectable ('block' is permanently disabled; it survives only as the text-row table). Use 'qwen3' / 'hw_thw' / 'interleave'."
            )
        B = cos.shape[1]
        idx = sec_per_row.unsqueeze(0).unsqueeze(0).expand(1, B, -1, -1)
        cos_pre = torch.gather(cos, 0, idx).squeeze(0)
        sin_pre = torch.gather(sin, 0, idx).squeeze(0)
        return (cos_pre, sin_pre)


def _unshuffle_packed(x: torch.Tensor, grid_thw: torch.Tensor, k: int) -> torch.Tensor:
    if k == 1:
        return x
    outs = []
    off = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = (int(t), int(h), int(w))
        n = t * h * w
        assert h % k == 0 and w % k == 0, f"cond_merge_size={k} requires an even token grid, got ({h}, {w})"
        blk = x[off : off + n].reshape(t, h // k, k, w // k, k, x.shape[-1])
        blk = blk.permute(0, 1, 3, 2, 4, 5).reshape(t * (h // k) * (w // k), k * k * x.shape[-1])
        outs.append(blk)
        off += n
    assert off == x.shape[0], f"grid_thw covers {off} tokens, got {x.shape[0]}"
    return torch.cat(outs, dim=0)


def _merged_grid(grid_thw: torch.Tensor, k: int) -> torch.Tensor:
    if k == 1:
        return grid_thw
    g = grid_thw.clone()
    g[:, 1] = g[:, 1] // k
    g[:, 2] = g[:, 2] // k
    return g


class CondViTAligner(nn.Module):
    def __init__(
        self,
        vit_hidden: int,
        hidden: int,
        merge: int = 2,
        depth: int = 2,
        projector_type: str = "glmv",
        context_dim: int = 0,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.merge = int(merge)
        self.projector_type = projector_type
        in_dim = vit_hidden * self.merge * self.merge
        if projector_type == "glmv":
            self.downsample = nn.Linear(in_dim, hidden)
            self.merger = Glm4vVisionPatchMerger(
                dim=hidden, context_dim=int(context_dim or 4 * hidden), hidden_act=hidden_act
            )
        elif projector_type == "linear":
            self.layers = nn.Linear(in_dim, hidden)
        elif projector_type == "mlp_gelu":
            mods: list[nn.Module] = [nn.Linear(in_dim, hidden)]
            for _ in range(1, max(1, int(depth))):
                mods.append(nn.GELU())
                mods.append(nn.Linear(hidden, hidden))
            self.layers = nn.Sequential(*mods)
        else:
            raise ValueError(f"Unknown cond_vit_aligner {projector_type!r}")

    @property
    def _first(self) -> nn.Linear:
        if self.projector_type == "glmv":
            return self.downsample
        return self.layers if isinstance(self.layers, nn.Linear) else self.layers[0]

    @property
    def in_features(self) -> int:
        return self._first.in_features

    @property
    def weight(self) -> torch.Tensor:
        return self._first.weight

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        h = _unshuffle_packed(x, grid_thw, self.merge)
        if self.projector_type == "glmv":
            return self.merger(self.downsample(h))
        return self.layers(h)


class GatedDiffDiTBlock(DiffHeadBlock):
    def __init__(self, text_cfg, config, layer_idx: int = 0):
        super().__init__(text_cfg, config, layer_idx=layer_idx)
        hidden = text_cfg.hidden_size
        _seq_cls = type(self.adaln_attn)
        time_dim = getattr(config, "diff_time_emb_dim", hidden)
        self.adaln_attn = _seq_cls(nn.SiLU(), nn.Linear(time_dim, 3 * hidden, bias=True))
        self.adaln_mlp = _seq_cls(nn.SiLU(), nn.Linear(time_dim, 3 * hidden, bias=True))
        for m in (self.adaln_attn, self.adaln_mlp):
            nn.init.zeros_(m[1].weight)
            nn.init.zeros_(m[1].bias)

    def _forward_with_attn(self, x, t_emb_item, row_ids, attn_fn):
        sc_a, sh_a, g_a = self.adaln_attn(t_emb_item).index_select(0, row_ids).chunk(3, dim=-1)
        modulated = _adaln_apply(self.input_layernorm, x, sc_a.unsqueeze(0), sh_a.unsqueeze(0), self._adaln_fp32)
        attn_out = attn_fn(modulated)
        x = _adaln_gate_residual(x, g_a.unsqueeze(0), attn_out, self._adaln_fp32)
        sc_m, sh_m, g_m = self.adaln_mlp(t_emb_item).index_select(0, row_ids).chunk(3, dim=-1)
        modulated = _adaln_apply(self.post_attn_layernorm, x, sc_m.unsqueeze(0), sh_m.unsqueeze(0), self._adaln_fp32)
        mlp_out = self.mlp(modulated)
        return _adaln_gate_residual(x, g_m.unsqueeze(0), mlp_out, self._adaln_fp32)

    def forward(self, x, t_emb_item, diff_item_ids, cos_pre, sin_pre, *attn_args):
        if len(attn_args) == 4:
            return self.forward_fullseq(x, t_emb_item, diff_item_ids, cos_pre, sin_pre, *attn_args)
        if len(attn_args) != 2:
            raise TypeError(
                f"{type(self).__name__}.forward: expected 2 (diff-only) or 4 (full-sequence) attention args, got {len(attn_args)}"
            )
        diff_cu_seqlens_i32, max_diff_item_len = attn_args
        return self._forward_with_attn(
            x,
            t_emb_item,
            diff_item_ids,
            lambda m: self._attn_diff_only(m, cos_pre, sin_pre, diff_cu_seqlens_i32, max_diff_item_len),
        )

    def forward_fullseq(
        self, x, t_emb_table, t_row_ids, cos_pre, sin_pre, item_cu, blk_start, blk_end, max_item_len: int
    ):
        return self._forward_with_attn(
            x,
            t_emb_table,
            t_row_ids,
            lambda m: _mla_block_mask_attention(
                self.self_attn, m, (cos_pre, sin_pre), item_cu, blk_start, blk_end, max_item_len
            ),
        )


# （已搬到 configuration_hy.py，原 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:944-1019）
# class _TransfusionConditioningConfig: ...


def _mla_block_mask_attention(
    attn, hidden_states, position_embeddings, item_cu_seqlens, blk_start, blk_end, max_item_len, check_meta=False,
    cp=None,
):
    out, _ = attn._fa3_block_fused_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        cu_seqlens=item_cu_seqlens,
        block_start=blk_start,
        block_end=blk_end,
        max_seqlen=max_item_len,
        cp=cp,
    )
    return out


class _TransfusionConditionedModel(_TransfusionImageEntryModel):
    config_class = _TransfusionConditioningConfig
    _no_split_modules = _TransfusionImageEntryModel._no_split_modules + ["GatedDiffDiTBlock"]

    def __init__(self, config: _TransfusionConditioningConfig):
        super().__init__(config)
        vq_vc = getattr(config, "vq_vision_config", None)
        assert self.model.vq_visual is not None and vq_vc is not None, (
            "HY needs the SigLIP2 ViT: build with init_vq=True (yaml: enable_online_vq: true). Only the VQ *projector* is retired, not the encoder."
        )
        self.model.vq_projector = None
        self.vq_to_llm_proj = None
        hidden = config.text_config.hidden_size
        self._cond_merge = int(config.cond_merge_size)
        self._cond_dual = bool(config.cond_image_dual_encoder)
        self._attn_block_mode = str(config.attn_block_mode)
        self._cond_dual_block_causal = bool(getattr(config, "cond_dual_block_causal", True))
        self._fullseq_diff_blocks = bool(getattr(config, "hy_fullseq_diff_blocks", False))
        sm = int(getattr(self.model.vq_visual, "spatial_merge_size", 1) or 1)
        assert self._cond_merge % sm == 0, (
            f"cond_merge_size={self._cond_merge} must be a multiple of the ViT's spatial_merge_size={sm}"
        )
        self._cond_merge_eff = self._cond_merge // sm
        self.cond_vit_aligner = self._build_cond_vit_aligner(vq_vc, hidden)
        self.diff_time_embed_out = None
        if self.diff_enabled and bool(getattr(config, "diff_time_embed_split", True)):
            self.diff_time_embed_out = type(self.diff_time_embed)(
                hidden_size=config.diff_time_emb_dim, freq_dim=config.diff_timestep_embed_dim
            )
        if self._cond_dual:
            assert self.diff_enabled and self.diff_input_proj is not None, (
                "cond_image_dual_encoder=True reuses the generation entry chain (diff_input_proj → input-process → diff_adaln_in); it requires diff_enabled=True and a real diff_input_proj (not diff_input_via_vit)."
            )
        if bool(getattr(config, "diff_block_gate", True)):

            class _DropShim:
                def __init__(self, drop):
                    self.diff_head_attention_dropout = drop
                    self.adaln_fp32 = config.adaln_fp32
                    self.diff_time_emb_dim = config.diff_time_emb_dim

            for _attr, _drop in (
                ("diff_head_blocks", getattr(config, "diff_head_attention_dropout", 0.0)),
                ("diff_input_process_blocks", getattr(config, "diff_input_process_attention_dropout", 0.0)),
            ):
                _blocks = getattr(self, _attr, None)
                if _blocks is None:
                    continue
                _shim = _DropShim(float(_drop or 0.0))
                _new = nn.ModuleList(
                    [GatedDiffDiTBlock(config.text_config, _shim, layer_idx=0) for _ in range(len(_blocks))]
                )
                for _b in _new:
                    nn.init.zeros_(_b.adaln_attn[1].weight)
                    nn.init.zeros_(_b.adaln_attn[1].bias)
                    nn.init.zeros_(_b.adaln_mlp[1].weight)
                    nn.init.zeros_(_b.adaln_mlp[1].bias)
                setattr(self, _attr, _new)
        if self._attn_block_mode == "blockmask" and getattr(self, "_diff_moe_enabled", False):
            raise NotImplementedError(
                "attn_block_mode='blockmask' together with diff_moe_enabled=True is not wired: the block-mask _run_layer path does not carry the diff-MoE FFN splice. Turn one of them off."
            )
        if self._attn_block_mode == "blockmask":
            _drop = float(getattr(config.text_config, "attention_dropout", 0.0) or 0.0)
            if _drop != 0.0:
                raise NotImplementedError(
                    f"attn_block_mode='blockmask' has no dropout path (the raw FA3 binding is called with dropout=0); text_config.attention_dropout={_drop}"
                )
        if self._fullseq_diff_blocks:
            for _k in ("diff_head_attention_dropout", "diff_input_process_attention_dropout"):
                _d = float(getattr(config, _k, 0.0) or 0.0)
                if _d != 0.0:
                    raise NotImplementedError(
                        f"hy_fullseq_diff_blocks=True routes the stacks through the block-mask kernel, which has no dropout path; {_k}={_d} would be silently dropped."
                    )
        logger.info_rank0(
            f"[HY] vq retired | cond_vit_aligner={config.cond_vit_aligner}(depth={config.cond_vit_aligner_depth}, merge={self._cond_merge_eff}) | dual_encoder={self._cond_dual} (VAE pre-merge, cms={self._cond_merge}) | attn={self._attn_block_mode} | cond_dual={('causal (ViT reads VAE, VAE does not read ViT)' if self._cond_dual_block_causal else 'merged (one block over [VAE ; ViT])')} | diff_input_process={self._diff_input_process_num_layers} | diff_head={self._diff_head_num_layers} | diff_stacks={('FULL-SEQUENCE (input_process=trunk prefix feeding the backbone; head=diff head over the full seq, AR reads pre-head; dense FFN + gated adaLN, t=0 off-latent)' if self._fullseq_diff_blocks else 'diff-only (latent rows, per-image bidirectional)')} | block_gate={bool(getattr(config, 'diff_block_gate', True))} | t_embed_split={self.diff_time_embed_out is not None}"
        )

    def _build_cond_vit_aligner(self, vision_config, hidden):
        return CondViTAligner(
            vit_hidden=int(vision_config.hidden_size),
            hidden=hidden,
            merge=self._cond_merge_eff,
            depth=self.config.cond_vit_aligner_depth,
            projector_type=self.config.cond_vit_aligner,
            context_dim=int(
                getattr(self.config, "cond_vit_aligner_context_dim", 0)
                or getattr(vision_config, "intermediate_size", 0)
                or 4 * hidden
            ),
            hidden_act=getattr(self.config.text_config, "hidden_act", "silu"),
        )

    def vq_encode_und_image(self, *args, **kwargs):
        raise NotImplementedError(
            "HY replaces the projector chain with cond_vit_aligner — call vq_visual_to_inputs_embeds."
        )

    def vq_visual_to_inputs_embeds(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor
    ) -> torch.FloatTensor:
        encoder = self.model.vq_visual
        k = self._cond_merge
        patch_grid_thw = image_grid_thw.clone()
        patch_grid_thw[:, 1] = image_grid_thw[:, 1] * k
        patch_grid_thw[:, 2] = image_grid_thw[:, 2] * k
        target_dtype = torch.bfloat16 if encoder.dtype == torch.float32 else encoder.dtype
        from .modeling_hy_extras import encode_vision

        feats = encode_vision(encoder, pixel_values.to(target_dtype), patch_grid_thw)
        sm = int(getattr(encoder, "spatial_merge_size", 1) or 1)
        grid_post = patch_grid_thw
        if sm != 1:
            grid_post = patch_grid_thw.clone()
            grid_post[:, 1] = patch_grid_thw[:, 1] // sm
            grid_post[:, 2] = patch_grid_thw[:, 2] // sm
        return self.cond_vit_aligner(feats.to(self.cond_vit_aligner.weight.dtype), grid_post)

    def _vq_pure_text_keepalive(self) -> Optional[torch.Tensor]:
        encoder = self.model.vq_visual
        if encoder is None:
            return None
        if getattr(self, "_hy_zero_vision", False):
            from .base import _eval_only_placeholder as _p_lazy
            can_skip_vision = _p_lazy('can_skip_vision', 'veomni.optimizations.hy.zero_loss，guard: self.training and torch.is_grad_enabled() / _hy_zero_vision')
            zero_parameters = _p_lazy('zero_parameters', 'veomni.optimizations.hy.zero_loss，guard: self.training and torch.is_grad_enabled() / _hy_zero_vision')
            from torch.distributed.fsdp import FSDPModule

            if (isinstance(encoder, FSDPModule) and encoder._get_fsdp_state()._fsdp_param_group.is_sharded
                    and can_skip_vision(encoder, self.cond_vit_aligner)):
                return zero_parameters(encoder, self.cond_vit_aligner)
        k = self._cond_merge
        vc = encoder.config
        _p = vc.patch_size
        if isinstance(_p, (tuple, list)):
            from math import prod as _prod

            _spatial = _prod(_p)
        else:
            _spatial = _p * _p
        patch_dim = vc.num_channels * _spatial
        if hasattr(encoder, "patch_embed"):
            device = encoder.patch_embed.proj.weight.device
        else:
            device = encoder.embeddings.patch_embedding.weight.device
        target_dtype = torch.bfloat16 if encoder.dtype == torch.float32 else encoder.dtype
        dummy_pixels = torch.zeros((k * k, patch_dim), dtype=target_dtype, device=device)
        dummy_grid_thw = torch.tensor([[1, k, k]], dtype=torch.long, device=device)
        saved_checkpointing = []
        if getattr(self, "_hy_fastpath", False) and torch.is_grad_enabled():
            from .base import _eval_only_placeholder as _p_lazy
            can_skip_vision = _p_lazy('can_skip_vision', 'veomni.optimizations.hy.zero_loss，guard: self.training and torch.is_grad_enabled() / _hy_zero_vision')

            if can_skip_vision(encoder, self.cond_vit_aligner):
                saved_checkpointing = [(module, module.gradient_checkpointing) for module in encoder.modules()
                                       if hasattr(module, "gradient_checkpointing")]
                for module, _ in saved_checkpointing:
                    module.gradient_checkpointing = False
        try:
            from .modeling_hy_extras import encode_vision

            feats = encode_vision(encoder, dummy_pixels, dummy_grid_thw)
        finally:
            for module, flag in saved_checkpointing:
                module.gradient_checkpointing = flag
        return self.cond_vit_aligner(feats.to(self.cond_vit_aligner.weight.dtype), dummy_grid_thw).sum()

    @staticmethod
    def _text_mask_to_unified(
        mask_text: torch.Tensor,
        ar_prefix_ranges: torch.Tensor,
        has_diff: list,
        diff_cu_eff: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        ranges = ar_prefix_ranges.tolist()
        cu_eff = diff_cu_eff.tolist() if diff_cu_eff is not None else None
        parts = []
        diff_idx = 0
        for i in range(ar_prefix_ranges.shape[0]):
            s, e = (int(ranges[i][0]), int(ranges[i][1]))
            parts.append(mask_text[s:e])
            if has_diff[i]:
                d_i = int(cu_eff[diff_idx + 1]) - int(cu_eff[diff_idx])
                parts.append(torch.zeros(d_i, dtype=torch.bool, device=device))
                diff_idx += 1
        if not parts:
            return torch.zeros(0, dtype=torch.bool, device=device)
        return torch.cat(parts, dim=0)

    @staticmethod
    def _runs_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.numel() == 0:
            z = torch.zeros(0, dtype=torch.int32, device=mask.device)
            return (z, z)
        m = mask.to(torch.int8)
        pad = torch.zeros(1, dtype=torch.int8, device=mask.device)
        d = torch.diff(torch.cat([pad, m, pad]))
        starts = (d == 1).nonzero(as_tuple=True)[0].to(torch.int32)
        ends = (d == -1).nonzero(as_tuple=True)[0].to(torch.int32)
        return (starts, ends)

    def _build_attn_blocks(
        self,
        cond_unified: Optional[torch.Tensor],
        item_cu: torch.Tensor,
        text_lens: torch.Tensor,
        image_lens: torch.Tensor,
        gen_lens: Optional[torch.Tensor] = None,
        cond_vae_lens: Optional[list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = item_cu.device
        starts, ends = (
            self._runs_from_mask(cond_unified)
            if cond_unified is not None
            else (torch.zeros(0, dtype=torch.int32, device=dev), torch.zeros(0, dtype=torch.int32, device=dev))
        )
        if cond_vae_lens is not None and cond_unified is not None:
            starts, ends = self._split_cond_dual(starts, ends, cond_vae_lens)
        if getattr(self.config, "image_input_mode", "hy") == "native":
            # Native VLM uses causal attention over the image rows too. The
            # generated diffusion blocks below retain their bidirectionality.
            starts, ends = starts[:0], ends[:0]
        d_start = (item_cu[:-1] + text_lens).to(torch.int32)
        d_end = item_cu[1:].to(torch.int32)
        keep = d_end - d_start > 0
        g_start, g_end = (d_start[keep], d_end[keep])
        if gen_lens is not None and gen_lens.numel():
            g_start, g_end = self._split_gen_per_image(g_start, g_end, gen_lens)
        blk_start = torch.cat([starts, g_start])
        blk_end = torch.cat([ends, g_end])
        return (blk_start.contiguous(), blk_end.contiguous())

    @staticmethod
    def _split_cond_dual(
        starts: torch.Tensor, ends: torch.Tensor, vae_lens: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.as_tensor(vae_lens, dtype=starts.dtype, device=starts.device)
        if v.numel() != starts.numel():
            raise ValueError(
                f"{v.numel()} conditional images but {starts.numel()} placeholder runs in the packed sequence — the run-per-image pairing this cut needs does not hold. A multi-frame image_grid_thw entry does exactly this (the data side emits one run PER FRAME while the dual-encoder splice walks one entry per image), and it is already unsupported."
            )
        mid = starts + v
        if bool((mid > ends).any()):
            raise ValueError(
                "a conditional placeholder run is shorter than the VAE part it is supposed to hold — reference VAE grids / image_grid_thw disagree with the placeholders the transform wrote"
            )
        has_vae = v > 0
        blk_start = torch.cat([starts[has_vae], mid])
        blk_end = torch.cat([mid[has_vae], ends])
        return (blk_start.contiguous(), blk_end.contiguous())

    @staticmethod
    def _split_gen_per_image(
        g_start: torch.Tensor, g_end: torch.Tensor, gen_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        span = (g_end - g_start).to(torch.int64)
        gl = gen_lens.to(torch.int64).to(span.device)
        if int(gl.sum()) != int(span.sum()):
            raise ValueError(
                f"generated rows disagree: gen_lens sums to {int(gl.sum())} but the items' diff spans sum to {int(span.sum())} — the latents cannot be a partition of the item layout"
            )
        item_off = span.cumsum(0) - span
        img_off = gl.cumsum(0) - gl
        owner = torch.searchsorted(item_off, img_off, right=True) - 1
        local = img_off - item_off[owner]
        if bool((local + gl > span[owner]).any()):
            raise ValueError("a generated image straddles two items' diff spans")
        s = g_start.to(torch.int64)[owner] + local
        return (s.to(torch.int32), (s + gl).to(torch.int32))

    def _run_layer(
        self,
        decoder_layer,
        hidden_states,
        position_embeddings,
        item_cu,
        text_idx,
        image_idx,
        max_item_len,
        text_cu,
        image_cu,
        max_text_len,
        max_image_len,
        blk_start=None,
        blk_end=None,
    ):
        if self._attn_block_mode != "blockmask":
            return super()._run_layer(
                decoder_layer,
                hidden_states,
                position_embeddings,
                item_cu,
                text_idx,
                image_idx,
                max_item_len,
                text_cu,
                image_cu,
                max_text_len,
                max_image_len,
            )
        assert blk_start is not None and blk_end is not None
        attn = decoder_layer.self_attn
        cp = getattr(self, "_hy_cp_shard", None)
        if cp is not None:
            # Same global boundaries and local Q/K/V exchange as native Transfusion.
            cached_cu = getattr(self, "_hy_cp_item_cu", None)
            item_cu = cached_cu if cached_cu is not None else cp.pad_cu(item_cu)
        gate = getattr(decoder_layer.mlp, "gate", None)
        if gate is not None:
            # Keep padded rows in the compute path, as native MoE does; only
            # exclude them from balancing/monitor counts. Compacting these rows
            # would change the expert GEMM shapes relative to native CP.
            gate._token_mask_for_stats = cp.valid_mask(hidden_states.device) if cp is not None else None
            # Logging is intermittent: a previous logging step may leave global
            # labels with a different pack length. Only shard current labels
            # when a consumer is active (the same condition as their producer).
            classes = (
                getattr(gate, "_router_token_class", None)
                if getattr(gate, "_stats_enabled", False) or getattr(gate, "class_bias_enabled", False)
                else None
            )
            gate._router_token_class_local = cp.local(classes, value=-1) if cp is not None and classes is not None else None
            if cp is not None and classes is not None:
                gate._vision_mask_for_stats = gate._router_token_class_local > 0

        def _layer_fn(hidden_states):
            orig_forward = attn.forward

            def _patched(hidden_states, position_embeddings=None, attention_mask=None, **_kw):
                out = _mla_block_mask_attention(
                    attn, hidden_states, position_embeddings, item_cu, blk_start, blk_end, max_item_len, cp=cp
                )
                return (out, None)

            _saved_gc = decoder_layer.gradient_checkpointing
            decoder_layer.gradient_checkpointing = False
            attn.forward = _patched
            try:
                out = decoder_layer(hidden_states, position_embeddings=position_embeddings)
                return out[0] if isinstance(out, tuple) else out
            finally:
                attn.forward = orig_forward
                decoder_layer.gradient_checkpointing = _saved_gc

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                _layer_fn, hidden_states, use_reentrant=False, **attention_checkpoint_kwargs()
            )
        return _layer_fn(hidden_states)

    def _run_fullseq_diff_block(
        self,
        blk,
        hidden_states,
        t_emb_table,
        t_row_ids,
        cos_pre,
        sin_pre,
        item_cu,
        blk_start,
        blk_end,
        max_item_len: int,
    ):
        skip_checkpoint = (getattr(self, "_hy_fastpath", False)
                           and not getattr(self, "_hy_checkpoint_diff_heads", True)
                           and any(blk is head for head in (self.diff_head_blocks or ())))
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled() and not skip_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                blk,
                hidden_states,
                t_emb_table,
                t_row_ids,
                cos_pre,
                sin_pre,
                item_cu,
                blk_start,
                blk_end,
                max_item_len,
                use_reentrant=False,
            )
        return blk(hidden_states, t_emb_table, t_row_ids, cos_pre, sin_pre, item_cu, blk_start, blk_end, max_item_len)

    def _encode_image_entry(
        self,
        latents: torch.Tensor,
        grid_thw: torch.Tensor,
        cu_seqlens: torch.Tensor,
        timesteps: torch.Tensor,
        pos_anchors: Optional[torch.Tensor],
        vit_grids: Optional[torch.Tensor],
        per_token_timesteps: bool = False,
    ):
        x = self.diff_input_proj(latents.to(self.diff_input_proj.weight.dtype))
        pos_ids = self._build_diff_position_ids(
            grid_thw, cu_seqlens, x.device, diff_pos_anchors=pos_anchors, diff_vit_grids=vit_grids
        )
        _ts_in = timesteps
        if self._fullseq_diff_blocks:
            _ts_in = torch.cat([torch.zeros(1, dtype=timesteps.dtype, device=timesteps.device), timesteps], dim=0)
        _t_emb_raw = self.diff_time_embed(_ts_in)
        t_emb_item = _t_emb_raw if self._adaln_fp32 else _t_emb_raw.to(x.dtype)
        n_items = cu_seqlens.shape[0] - 1
        per_item = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        if per_token_timesteps:
            if timesteps.shape[0] != x.shape[0]:
                raise ValueError(
                    f"per_token_timesteps=True requires one τ per token: got {timesteps.shape[0]} vs L={x.shape[0]}"
                )
            item_ids = torch.arange(x.shape[0], device=x.device)
        else:
            item_ids = torch.repeat_interleave(torch.arange(n_items, device=x.device), per_item)
        if self._fullseq_diff_blocks:
            item_ids = item_ids + 1
        if self.diff_input_process_blocks is not None and (not self._fullseq_diff_blocks):
            rot_emb = self.model.rotary_emb
            cos3d, sin3d = _rope_tables(rot_emb, pos_ids, x, dtype=self._rope_table_dtype)
            is_img = torch.ones(pos_ids.shape[-1], dtype=torch.bool, device=x.device)
            cos_pre, sin_pre = self._assemble_mrope(cos3d, sin3d, is_img)
            cu_i32 = cu_seqlens.to(torch.int32)
            max_len = int(per_item.max().item())
            h = x.unsqueeze(0)
            for blk in self.diff_input_process_blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(
                        blk, h, t_emb_item, item_ids, cos_pre, sin_pre, cu_i32, max_len, use_reentrant=False
                    )
                else:
                    h = blk(h, t_emb_item, item_ids, cos_pre, sin_pre, cu_i32, max_len)
            x = h.squeeze(0)
        sc_in, sh_in = self.diff_adaln_in(t_emb_item).index_select(0, item_ids).chunk(2, dim=-1)
        x = _adaln_apply(self.diff_input_norm, x, sc_in, sh_in, self._adaln_fp32)
        return (x, t_emb_item, item_ids, pos_ids)


@torch.no_grad()
def diff_time_embed_out_in_weights(model_path: Optional[str]) -> bool:
    import json
    import os
    import re

    if not model_path or not os.path.isdir(model_path):
        return False
    pat = re.compile("^(?:model\\.)?diff_time_embed_out\\.")
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            keys = json.load(f).get("weight_map", {}).keys()
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(single):
            return False
        from safetensors import safe_open

        with safe_open(single, "pt") as f:
            keys = list(f.keys())
    return any((pat.match(k) for k in keys))


@torch.no_grad()
def clone_diff_time_embed_out(model: nn.Module) -> bool:
    src = getattr(model, "diff_time_embed", None)
    dst = getattr(model, "diff_time_embed_out", None)
    if src is None or dst is None:
        return False
    if str(getattr(model.config, "diff_time_embed_out_init", "clone")) != "clone":
        return False
    s_params = dict(src.named_parameters())
    for name, p_dst in dst.named_parameters():
        p_src = s_params.get(name)
        assert p_src is not None and p_src.shape == p_dst.shape, (
            f"diff_time_embed_out.{name} has no matching source parameter (shapes {(None if p_src is None else tuple(p_src.shape))} vs {tuple(p_dst.shape)})"
        )
        _local(p_dst).copy_(_local(p_src))
    return True


class TransfusionSelfFlowForwardMixin:
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_per_token_loss: bool = False,
        return_logits: bool = True,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        placeholder_token_id: Optional[int] = None,
        precomputed_vq_z_q: Optional[torch.FloatTensor] = None,
        vae_noised_latents: Optional[torch.Tensor] = None,
        diff_timesteps: Optional[torch.Tensor] = None,
        diff_target: Optional[torch.Tensor] = None,
        diff_flow_sigmas: Optional[torch.Tensor] = None,
        diff_grid_thw: Optional[torch.LongTensor] = None,
        diff_cu_seqlens: Optional[torch.LongTensor] = None,
        ar_prefix_ranges: Optional[torch.LongTensor] = None,
        diff_pos_anchors: Optional[torch.LongTensor] = None,
        diff_vit_grids: Optional[torch.LongTensor] = None,
        is_ref_vae: Optional[torch.BoolTensor] = None,
        reference_image_mask: Optional[torch.Tensor] = None,
        diff_timesteps_per_token: Optional[torch.Tensor] = None,
        selfflow_teacher_mode: bool = False,
        selfflow_teacher_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "GlmImageMoeTransfusionSelfFlowOutput":
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            text_embeds = self.get_input_embeddings()(input_ids)
        else:
            text_embeds = inputs_embeds
        device = text_embeds.device
        # Native callers may interleave references and targets across samples.
        # Retain the established internal projection order (references first),
        # and restore the caller's token order for inference outputs below.
        _vae_output_order = None
        _reference_count = (int(is_ref_vae.sum().item())
                            if vae_noised_latents is not None and diff_grid_thw is not None
                            and is_ref_vae is not None else 0)
        if _reference_count and (diff_target is None or diff_target.shape[0] == vae_noised_latents.shape[0]):
            _reference_rows = is_ref_vae.to(device=device, dtype=torch.bool)
            _source_lengths = diff_grid_thw.prod(-1).to(torch.long)
            _source_references = _reference_rows.repeat_interleave(_source_lengths)
            if diff_target is not None and diff_target.shape[0] == vae_noised_latents.shape[0]:
                diff_target = diff_target[~_source_references]
            if diff_flow_sigmas is not None and diff_flow_sigmas.numel() == diff_grid_thw.shape[0]:
                diff_flow_sigmas = diff_flow_sigmas[~_reference_rows]
            _image_order = torch.cat((_reference_rows.nonzero().flatten(), (~_reference_rows).nonzero().flatten()))
            if not torch.equal(_image_order, torch.arange(_image_order.numel(), device=device)):
                _source_cu = torch.cat((_source_lengths.new_zeros(1), _source_lengths.cumsum(0)))
                _token_order = torch.cat([
                    torch.arange(_source_cu[i], _source_cu[i + 1], device=device) for i in _image_order.tolist()
                ])
                _vae_output_order = _token_order.argsort()
                vae_noised_latents = vae_noised_latents.index_select(0, _token_order)
                diff_grid_thw = diff_grid_thw.index_select(0, _image_order)
                is_ref_vae = _reference_rows.index_select(0, _image_order)
                if diff_timesteps is not None:
                    diff_timesteps = diff_timesteps.index_select(0, _image_order)
                if diff_timesteps_per_token is not None:
                    diff_timesteps_per_token = diff_timesteps_per_token.index_select(0, _token_order)
                if diff_pos_anchors is not None:
                    diff_pos_anchors = diff_pos_anchors.index_select(0, _image_order)
                if diff_vit_grids is not None:
                    diff_vit_grids = diff_vit_grids.index_select(0, _image_order)
                _lengths = diff_grid_thw.prod(-1)
                diff_cu_seqlens = torch.cat((_lengths.new_zeros(1), _lengths.cumsum(0)))
        # The native VAE input covers references and targets. Packed SelfFlow
        # supplies reference rows first, retaining the previous GEMM row order.
        _vae_latents, _vae_grids = vae_noised_latents, diff_grid_thw
        _vae_times, _vae_token_times = diff_timesteps, diff_timesteps_per_token
        _vae_anchors, _vae_spatial = diff_pos_anchors, diff_vit_grids
        _n_cond, _entry_offset = 0, 0
        reference_latents = reference_grid_thw = None
        if _vae_latents is not None and _vae_grids is not None:
            _vae_lengths = _vae_grids.prod(-1).to(torch.long)
            _vae_cu = (diff_cu_seqlens if diff_cu_seqlens is not None else
                       torch.cat((_vae_lengths.new_zeros(1), _vae_lengths.cumsum(0))))
            if is_ref_vae is not None:
                _n_cond = _reference_count
                _entry_offset = int(_vae_cu[_n_cond].item())
            if _n_cond:
                reference_latents = _vae_latents[:_entry_offset]
                reference_grid_thw = _vae_grids[:_n_cond]
            vae_noised_latents = _vae_latents[_entry_offset:]
            diff_grid_thw = _vae_grids[_n_cond:]
            diff_cu_seqlens = _vae_cu[_n_cond:] - _entry_offset
            diff_timesteps = _vae_times[_n_cond:] if _vae_times is not None else None
            diff_timesteps_per_token = (_vae_token_times[_entry_offset:]
                                        if _vae_token_times is not None else None)
            diff_pos_anchors = _vae_anchors[_n_cond:] if _vae_anchors is not None else None
            diff_vit_grids = _vae_spatial[_n_cond:] if _vae_spatial is not None else None
        assert not (pixel_values is not None and precomputed_vq_z_q is not None), (
            "pixel_values and precomputed_vq_z_q are mutually exclusive"
        )
        run_pixel = pixel_values is not None and self.model.vq_visual is not None
        assert precomputed_vq_z_q is None, (
            "HY replaces the projector chain with cond_vit_aligner, so a precomputed projector output carries no meaning here — pass pixel_values instead."
        )
        run_index = False
        _hy106_optimized = getattr(self, "_hy106_optimized_runtime", False)
        _hy70_sparse_ar = (self.config.model_type == "glm5_next_hy_selfflow"
                           and _os_lmh.environ.get("VEOMNI_HY70_SPARSE_AR", "0") == "1")
        if _hy106_optimized or _hy70_sparse_ar:
            # Reuse the early supervision fence; avoid nonzero after the decoder.
            _ar_supervised_indices = (
                (labels.reshape(-1) != -100).nonzero(as_tuple=True)[0]
                if labels is not None and not getattr(self, "_hy_zero_ar", False) else None
            )
            _labels_have_supervision = _ar_supervised_indices is not None and _ar_supervised_indices.numel() > 0
        else:
            _labels_have_supervision = (
                False if getattr(self, "_hy_zero_ar", False)
                else bool((labels != -100).any()) if labels is not None else False
            )
        run_cond_vae = (
            self._cond_dual
            and reference_latents is not None
            and reference_grid_thw is not None
        )
        assert not (run_cond_vae and (not run_pixel)), (
            "conditional VAE latents arrived without ViT pixel_values — the dual encoder emits BOTH runs for every conditional image."
        )
        vq_keepalive: Optional[torch.Tensor] = None
        if not run_pixel and (not run_index) and not (getattr(self, "_hy_fastpath", False) and selfflow_teacher_mode):
            vq_keepalive = self._vq_pure_text_keepalive()
            if vq_keepalive is None and self.vq_to_llm_proj is not None:
                z = torch.zeros(
                    (1, self.vq_to_llm_proj.in_features),
                    dtype=self.vq_to_llm_proj.weight.dtype,
                    device=self.vq_to_llm_proj.weight.device,
                )
                vq_keepalive = self.vq_to_llm_proj(z).sum()
        cond_vit_embed: Optional[torch.Tensor] = None
        if run_pixel:
            assert input_ids is not None, "conditional-image branch requires input_ids"
            assert placeholder_token_id is not None, "conditional-image branch requires placeholder_token_id"
            cond_vit_embed = self.vq_visual_to_inputs_embeds(pixel_values, image_grid_thw)
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention")
            )
            if run_pixel and getattr(self.config, "image_input_mode", "hy") == "native":
                position_ids = self.native_image_position_ids(input_ids, image_grid_thw, attention_mask_tensor)
            else:
                position_ids = NativeBackbone.get_rope_index(
                    self.model, input_ids, image_grid_thw, reference_grid_thw,
                    attention_mask=attention_mask_tensor, image_grid_is_merged=True,
                )
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[:3]
        text_pos = position_ids.float()[:, 0, :]
        run_diff = (
            self.diff_enabled
            and vae_noised_latents is not None
            and (diff_timesteps is not None)
            and (diff_grid_thw is not None)
            and diff_grid_thw.shape[0] > 0
            and (diff_cu_seqlens is not None)
            and (ar_prefix_ranges is not None)
        )
        diff_h_seq = None
        diff_pos = None
        diff_cu_eff = diff_cu_seqlens
        diff_t_token_mask: Optional[torch.Tensor] = None
        t_emb_item = None
        diff_item_ids = None
        _item_ids_all = None
        _split = 0
        _hy_all_ts = None
        _diff_pre_ka: Optional[torch.Tensor] = None
        _diff_post_ka: Optional[torch.Tensor] = None
        _fs_t_in: Optional[torch.Tensor] = None
        _fs_t_out: Optional[torch.Tensor] = None
        _fs_t_ids: Optional[torch.Tensor] = None
        run_entry = run_diff or run_cond_vae
        cond_vae_embed: Optional[torch.Tensor] = None
        if run_entry:
            assert self.config.diff_time_embed_mode == "adaln", (
                f"HY requires diff_time_embed_mode='adaln' (got {self.config.diff_time_embed_mode!r}); no timestep token is appended."
            )
            _all_lat, _all_grid = _vae_latents, _vae_grids
            _all_len, _all_cu = _vae_lengths, _vae_cu
            _all_ts = _vae_times.to(torch.float32)
            _hy_all_ts = _vae_token_times if _vae_token_times is not None else _all_ts
            _sf_per_token = _vae_token_times is not None
            _max_diff_len = (int(diff_grid_thw.prod(-1).max().item())
                             if run_diff and self.diff_head_blocks is not None
                             and not self._fullseq_diff_blocks else 0)
            _x_all, t_emb_item, _item_ids_all, _pos_all = self._encode_image_entry(
                _all_lat,
                _all_grid,
                _all_cu,
                _hy_all_ts,
                _vae_anchors if _vae_anchors is not None else _all_grid.new_zeros(_all_grid.shape[0]),
                _vae_spatial if _vae_spatial is not None else _all_grid,
                per_token_timesteps=_sf_per_token,
            )
            if _hy106_optimized:
                from .modeling_hy_extras import entry_split
                _split = entry_split(reference_latents if run_cond_vae else None, _all_cu, _n_cond)
            else:
                _split = int(_all_cu[_n_cond].item())
            if run_cond_vae:
                cond_vae_embed = _x_all[:_split]
            if run_diff:
                x_t_proj = _x_all[_split:]
                diff_item_ids = _item_ids_all[_split:]
                diff_pos_ids = _pos_all[:, :, _split:]
                diff_h_seq = x_t_proj.unsqueeze(0)
                diff_pos = diff_pos_ids[:, 0, :]
        elif self.diff_enabled and (self.diff_input_proj is not None or self.diff_time_embed is not None):
            _hs = self.config.text_config.hidden_size
            _z_t = torch.zeros((1,), dtype=torch.float32, device=self.diff_time_embed.mlp[0].weight.device)
            _t_embed = self.diff_time_embed(_z_t)
            _hy_all_ts = _z_t
            _diff_pre_ka = _t_embed.sum()
            _fs_t_in = _t_embed
            if self.diff_input_proj is not None:
                _patch_ch = self.config.diff_vae_latent_dim * self.config.diff_patch_size**2
                _w = self.diff_input_proj.weight
                _z_pix = torch.zeros((1, _patch_ch), dtype=_w.dtype, device=_w.device)
                _x_t_proj = self.diff_input_proj(_z_pix)
                _diff_pre_ka = _diff_pre_ka + _x_t_proj.sum()
            if self.diff_input_process_blocks is not None and (not self._fullseq_diff_blocks):
                _dev = self.diff_time_embed.mlp[0].weight.device
                _dt = self.diff_time_embed.mlp[0].weight.dtype
                _dum_grid_thw_ip = torch.tensor([[1, 1, 1]], dtype=torch.long, device=_dev)
                _dum_cu_ip = torch.tensor([0, 1], dtype=torch.long, device=_dev)
                _dum_pos_ids_ip = self._build_diff_position_ids(_dum_grid_thw_ip, _dum_cu_ip, _dev)
                _rot_emb_ip = self.model.rotary_emb
                _dum_cos3d_ip, _dum_sin3d_ip = _rope_tables(
                    _rot_emb_ip, _dum_pos_ids_ip, torch.empty(1, dtype=_dt, device=_dev), dtype=self._rope_table_dtype
                )
                _dum_is_img_ip = torch.ones(1, dtype=torch.bool, device=_dev)
                _dum_cos_pre_ip, _dum_sin_pre_ip = self._assemble_mrope(_dum_cos3d_ip, _dum_sin3d_ip, _dum_is_img_ip)
                _dum_cu_i32_ip = _dum_cu_ip.to(torch.int32)
                _dum_item_ids_ip = torch.zeros(1, dtype=torch.long, device=_dev)
                _dum_ip_x = torch.zeros(1, 1, _hs, dtype=_dt, device=_dev)
                for _blk in self.diff_input_process_blocks:
                    if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                        _dum_ip_x = torch.utils.checkpoint.checkpoint(
                            _blk,
                            _dum_ip_x,
                            _t_embed,
                            _dum_item_ids_ip,
                            _dum_cos_pre_ip,
                            _dum_sin_pre_ip,
                            _dum_cu_i32_ip,
                            1,
                            use_reentrant=False,
                        )
                    else:
                        _dum_ip_x = _blk(
                            _dum_ip_x, _t_embed, _dum_item_ids_ip, _dum_cos_pre_ip, _dum_sin_pre_ip, _dum_cu_i32_ip, 1
                        )
                _diff_pre_ka = _diff_pre_ka + _dum_ip_x.sum()
            if self._use_adaln and self.diff_adaln_in is not None:
                _sc_in_z, _sh_in_z = self.diff_adaln_in(_t_embed).chunk(2, dim=-1)
                _base_z = torch.zeros_like(_sh_in_z)
                _diff_pre_ka = (
                    _diff_pre_ka
                    + _adaln_apply(self.diff_input_norm, _base_z, _sc_in_z, _sh_in_z, self._adaln_fp32).sum()
                )
        _cond_vae_lens: Optional[list] = None
        cond_token_mask = None
        if cond_vit_embed is not None:
            vit_mask = input_ids == placeholder_token_id
            cond_token_mask = vit_mask
            if cond_vae_embed is not None:
                vae_lengths = iter(reference_grid_thw.prod(dim=-1).tolist())
                has_vae = (reference_image_mask.to(torch.bool).tolist() if reference_image_mask is not None
                           else [True] * image_grid_thw.shape[0])
                if len(has_vae) != image_grid_thw.shape[0]:
                    raise ValueError('reference_image_mask does not match the number of conditional images')
                _cond_vae_lens = [int(next(vae_lengths)) if present else 0 for present in has_vae]
                if next(vae_lengths, None) is not None:
                    raise ValueError('Conditional VAE grids do not match reference_image_mask')
            if _hy106_optimized:
                from .modeling_hy_extras import scatter

                text_embeds = scatter(text_embeds, vit_mask, cond_vit_embed.to(text_embeds.dtype))
            else:
                text_embeds = text_embeds.masked_scatter(
                    vit_mask.unsqueeze(-1), cond_vit_embed.to(text_embeds.dtype)
                )
            if cond_vae_embed is not None:
                vae_mask = input_ids == self.config.vae_token_id
                cond_token_mask = cond_token_mask | vae_mask
                if _hy106_optimized:
                    from .modeling_hy_extras import scatter

                    text_embeds = scatter(text_embeds, vae_mask, cond_vae_embed.to(text_embeds.dtype))
                else:
                    text_embeds = text_embeds.masked_scatter(
                        vae_mask.unsqueeze(-1), cond_vae_embed.to(text_embeds.dtype)
                    )
        _has_diff_forward = kwargs.get("has_diff_per_item")
        _ar_prefix_from_cu: Optional[torch.Tensor] = None
        if not run_diff:
            _cu = kwargs.get("ar_cu_seqlens")
            if _cu is not None:
                _cu = _cu.to(torch.long)
                _N = _cu.shape[0] - 1
                if _N > 0:
                    _ar_prefix_from_cu = torch.stack([torch.stack([_cu[i], _cu[i + 1]]) for i in range(_N)], dim=0)
                    if _has_diff_forward is None:
                        _has_diff_forward = torch.zeros(_N, dtype=torch.bool, device=_cu.device)
        (
            inputs_embeds_u,
            position_ids_u,
            is_image,
            t_token_in_unified,
            item_cu,
            max_item_len,
            text_cu,
            image_cu,
            max_text_len,
            max_image_len,
        ) = self._interleave(
            text_embeds,
            text_pos,
            diff_h_seq,
            diff_pos,
            ar_prefix_ranges if run_diff else _ar_prefix_from_cu,
            diff_cu_eff if run_diff else None,
            diff_t_token_mask,
            device,
            has_diff_per_item=_has_diff_forward,
        )
        blk_start = blk_end = None
        _cond_unified = None
        if self._attn_block_mode == "blockmask":
            _ar_ranges = ar_prefix_ranges if run_diff else _ar_prefix_from_cu
            if run_pixel and input_ids is not None and (_ar_ranges is not None):
                _hd = (
                    _has_diff_forward.to(torch.bool).tolist()
                    if _has_diff_forward is not None
                    else [bool(run_diff)] * _ar_ranges.shape[0]
                )
                _cond_unified = self._text_mask_to_unified(
                    cond_token_mask[0], _ar_ranges, _hd, diff_cu_eff if run_diff else None, device
                )
            blk_start, blk_end = self._build_attn_blocks(
                _cond_unified,
                item_cu,
                text_cu[1:] - text_cu[:-1],
                image_cu[1:] - image_cu[:-1],
                diff_cu_eff[1:] - diff_cu_eff[:-1] if run_diff and diff_cu_eff is not None else None,
                cond_vae_lens=_cond_vae_lens if self._cond_dual_block_causal else None,
            )
        if _diff_pre_ka is not None:
            inputs_embeds_u = inputs_embeds_u + 0.0 * _diff_pre_ka.to(inputs_embeds_u.dtype)
            _diff_pre_ka = None
        position_embeddings = _rope_tables(
            self.model.rotary_emb, position_ids_u, inputs_embeds_u, dtype=self._rope_table_dtype
        )
        image_rope_mask = is_image
        if run_pixel:
            cond_rope_mask = _cond_unified
            if cond_rope_mask is None:
                # Two-pass attention and single-item inference also need the
                # conditional image rows, even without conditional block bounds.
                cond_rope_mask = cond_token_mask[0]
                rope_ranges = ar_prefix_ranges if run_diff else _ar_prefix_from_cu
                if rope_ranges is not None:
                    has_diff = (_has_diff_forward.to(torch.bool).tolist() if _has_diff_forward is not None
                                else [bool(run_diff)] * rope_ranges.shape[0])
                    cond_rope_mask = self._text_mask_to_unified(
                        cond_rope_mask, rope_ranges, has_diff, diff_cu_eff if run_diff else None, device
                    )
            image_rope_mask = is_image | cond_rope_mask
        cos3, sin3 = position_embeddings
        cos_pre, sin_pre = self._assemble_mrope(cos3, sin3, image_rope_mask)
        position_embeddings = (cos_pre, sin_pre)
        is_text = ~is_image
        text_idx = is_text.nonzero(as_tuple=True)[0]
        image_idx = is_image.nonzero(as_tuple=True)[0]
        if not hasattr(self, "_routing_stats_gates"):
            self._routing_stats_gates = [
                module for layer in self.model.layers for module in layer.modules()
                if hasattr(module, "local_tokens_per_expert")
            ]
        _stats_gates = self._routing_stats_gates
        if any(getattr(gate, "_stats_enabled", False) or getattr(gate, "class_bias_enabled", False)
               for gate in _stats_gates):
            _stats_cls = torch.where(
                is_image & ~t_token_in_unified,
                torch.full_like(is_image, 2, dtype=torch.int8),
                torch.zeros_like(is_image, dtype=torch.int8),
            )
            if _cond_unified is not None:
                _stats_cls = torch.where(_cond_unified, torch.ones_like(_stats_cls), _stats_cls)
            if float(kwargs.get("diff_loss_scale", 1.0)) == 0.0:
                _stats_cls = _stats_cls.masked_fill(is_image & ~t_token_in_unified, -1)
            for gate in _stats_gates:
                gate._router_token_class = _stats_cls
        if self._fullseq_diff_blocks:
            if t_emb_item is not None:
                _fs_t_in = t_emb_item
            if _fs_t_in is None:
                raise RuntimeError(
                    "hy_fullseq_diff_blocks: no timestep table on this rank — neither the image entry chain nor the diff pre-keepalive ran, so diff_time_embed never fired. The full-sequence stacks cannot be conditioned and the ranks that DID take one of those paths would deadlock against this one."
                )
            if not self._adaln_fp32:
                _fs_t_in = _fs_t_in.to(inputs_embeds_u.dtype)
            _fs_t_ids = torch.zeros(inputs_embeds_u.shape[1], dtype=torch.long, device=inputs_embeds_u.device)
            if run_diff and diff_item_ids is not None:
                _fs_lat_idx = (is_image & ~t_token_in_unified).nonzero(as_tuple=True)[0]
                _fs_t_ids = _fs_t_ids.index_copy(0, _fs_lat_idx, diff_item_ids)
            if self.diff_time_embed_out is not None and not (getattr(self, "_hy_fastpath", False) and selfflow_teacher_mode):
                if t_emb_item is not None:
                    _fs_ts_out = torch.cat(
                        [torch.zeros(1, dtype=_hy_all_ts.dtype, device=_hy_all_ts.device), _hy_all_ts], dim=0
                    )
                else:
                    _fs_ts_out = torch.zeros(
                        1, dtype=torch.float32, device=self.diff_time_embed_out.mlp[0].weight.device
                    )
                _fs_raw_out = self.diff_time_embed_out(_fs_ts_out)
                _fs_t_out = _fs_raw_out if self._adaln_fp32 else _fs_raw_out.to(inputs_embeds_u.dtype)
            else:
                _fs_t_out = _fs_t_in
        if self._fullseq_diff_blocks and self.diff_input_process_blocks is not None:
            _h_pre_fs = inputs_embeds_u
            for _blk_fs in self.diff_input_process_blocks:
                _h_pre_fs = self._run_fullseq_diff_block(
                    _blk_fs,
                    _h_pre_fs,
                    _fs_t_in,
                    _fs_t_ids,
                    cos_pre,
                    sin_pre,
                    item_cu,
                    blk_start,
                    blk_end,
                    max_item_len,
                )
            inputs_embeds_u = _h_pre_fs
        _backbone_layout = {}
        _prepare_attention = getattr(self, "prepare_backbone_attention_layout", None)
        if _prepare_attention is not None:
            _backbone_layout.update(_prepare_attention(
                item_cu=item_cu, block_start=blk_start, block_end=blk_end,
                position_ids=position_ids_u, is_image=is_image,
            ))
        _prepare_layout = getattr(self, "prepare_backbone_token_layout", None)
        if _prepare_layout is not None:
            _backbone_layout = _prepare_layout(
                is_image=is_image, cond_unified=_cond_unified,
                cond_vae_lens=_cond_vae_lens, image_grid_thw=image_grid_thw,
            )
        _prepare_time = getattr(self, "prepare_backbone_time_condition", None)
        if _prepare_time is not None:
            _backbone_layout.update(_prepare_time(
                _backbone_layout, time_embeddings=t_emb_item if t_emb_item is not None else _fs_t_in,
                time_rows=_item_ids_all, cond_token_count=_split, is_image=is_image,
                time_values=_hy_all_ts,
            ))
        _selfflow_student_hidden: Optional[torch.Tensor] = None
        hidden_states = inputs_embeds_u
        _hy_cp = None
        if getattr(self, "_hy106_optimized_runtime", False) or self.config.model_type in (
            "glm_image_moe_transfusion_hy", "glm_image_moe_transfusion_hy_selfflow"
        ):
            from .modeling_hy_extras import transfusion_cp_shard
            _hy_cp = transfusion_cp_shard(hidden_states.shape[1])
        self._hy_cp_shard = _hy_cp
        self._hy_cp_stats_scale = 1.0
        if _hy_cp is not None:
            if self._fullseq_diff_blocks:
                raise ValueError("HY106 CP does not yet support full-sequence diffusion blocks")
            import os as _hy_cp_os
            if (hidden_states.device.type == 'npu' and self.config._attn_implementation == 'fa3_block'
                    and _hy_cp_os.environ.get('VEOMNI_HY106_CP_INDEPENDENT_ITEMS', '0') == '1'):
                from .base import _eval_only_placeholder as _p_lazy
                independent_items = _p_lazy('independent_items', 'veomni.optimizations.hy106.context_parallel，guard: _hy_cp is not None（单进程下 transfusion_cp_shard 返回 None）')
                _base_cp = _hy_cp
                _hy_cp = independent_items(_base_cp, item_cu)
                self._hy_cp_shard = _hy_cp
                if _hy_cp is None:
                    self._hy_cp_stats_scale = 1.0 / _base_cp.size
        if _hy_cp is not None:
            if torch.is_grad_enabled() and _hy_cp_os.environ.get("VEOMNI_HY106_CP_CANONICAL_INPUT", "0") == "1":
                from .base import _eval_only_placeholder as _p_lazy
                canonical_input = _p_lazy('canonical_input', 'veomni.optimizations.hy106.context_parallel，guard: _hy_cp is not None（单进程下 transfusion_cp_shard 返回 None）')
                hidden_states = canonical_input(hidden_states, _hy_cp)
            hidden_states = _hy_cp.local(hidden_states, dim=1)
            position_embeddings = tuple(_hy_cp.local(p, dim=-2).contiguous() for p in position_embeddings)
        # Cache once per forward, matching native Transfusion's stable metadata
        # identity across all layers (including checkpoint recomputation).
        self._hy_cp_item_cu = _hy_cp.pad_cu(item_cu) if _hy_cp is not None else None
        _pack_hidden = getattr(self, "pack_backbone_hidden", None)
        _unpack_hidden = getattr(self, "unpack_backbone_hidden", None)
        if _pack_hidden is not None:
            hidden_states = _pack_hidden(hidden_states, _backbone_layout)
        for _sf_layer_idx, decoder_layer in enumerate(self.model.layers):
            hidden_states = self._run_layer(
                decoder_layer,
                hidden_states,
                position_embeddings,
                item_cu,
                text_idx,
                image_idx,
                max_item_len,
                text_cu,
                image_cu,
                max_text_len,
                max_image_len,
                blk_start,
                blk_end,
                **_backbone_layout,
            )
            if (
                not selfflow_teacher_mode
                and self.config.selfflow_enabled
                and (_sf_layer_idx == self._selfflow_student_layer_idx)
            ):
                _selfflow_student_hidden = (
                    _hy_cp.gather(hidden_states[0]).unsqueeze(0) if _hy_cp is not None
                    else _unpack_hidden(hidden_states, _backbone_layout) if _unpack_hidden is not None
                    else hidden_states
                )
            if selfflow_teacher_mode and _sf_layer_idx == self._selfflow_teacher_layer_idx:
                if not run_diff:
                    raise RuntimeError(
                        "Self-Flow teacher 模式要求 run_diff=True（vae_noised_latents / diff_timesteps / diff_grid_thw / diff_cu_seqlens / ar_prefix_ranges 必须都给）"
                    )
                _sf_lat_idx = (is_image & ~t_token_in_unified).nonzero(as_tuple=True)[0]
                _teacher_hidden = (
                    _hy_cp.gather(hidden_states[0]) if _hy_cp is not None
                    else _unpack_hidden(hidden_states, _backbone_layout)[0] if _unpack_hidden is not None
                    else hidden_states[0]
                )
                if getattr(_hy_cp, 'independent_attention', False):
                    _hy_cp.release_attention_metadata()
                return GlmImageMoeTransfusionSelfFlowOutput(
                    loss=None, teacher_feats=_teacher_hidden.index_select(0, _sf_lat_idx)
                )
        if _unpack_hidden is not None:
            hidden_states = _unpack_hidden(hidden_states, _backbone_layout)
        if _hy_cp is not None:
            hidden_states = _hy_cp.gather(hidden_states[0]).unsqueeze(0)
        _h_post_fs: Optional[torch.Tensor] = None
        if self._fullseq_diff_blocks and self.diff_head_blocks is not None:
            _h_fs_out = hidden_states
            for _blk_fs_out in self.diff_head_blocks:
                _h_fs_out = self._run_fullseq_diff_block(
                    _blk_fs_out,
                    _h_fs_out,
                    _fs_t_out,
                    _fs_t_ids,
                    cos_pre,
                    sin_pre,
                    item_cu,
                    blk_start,
                    blk_end,
                    max_item_len,
                )
            _h_post_fs = _h_fs_out[0]
        _probe_h = hidden_states[0]
        text_hidden = self.model.norm(_probe_h.index_select(0, text_idx))
        ar_loss = None
        ar_logits = None
        per_token_loss = None
        _lmh_ka: Optional[torch.Tensor] = None
        if labels is None:
            if return_logits:
                ar_logits = self.lm_head(text_hidden)
        elif _labels_have_supervision:
            from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.loss import fused_linear_cross_entropy as _fused
            from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.loss import fused_linear_cross_entropy_none as _fused_none
            from .base import is_seed_kernels_available

            lbl_flat = labels.reshape(-1)
            if (getattr(self, "_hy_fastpath", False)
                    or _os_lmh.environ.get("VEOMNI_FSDP_SPLIT_LM_HEAD", "0") == "1"):
                _lmh_ka = self.lm_head(text_hidden[:1])
            if _fused is not None:
                ar_features, ar_labels = text_hidden, lbl_flat
                if _hy106_optimized or _hy70_sparse_ar:
                    from .modeling_hy_extras import select
                    ar_features, ar_labels = select(text_hidden, lbl_flat, _ar_supervised_indices)
                if is_seed_kernels_available():
                    ar_loss = _fused(ar_features, self.lm_head.weight, ar_labels, ignore_index=-100)
                else:
                    ar_loss = _fused(self.lm_head.weight, ar_features, ar_labels)
            else:
                logger.warning_once(
                    f"fused_linear_cross_entropy unavailable (liger/seed kernels missing) — falling back to full-logits CE over {text_hidden.shape[0]}x{self.lm_head.weight.shape[0]} logits. Expect OOM at train seqlen."
                )
                logits_flat = F.linear(text_hidden, self.lm_head.weight).float()
                ar_loss = F.cross_entropy(logits_flat, lbl_flat, ignore_index=-100)
            if return_per_token_loss and _fused_none is not None:
                with torch.no_grad():
                    per_token_loss = _fused_none(self.lm_head.weight.detach(), text_hidden.detach(), lbl_flat)
        else:
            if getattr(self, "_hy_fastpath", False):
                if getattr(self, "_hy_zero_ar", False):
                    from .base import _eval_only_placeholder as _p_lazy
                    zero_parameters = _p_lazy('zero_parameters', 'veomni.optimizations.hy.zero_loss，guard: self.training and torch.is_grad_enabled() / _hy_zero_vision')

                    _lmh_ka = zero_parameters(self.lm_head) + 0.0 * text_hidden[:1].float().sum()
                else:
                    _lmh_ka = self.lm_head(text_hidden[:1])
            elif _hy106_optimized:
                # Other ranks may have AR supervision. Preserve the same head
                # collectives and zero gradient path for an all-ignored pack.
                _lmh_ka = self.lm_head(text_hidden[:1])
            elif self.training and not return_logits:
                # Preserve head/norm zero gradients and FSDP participation
                # without materializing unused [AR tokens, vocabulary] logits.
                _lmh_ka = self.lm_head(text_hidden[:1])
            elif return_logits:
                ar_logits = self.lm_head(text_hidden)
        diff_loss = None
        diff_pred = None
        if run_diff:
            image_latent_mask = is_image & ~t_token_in_unified
            image_latent_idx = image_latent_mask.nonzero(as_tuple=True)[0]
            _diff_src_h = _h_post_fs if _h_post_fs is not None else _probe_h
            image_hidden_raw = _diff_src_h.index_select(0, image_latent_idx)
            if self._fullseq_diff_blocks:
                t_emb_item_out = _fs_t_out
            elif self.diff_time_embed_out is not None:
                _t_out_raw = self.diff_time_embed_out(_hy_all_ts)
                t_emb_item_out = _t_out_raw if self._adaln_fp32 else _t_out_raw.to(image_hidden_raw.dtype)
            else:
                t_emb_item_out = t_emb_item
            if self.diff_head_blocks is not None and (not self._fullseq_diff_blocks):
                _diff_only_pos = diff_pos_ids
                _rot_emb = self.model.rotary_emb
                _cos3d, _sin3d = _rope_tables(_rot_emb, _diff_only_pos, image_hidden_raw, dtype=self._rope_table_dtype)
                _diff_is_image = torch.ones(_diff_only_pos.shape[-1], dtype=torch.bool, device=image_hidden_raw.device)
                _cos_pre_head, _sin_pre_head = self._assemble_mrope(_cos3d, _sin3d, _diff_is_image)
                _diff_cu_i32 = diff_cu_seqlens.to(torch.int32)
                _head_x = image_hidden_raw.unsqueeze(0)
                for _blk in self.diff_head_blocks:
                    if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                        _head_x = torch.utils.checkpoint.checkpoint(
                            _blk,
                            _head_x,
                            t_emb_item_out,
                            diff_item_ids,
                            _cos_pre_head,
                            _sin_pre_head,
                            _diff_cu_i32,
                            _max_diff_len,
                            use_reentrant=False,
                        )
                    else:
                        _head_x = _blk(
                            _head_x,
                            t_emb_item_out,
                            diff_item_ids,
                            _cos_pre_head,
                            _sin_pre_head,
                            _diff_cu_i32,
                            _max_diff_len,
                        )
                image_hidden_raw = _head_x.squeeze(0)
            if self.config.diff_time_embed_mode == "adaln":
                sc_out, sh_out = self.diff_adaln_out(t_emb_item_out).index_select(0, diff_item_ids).chunk(2, dim=-1)
                image_hidden = _adaln_apply(self.diff_norm, image_hidden_raw, sc_out, sh_out, self._adaln_fp32)
            else:
                image_hidden = self.diff_norm(image_hidden_raw)
            diff_pred = self.diff_output_proj(image_hidden)
            if diff_target is not None:
                if self.config.diff_pred_type == "x0" and diff_flow_sigmas is not None:
                    _lens = diff_grid_thw.prod(dim=-1).to(torch.long)
                    _t_min = self.config.diff_t_min
                    _sig = diff_flow_sigmas.repeat_interleave(_lens).clamp_min(_t_min).unsqueeze(-1).float()
                    _xt = vae_noised_latents.float()
                    diff_loss = F.mse_loss((_xt - diff_pred.float()) / _sig, (_xt - diff_target.float()) / _sig)
                else:
                    diff_loss = F.mse_loss(diff_pred.float(), diff_target.float())
        elif self.diff_enabled and self.diff_output_proj is not None:
            _hs = self.config.text_config.hidden_size
            _w = self.diff_output_proj.weight
            _z = 0.0 * _probe_h[:1].to(_w.dtype)
            if self._fullseq_diff_blocks and _fs_t_out is not None:
                _t_z = _fs_t_out[:1]
            elif self.diff_time_embed_out is not None:
                _t_z = self.diff_time_embed_out(torch.zeros(1, dtype=torch.float32, device=_w.device))
                if not self._adaln_fp32:
                    _t_z = _t_z.to(_w.dtype)
            else:
                _t_z = torch.zeros((1, _hs), dtype=_w.dtype, device=_w.device)
            if self._use_adaln and self.diff_adaln_out is not None:
                _sc_o_z, _sh_o_z = self.diff_adaln_out(_t_z).chunk(2, dim=-1)
                _norm_z = _adaln_apply(self.diff_norm, _z, _sc_o_z, _sh_o_z, self._adaln_fp32)
            else:
                _norm_z = self.diff_norm(_z)
            _diff_post_ka = self.diff_output_proj(_norm_z).sum()
            if (self.diff_head_blocks is not None or self.diff_input_process_blocks is not None) and (
                not self._fullseq_diff_blocks
            ):
                _dev = _w.device
                _dt = _w.dtype
                _dum_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.long, device=_dev)
                _dum_cu = torch.tensor([0, 1], dtype=torch.long, device=_dev)
                _dum_pos_ids = self._build_diff_position_ids(_dum_grid_thw, _dum_cu, _dev)
                _rot_emb = self.model.rotary_emb
                _dum_cos3d, _dum_sin3d = _rope_tables(
                    _rot_emb, _dum_pos_ids, torch.empty(1, dtype=_dt, device=_dev), dtype=self._rope_table_dtype
                )
                _dum_is_img = torch.ones(1, dtype=torch.bool, device=_dev)
                _dum_cos_pre, _dum_sin_pre = self._assemble_mrope(_dum_cos3d, _dum_sin3d, _dum_is_img)
                _dum_cu_i32 = _dum_cu.to(torch.int32)
                if self.diff_time_embed_out is not None:
                    _dum_t_emb = _t_z
                else:
                    _dum_t = torch.zeros(1, dtype=torch.float32, device=_dev)
                    _dum_t_emb = self.diff_time_embed(_dum_t)
                _dum_item_ids = torch.zeros(1, dtype=torch.long, device=_dev)
                _dum_x = torch.zeros(1, 1, _hs, dtype=_dt, device=_dev)
                if self.diff_head_blocks is not None:
                    _dum_hd_x = _dum_x
                    for _blk in self.diff_head_blocks:
                        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                            _dum_hd_x = torch.utils.checkpoint.checkpoint(
                                _blk,
                                _dum_hd_x,
                                _dum_t_emb,
                                _dum_item_ids,
                                _dum_cos_pre,
                                _dum_sin_pre,
                                _dum_cu_i32,
                                1,
                                use_reentrant=False,
                            )
                        else:
                            _dum_hd_x = _blk(
                                _dum_hd_x, _dum_t_emb, _dum_item_ids, _dum_cos_pre, _dum_sin_pre, _dum_cu_i32, 1
                            )
                    _diff_post_ka = _diff_post_ka + _dum_hd_x.sum()
        align_loss: Optional[torch.Tensor] = None
        per_token_align_loss: Optional[torch.Tensor] = None
        # MoT can finish the student first, then run a diffusion-only teacher
        # against this forward's detached AR K/V, before building the SF loss.
        if callable(selfflow_teacher_feats):
            selfflow_teacher_feats = selfflow_teacher_feats()
        if (
            run_diff
            and self.config.selfflow_enabled
            and (_selfflow_student_hidden is not None)
            and (selfflow_teacher_feats is not None)
        ):
            _sf_lat_idx = image_latent_idx
            _sf_student = _selfflow_student_hidden[0].index_select(0, _sf_lat_idx)
            _sf_t = selfflow_teacher_feats.to(_sf_student.device).detach()
            if _sf_t.shape[0] != _sf_student.shape[0]:
                raise ValueError(
                    f"selfflow_teacher_feats 长度 ({_sf_t.shape[0]}) != L_diff ({_sf_student.shape[0]}) —— 师生两侧的 diff item 不是同一批"
                )
            if _sf_student.shape[0] > 0:
                _sf_s = self.selfflow_align_head(_sf_student)
                _sf_pt = 1.0 - F.cosine_similarity(_sf_s.float(), _sf_t.float(), dim=-1)
                _sf_m = kwargs.get("selfflow_align_token_mask")
                if _sf_m is None:
                    align_loss = _sf_pt.mean()
                else:
                    if _sf_m.shape[0] != _sf_pt.shape[0]:
                        raise ValueError(
                            f"selfflow_align_token_mask 长度 ({_sf_m.shape[0]}) != L_diff ({_sf_pt.shape[0]}) —— 掩码按全局 diff_cu_seqlens 建，序列被切过（ulysses/CP）就会对不上"
                        )
                    _sf_mw = _sf_m.to(_sf_pt.dtype)
                    align_loss = (_sf_pt * _sf_mw).sum() / _sf_mw.sum().clamp(min=1.0)
                if return_per_token_loss:
                    per_token_align_loss = _sf_pt.detach()
        loss = None
        w = self.config.diff_loss_weight * float(kwargs.get("diff_loss_scale", 1.0))
        w_ar = getattr(self.config, "ar_loss_weight", 1.0)
        d_const = float(getattr(self.config, "loss_norm_tokens", 5642.0))
        n_diff = diff_pred.shape[0] if diff_pred is not None else 0
        if ar_loss is not None:
            n_ar = (labels != -100).sum()
            loss = w_ar * (n_ar / d_const) * ar_loss
        if diff_loss is not None:
            _diff_term = w * (n_diff / d_const) * diff_loss
            loss = _diff_term if loss is None else loss + _diff_term
        if align_loss is not None:
            _sf_w = float(self.config.selfflow_align_weight) * float(kwargs.get("diff_loss_scale", 1.0))
            if _sf_m is None:
                _sf_w = _sf_w * (n_diff / d_const)
                if loss is None:
                    loss = _sf_w * align_loss
                else:
                    loss = loss + _sf_w * align_loss.to(loss.dtype)
            else:
                _sf_wt = _sf_w * (_sf_mw.sum() / d_const)
                if loss is None:
                    loss = (_sf_wt * align_loss).to(align_loss.dtype)
                else:
                    loss = loss + (_sf_wt * align_loss).to(loss.dtype)
        if ar_loss is None and ar_logits is not None and (loss is not None):
            loss = loss + 0.0 * ar_logits.sum().to(loss.dtype)
        if _lmh_ka is not None and loss is not None:
            loss = loss + 0.0 * _lmh_ka.sum().to(loss.dtype)
        diff_keepalive: Optional[torch.Tensor] = None
        if _diff_pre_ka is not None and _diff_post_ka is not None:
            diff_keepalive = _diff_pre_ka + _diff_post_ka
        elif _diff_pre_ka is not None:
            diff_keepalive = _diff_pre_ka
        elif _diff_post_ka is not None:
            diff_keepalive = _diff_post_ka
        if diff_keepalive is not None and loss is not None:
            loss = loss + 0.0 * diff_keepalive.to(loss.dtype)
        if vq_keepalive is not None and loss is not None:
            loss = loss + 0.0 * vq_keepalive
        if (
            self.config.selfflow_enabled
            and align_loss is None
            and (self.selfflow_align_head is not None)
            and (loss is not None)
        ):
            _sf_w0 = self.selfflow_align_head.linear1.weight
            _sf_dum = torch.zeros((1, self.config.text_config.hidden_size), dtype=_sf_w0.dtype, device=_sf_w0.device)
            loss = loss + 0.0 * self.selfflow_align_head(_sf_dum).sum().to(loss.dtype)
        if diff_target is None and diff_pred is not None:
            # Match native inference's all-image output contract. Reference
            # rows are fixed conditions: their velocity is zero and their x0
            # is the clean input. No extra head/FSDP calls are needed for rows
            # that never contribute to denoising or training supervision.
            if _entry_offset:
                fixed = (_vae_latents[:_entry_offset].to(diff_pred.dtype)
                         if self.config.diff_pred_type == "x0" else
                         diff_pred.new_zeros((_entry_offset, diff_pred.shape[-1])))
                diff_pred = torch.cat((fixed, diff_pred))
            if _vae_output_order is not None:
                diff_pred = diff_pred.index_select(0, _vae_output_order)
        return GlmImageMoeTransfusionSelfFlowOutput(
            loss=loss,
            ar_loss=ar_loss,
            diff_loss=diff_loss,
            logits=ar_logits,
            aux_loss=None,
            per_token_loss=per_token_loss,
            diff_pred=diff_pred,
            align_loss=align_loss,
            per_token_align_loss=per_token_align_loss,
            teacher_feats=None,
        )


@dataclass
class GlmImageMoeTransfusionSelfFlowOutput(NativeOutput):
    teacher_feats: Optional[torch.FloatTensor] = None
    align_loss: Optional[torch.FloatTensor] = None
    per_token_align_loss: Optional[torch.FloatTensor] = None


# （已搬到 configuration_hy.py，原 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:2371-2390）
# class GlmImageMoeTransfusionSelfFlowBaseConfig: ...


# （已搬到 configuration_hy.py，原 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:2393-2397）
# class GlmImageMoeTransfusionSelfFlowConfig: ...


class SelfFlowProjHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear1 = nn.Linear(hidden, 2 * hidden)
        self.linear2 = nn.Linear(2 * hidden, hidden)
        self.act = nn.SiLU()
        self.detach_input = False

    def forward(self, x):
        if self.detach_input:
            x = x.detach()
        return self.linear2(self.act(self.linear1(x)))


class GlmImageMoeTransfusionSelfFlowModel(TransfusionSelfFlowForwardMixin, _TransfusionConditionedModel):
    config_class = GlmImageMoeTransfusionSelfFlowBaseConfig

    def __init__(self, config):
        super().__init__(config)
        adapt_backbone(self)
        hidden = config.text_config.hidden_size
        self.selfflow_align_head = SelfFlowProjHead(hidden) if config.selfflow_enabled else None
        depth = config.text_config.num_hidden_layers
        self._selfflow_student_layer_idx = max(
            0, min(depth - 1, round(config.selfflow_student_layer_ratio * depth) - 1)
        )
        self._selfflow_teacher_layer_idx = max(
            0, min(depth - 1, round(config.selfflow_teacher_layer_ratio * depth) - 1)
        )

    def forward(self, *args, sf_teacher_layer=0, sf_student_layer=0, diff_timesteps_per_token=None, **kwargs):
        if isinstance(diff_timesteps_per_token, bool):
            diff_timesteps_per_token = kwargs.get("diff_timesteps") if diff_timesteps_per_token else None
        if sf_teacher_layer:
            kwargs["selfflow_teacher_mode"] = True
            self._selfflow_teacher_layer_idx = sf_teacher_layer - 1
        if sf_student_layer:
            self._selfflow_student_layer_idx = sf_student_layer - 1
        old_zero = (getattr(self, "_hy_zero_vision", False), getattr(self, "_hy_zero_ar", False))
        if getattr(self, "_hy_fastpath", False) and self.training and torch.is_grad_enabled():
            from .base import _eval_only_placeholder as _p_lazy
            globally_zero_branches = _p_lazy('globally_zero_branches', 'veomni.optimizations.hy.zero_loss，guard: self.training and torch.is_grad_enabled() / _hy_zero_vision')

            self._hy_zero_vision, self._hy_zero_ar = globally_zero_branches(self, args, kwargs)
        saved_prefetch = []
        if sf_teacher_layer and (
            getattr(self, "_hy_fastpath", False) or getattr(self, "_hy_bound_teacher_prefetch", False)
        ):
            from .base import _eval_only_placeholder as _p_lazy
            bound_teacher_prefetch = _p_lazy('bound_teacher_prefetch', 'veomni.optimizations.hy.parallel，guard: teacher prefetch，蒸馏训练路径')

            saved_prefetch = bound_teacher_prefetch(self, sf_teacher_layer - 1)
        try:
            output = super().forward(*args, diff_timesteps_per_token=diff_timesteps_per_token, **kwargs)
        finally:
            self._hy_zero_vision, self._hy_zero_ar = old_zero
            if saved_prefetch:
                from .base import _eval_only_placeholder as _p_lazy
                restore_prefetch = _p_lazy('restore_prefetch', 'veomni.optimizations.hy.parallel，guard: teacher prefetch，蒸馏训练路径')

                restore_prefetch(saved_prefetch)
        output["sf_features"] = output.teacher_feats
        return output


class GlmImageMoeTransfusionSelfFlowForMMGeneration(GlmImageMoeTransfusionSelfFlowModel):
    config_class = GlmImageMoeTransfusionSelfFlowConfig


from .compat_hy import install_model_aliases

install_model_aliases(globals())

# Keep checkpoint model_type values while registering the canonical classes here.
from transformers import AutoConfig, AutoModel

for _config, _model in (
    (GlmImageMoeTransfusionSelfFlowBaseConfig, GlmImageMoeTransfusionSelfFlowModel),
    (GlmImageMoeTransfusionSelfFlowConfig, GlmImageMoeTransfusionSelfFlowForMMGeneration),
):
    AutoConfig.register(_config.model_type, _config)
    AutoModel.register(_config, _model)

