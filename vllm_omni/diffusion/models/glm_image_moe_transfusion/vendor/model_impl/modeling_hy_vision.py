"""【自动生成，勿手改】HY 的视觉塔（ViT / VQ）。

由 `model_impl/tools/gen_vision.py` 从 cogview 的三个文件按**依赖闭包**抽取：
从 `GlmImageVQVisionModel` / `GlmImageVisionIBQ` / `VQPrePoolProjector` 出发，
只保留闭包里的定义，裁掉 cogview 自己那套 AR 生成栈
（`GlmImageModel` 456 行、`GlmImageForConditionalGeneration` 393 行等，HY 用不到）。

**函数体一字未改**，改动只有 import。import 区由 `collect_imports()` 自动筛
（手写会漏 —— 试过漏掉 `dynamic_rope_update` 这种装饰器）。
"""

from __future__ import annotations

from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as logging
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.base import get_parallel_state, is_torch_npu_available, is_triton_available  # noqa: F401

# ── 下面的 import 由 collect_imports() 从源文件自动筛出 ──
import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.init import _calculate_fan_in_and_fan_out
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    can_return_tuple,
    is_torchdynamo_compiling,
)
from transformers.utils.generic import check_model_inputs
from .configuration_hy import (
    Glm4vVisionConfig,
    GlmImageConfig,
    GlmImageTextConfig,
    GlmImageVisionConfig,
)


logger = logging.get_logger(__name__)

# ══════════ 依赖闭包抽自 veomni/models/cogview/modeling_glm_image_omni.py （25/36 个定义，裁掉 11 个）══════════

# ── veomni/models/cogview/modeling_glm_image_omni.py:62-81 ──
def maybe_autocast(
    device_type: str,
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True,
    cache_enabled: Optional[bool] = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()


# ── veomni/models/cogview/modeling_glm_image_omni.py:84-107 ──
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ── veomni/models/cogview/modeling_glm_image_omni.py:110-119 ──
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ── veomni/models/cogview/modeling_glm_image_omni.py:184-188 ──
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ── veomni/models/cogview/modeling_glm_image_omni.py:191-245 ──
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # # Keep half or full tensor for later concatenation
    # rotary_dim = cos.shape[-1]
    # q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    # k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # # Apply rotary embeddings on the first half or full tensor
    # q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    # k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # # Concatenate back to full shape
    # q_embed = torch.cat([q_embed, q_pass], dim=-1)
    # k_embed = torch.cat([k_embed, k_pass], dim=-1)

    # return q_embed, k_embed

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ── veomni/models/cogview/modeling_glm_image_omni.py:248-265 ──
def apply_multimodal_rotary_pos_emb_glm(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed, k_embed = apply_multimodal_rotary_pos_emb(
        q=q_rot,
        k=k_rot,
        cos=cos,
        sin=sin,
        mrope_section=mrope_section,
        unsqueeze_dim=unsqueeze_dim,
    )

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


# ── veomni/models/cogview/modeling_glm_image_omni.py:268-279 ──
def apply_rotary_pos_emb_hf(q, k, cos, sin, unsqueeze_dim=1):
    """HF-style rotary pos emb: mrope already baked into cos/sin, just do partial rotation."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ── veomni/models/cogview/modeling_glm_image_omni.py:282-363 ──
class GlmImageTextAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: GlmImageTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_parameters = config.rope_parameters
        self.hf_rope_style = config.rope_parameters.get("hf_rope_style", False)
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(
            1, 2
        )  # e.g, shape = torch.Size([1, 32, 389, 128])
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(
            1, 2
        )  # e.g., shape = torch.Size([1, 2, 389, 128])
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(
            1, 2
        )  # e.g., torch.Size([1, 2, 389, 128])

        cos, sin = position_embeddings
        if self.hf_rope_style:
            query_states, key_states = apply_rotary_pos_emb_hf(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_multimodal_rotary_pos_emb_glm(
                query_states, key_states, cos, sin, self.rope_parameters["mrope_section"]
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ── veomni/models/cogview/modeling_glm_image_omni.py:366-446 ──
class GlmImageTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: GlmImageTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [8, 12, 12])
        self.hf_rope_style = config.rope_parameters.get("hf_rope_style", False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[GlmImageTextConfig] = None,
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

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # e.g., x.shape = torch.Size([1, 389, 4096]) (bs, seq_len, hidden_size)
        # e.g., position_ids.shape = torch.Size([3, 1, 389]) (3, bs, seq_len))

        # In contrast to other models, GLM_IMAGE different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions) # e.g., shape = torch.Size([3, 1, 1, 389])

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            if self.hf_rope_style:
                freqs = self._apply_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _apply_mrope(self, freqs, mrope_section):
        """Apply multimodal RoPE section selection on raw frequencies (HF style)."""
        chunks = freqs.split(mrope_section, dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)


# ── veomni/models/cogview/modeling_glm_image_omni.py:450-467 ──
class GlmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GlmRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ── veomni/models/cogview/modeling_glm_image_omni.py:470-482 ──
class GlmMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# ── veomni/models/cogview/modeling_glm_image_omni.py:485-533 ──
class GlmImageTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmImageTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmImageTextAttention(config, layer_idx)
        self.mlp = GlmMLP(config)
        self.input_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ── veomni/models/cogview/modeling_glm_image_omni.py:536-555 ──
def variance_scaling_(tensor, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = 1.0 / denom

    if distribution == "truncated_normal":
        nn.init.trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        nn.init.normal_(tensor, std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        nn.init.uniform_(tensor, -bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


# ── veomni/models/cogview/modeling_glm_image_omni.py:558-559 ──
def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# ── veomni/models/cogview/modeling_glm_image_omni.py:562-563 ──
def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


# ── veomni/models/cogview/modeling_glm_image_omni.py:566-624 ──
class GlmImagePreTrainedModel(PreTrainedModel):
    config: GlmImageConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmImageTextDecoderLayer"]  # ,"Glm4vVisionBlock"]#, "GlmImageVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmImageTextDecoderLayer,
        "attentions": GlmImageTextAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, GlmImageTextRotaryEmbedding):
            config = module.config
            base = config.rope_parameters["rope_theta"]
            partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            with torch.no_grad():
                module.inv_freq.copy_(inv_freq)
                module.original_inv_freq.copy_(inv_freq)
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # Module-level marker `_zero_init` (set during construction) opts the
            # layer into zero-weight init. Kept for backward-compat; new code
            # should prefer the `__zero_init_module__` mechanism below.
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            else:
                lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif getattr(module, "__zero_init_module__", False):
            # Module-level marker for DiT-Zero-style "gate" modules — every
            # direct parameter is zero-initialized so the residual branch starts
            # as identity. Used by `_ZeroInitGate` in the pixel conv heads.
            # See project_ardiff_arch_bugs memo (2026-05-21).
            for _name, _p in module.named_parameters(recurse=False):
                nn.init.zeros_(_p)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, GlmRMSNorm):
            # GlmRMSNorm 自己直接继承 nn.Module，不是 nn.LayerNorm 的子类，
            # 上面的 isinstance(nn.LayerNorm) 不会命中。如果走 missing-key
            # 路径（base ckpt 没有这些权重，比如 ARDiff 分支），γ 就会留在
            # to_empty() 给的未初始化内存里（CUDA 上通常 ≈ 0），导致整个分支
            # forward 输出 0 → 梯度断流 → 永远学不到东西。
            nn.init.ones_(module.weight)


# ── veomni/models/cogview/modeling_glm_image_omni.py:627-643 ──
@dataclass
class GlmImageModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


# ── veomni/models/cogview/modeling_glm_image_omni.py:916-924 ──
class GlmImageVQVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


# ── veomni/models/cogview/modeling_glm_image_omni.py:927-1001 ──
class GlmImageVQVisionAttention(nn.Module):
    """CLIP-style MHA with cu_seqlens sequence packing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]
            attn_output = torch.cat(
                [
                    attention_interface(
                        self,
                        qi,
                        ki,
                        vi,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for qi, ki, vi in zip(*splits)
                ],
                dim=1,
            )

        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        return self.out_proj(attn_output)


# ── veomni/models/cogview/modeling_glm_image_omni.py:1004-1015 ──
class GlmImageVQVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = GlmImageVQVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GlmImageVQVisionMLP(config)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), **kwargs)
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


# ── veomni/models/cogview/modeling_glm_image_omni.py:1018-1063 ──
class GlmImageVQVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_positions = (config.image_size // config.patch_size) ** 2

        # Linear patch projection (converted from Conv2d by ckpt script)
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size * config.patch_size,
            self.embed_dim,
        )
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, hidden_states, lengths, grid_thw, h_coords, w_coords):
        """
        Args:
            hidden_states: (total_patches, C*ph*pw)
            lengths: per-image patch counts
            grid_thw: (num_images, 3)
            h_coords, w_coords: (total_patches,) patch grid coordinates
        """
        embeddings = self.patch_embedding(hidden_states.to(self.patch_embedding.weight.dtype))

        # Interpolate 2D position embeddings
        device = self.position_embedding.weight.device
        orig_size = int(self.num_positions**0.5)
        pos_2d = (
            self.position_embedding.weight.view(orig_size, orig_size, self.embed_dim)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        target_h = torch.repeat_interleave(grid_thw[:, 1].to(device), lengths).float()
        target_w = torch.repeat_interleave(grid_thw[:, 2].to(device), lengths).float()
        norm_w = ((w_coords.to(device).float() + 0.5) / target_w) * 2 - 1
        norm_h = ((h_coords.to(device).float() + 0.5) / target_h) * 2 - 1
        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

        pos_embed = F.grid_sample(pos_2d, grid, mode="bicubic", align_corners=False, padding_mode="border")
        pos_embed = pos_embed.squeeze(0).squeeze(-1).permute(1, 0)
        return embeddings + pos_embed.to(embeddings.dtype).to(embeddings.device)


# ── veomni/models/cogview/modeling_glm_image_omni.py:1066-1124 ──
class GlmImageVQVisionModel(GlmImagePreTrainedModel):
    """
    CLIP-style ViT for the VQ path.
    Interface-aligned with Glm4vVisionModel: forward(hidden_states, grid_thw) → (total_tokens, hidden_size).
    """

    config: GlmImageVisionConfig

    def __init__(self, config: GlmImageVisionConfig):
        super().__init__(config)
        self.spatial_merge_size = 1  # CLIP ViT has no spatial merge
        self.patch_size = config.patch_size

        self.embeddings = GlmImageVQVisionEmbeddings(config)
        self.blocks = nn.ModuleList([GlmImageVQVisionBlock(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.post_init()

    @property
    def dtype(self):
        return self.embeddings.patch_embedding.weight.dtype

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_patches, C*ph*pw) patchified pixel values.
            grid_thw: (num_images, 3) with [t, h_patches, w_patches] per image.

        Returns:
            (total_patches, hidden_size)
        """
        # Fully device-side; no D2H sync, no per-image python loop.
        device = grid_thw.device
        B = grid_thw.shape[0]
        t_per = grid_thw[:, 0]
        h_per = grid_thw[:, 1]
        w_per = grid_thw[:, 2]
        hw_per = h_per * w_per
        total_per = t_per * hw_per

        cu_seqlens = F.pad(hw_per.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
        seqlens = hw_per  # tensor; embeddings.forward 接受 tensor
        max_seqlen = seqlens.max()  # 0-d tensor on device, varlen attn 接受

        total_patches = hidden_states.shape[0]  # python int, no sync
        image_id = torch.repeat_interleave(torch.arange(B, device=device), total_per)
        total_cu = F.pad(total_per.cumsum(dim=0), (1, 0), value=0)
        local_idx = torch.arange(total_patches, device=device) - total_cu[image_id]
        spatial_idx = local_idx % hw_per[image_id]  # 丢掉 frame 维 (t)
        h_coords = spatial_idx // w_per[image_id]
        w_coords = spatial_idx % w_per[image_id]

        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, h_coords, w_coords)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return hidden_states


# ── veomni/models/cogview/modeling_glm_image_omni.py:1127-1141 ──
class Glm4vVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# ── veomni/models/cogview/modeling_glm_image_omni.py:1247-1361 ──
class GlmImageTextModel(GlmImagePreTrainedModel):
    config: GlmImageTextConfig
    input_modalities = ("text",)

    def __init__(self, config: GlmImageTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmImageTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmImageTextRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[-1]
            position_ids = position_ids[:-1]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds  # e.g., shape = torch.Size([1, 389, 4096]) (bs, seq_len, hidden_size)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # import torch.distributed as _dist
        # _debug_fwd = _dist.is_initialized() and _dist.get_rank() == 0

        for _layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,  # e.g., shape = torch.Size([1, 389, 4096])
                attention_mask=causal_mask,  # e.g., None
                position_ids=position_ids,  # e.g, shape = torch.Size([3, 1, 389]) which is not used in decoder_layer
                past_key_values=past_key_values,  # e.g., DynamicCache(layers=[DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer])
                cache_position=cache_position,  # e.g., shape = torch.Size([389]) 0~388
                position_embeddings=position_embeddings,  # (cos, sin), e.g., shape = (torch.Size([3, 1, 389, 64]))*2
                **kwargs,
            )
            hidden_states = layer_outputs
            # if _debug_fwd:
            #     _h = hidden_states.float()
            #     _has_nan = torch.isnan(_h).any().item()
            #     _has_inf = torch.isinf(_h).any().item()
            #     _tag = " *** NAN ***" if _has_nan else (" *** INF ***" if _has_inf else "")
            #     print(
            #         f"[FWD] layer {_layer_idx:2d}: norm={_h.norm():.4e}, "
            #         f"absmax={_h.abs().max():.4e}, mean={_h.mean():.4e}{_tag}"
            #     )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ── veomni/models/cogview/modeling_glm_image_omni.py:1364-1819 ──
class GlmImageModel(GlmImagePreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: GlmImageConfig
    _no_split_modules = ["GlmImageTextDecoderLayer"]  # , "Glm4vVisionBlock"]#"GlmImageVisionBlock"]

    def __init__(self, config):
        super().__init__(config)

        # NOTE: understanding VIT (self.visual) disabled — single VIT design now uses vq_visual for both
        # understanding and VQ encoding. Old init_visual branch kept commented for reference.
        # if getattr(config, 'init_visual', True):
        #     self.visual = Glm4vVisionModel._from_config(config.vision_config)
        # else:
        #     self.visual = None

        self.language_model = GlmImageTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Single VIT path: always build vq_visual when init_vq=True; understanding shares it.
        vq_vc = getattr(config, "vq_vision_config", None)
        if getattr(config, "init_vq", False) and vq_vc is not None and getattr(vq_vc, "vq_codebook_size", 0) > 0:
            from .vq_projector import GlmImageVisionVQProjector

            # NOTE: share_vit_for_vq logic removed — vq_visual is always built.
            # if getattr(config, 'share_vit_for_vq', False):
            #     self.vq_visual = None
            # else:
            #     self.vq_visual = GlmImageVQVisionModel._from_config(vq_vc)
            self.vq_visual = GlmImageVQVisionModel._from_config(vq_vc)

            self.vq_projector = GlmImageVisionVQProjector(
                in_channels=vq_vc.hidden_size,
                codebook_size=vq_vc.vq_codebook_size,
                codebook_dim=getattr(vq_vc, "vq_codebook_dim", vq_vc.hidden_size),
                num_conv_layers=getattr(vq_vc, "vq_num_conv_layers", 2),
                pre_pool_factor=getattr(vq_vc, "vq_pre_pool_factor", 1),
                pre_pool_type=getattr(vq_vc, "vq_pre_pool_type", "none"),
            )
        else:
            self.vq_visual = None
            self.vq_projector = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            video_group_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]

                # # --- Original per-token Python loop (slow for long sequences) ---
                # input_tokens = input_ids.tolist()
                # input_token_type = []
                # video_check_flg = False
                # for token in input_tokens:
                #     if token == video_start_token_id:
                #         video_check_flg = True
                #     elif token == video_end_token_id:
                #         video_check_flg = False
                #     if token == image_token_id and not video_check_flg:
                #         input_token_type.append("image")
                #     elif token == image_token_id and video_check_flg:
                #         input_token_type.append("video")
                #     else:
                #         input_token_type.append("text")
                # input_type_group = []
                # for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                #     group = list(group)
                #     start_index = group[0][0]
                #     end_index = group[-1][0] + 1
                #     input_type_group.append((key, start_index, end_index))

                # --- Vectorized version (avoids per-token Python loop + GPU→CPU tolist) ---
                is_image_tok = input_ids == image_token_id
                is_vid_start = input_ids == video_start_token_id
                is_vid_end = input_ids == video_end_token_id

                # Boolean "in video region" mask via forward-fill of last VS/VE event
                event_mask = is_vid_start | is_vid_end
                in_video = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                if event_mask.any():
                    event_indices = torch.where(event_mask)[0]
                    event_vals = is_vid_start[event_mask]  # True=enter video, False=leave video
                    positions = torch.arange(input_ids.shape[0], device=input_ids.device)
                    idx = torch.searchsorted(event_indices, positions, right=True) - 1
                    valid = idx >= 0
                    in_video[valid] = event_vals[idx[valid]]

                # type_code: 0=text, 1=image, 2=video
                type_codes = torch.zeros(input_ids.shape[0], dtype=torch.int8, device=input_ids.device)
                type_codes[is_image_tok & ~in_video] = 1
                type_codes[is_image_tok & in_video] = 2

                # Find group boundaries via torch.diff on GPU, only transfer boundary indices to CPU
                type_map = {0: "text", 1: "image", 2: "video"}
                n_codes = type_codes.shape[0]
                if n_codes == 0:
                    input_type_group = []
                else:
                    changes = torch.where(torch.diff(type_codes) != 0)[0] + 1
                    starts = torch.cat(
                        [
                            type_codes.new_zeros(1, dtype=torch.long),
                            changes,
                            type_codes.new_tensor([n_codes], dtype=torch.long),
                        ]
                    )
                    group_keys = type_codes[starts[:-1]]
                    starts_cpu = starts.cpu().tolist()
                    keys_cpu = group_keys.cpu().tolist()
                    input_type_group = [
                        (type_map[k], starts_cpu[j], starts_cpu[j + 1]) for j, k in enumerate(keys_cpu)
                    ]

                llm_pos_ids_list = []
                video_frame_num = 1
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_frame_num,
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )

                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()

                            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        video_group_index += 1

                        if video_group_index >= video_grid_thw[video_index][0]:
                            video_index += 1
                            video_group_index = 0

                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        video_frame_num = 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(self):
        """
        Not Using now
        """
        return None

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # NOTE: understanding now reuses the VQ VIT (self.vq_visual). Old self.visual path commented.
        # assert self.visual is not None, "VIT encoder not initialized (init_visual=False)"
        # pixel_values = pixel_values.type(self.visual.dtype)
        # image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        assert self.vq_visual is not None, "VIT encoder not initialized (init_vq=False)"
        pixel_values = pixel_values.type(self.vq_visual.dtype)
        image_embeds = self.vq_visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor = None,
        video_features: torch.FloatTensor = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            # GLM-4.1V and GLM-4.5V special_video_mask is special_image_mask
            special_video_mask = special_image_mask = input_ids == self.config.image_token_id
            # special_video_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, GlmImageModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            # image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        # else:
        #     # Lightweight dummy op so gradients flow to VIT under FSDP/DDP.
        #     # Use a single cat + sum instead of a python loop over every parameter.
        #     trainable_params = [p.view(-1) for p in self.visual.parameters() if p.requires_grad]
        #     if trainable_params:
        #         dummy_sum = torch.cat(trainable_params).sum()
        #         inputs_embeds = inputs_embeds + (0.0 * dummy_sum)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    # video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return GlmImageModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


# ══════════ 依赖闭包抽自 veomni/models/cogview/vq_projector.py （2/3 个定义，裁掉 1 个）══════════

# ── veomni/models/cogview/vq_projector.py:16-37 ──
class GlmImageVisionResidualBlock(nn.Module):
    """
    Implementation of the GLM-Image residual block.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding="same")
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding="same")
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x + res


# ── veomni/models/cogview/vq_projector.py:40-156 ──
class GlmImageVisionIBQ(nn.Module):
    """
    Index-Based Quantization Module of GLM-Image.

    Forward returns (z_q, indices). In training mode (``return_commit_loss=True``)
    the returned ``z_q`` carries STE gradient back to ``z`` (encoder output) and
    a third ``commit_loss`` value is appended so the codebook itself trains via
    the standard VQ-VAE commitment formulation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        l2_norm: bool = True,
        beta: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(
        self,
        z: torch.Tensor,
        *,
        return_commit_loss: bool = False,
    ):
        """
        Forward pass.

        Args:
            z: Input features, shape: [B, C, H, W]
            return_commit_loss: when True, applies STE so ``z_q`` is
                differentiable wrt ``z`` and returns an additional scalar
                ``commit_loss``. When False (default), runs no-grad inference.

        Returns:
            return_commit_loss=False:  (z_q, indices)
            return_commit_loss=True:   (z_q, indices, commit_loss)
        """
        batch_size, channels, height, width = z.shape

        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_permuted.view(-1, self.embedding_dim)

        if self.l2_norm:
            z_flat_n = F.normalize(z_flattened, p=2, dim=-1)
            emb_n = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            z_flat_n = z_flattened
            emb_n = self.embedding.weight

        if return_commit_loss:
            # Distances are computed in the (potentially L2-normed) input space;
            # we don't backprop through argmin (it's discrete), only through z
            # via the STE residual and through the codebook table via the
            # commit loss.
            d = (
                torch.sum(z_flat_n**2, dim=1, keepdim=True)
                + torch.sum(emb_n**2, dim=1)
                - 2 * torch.matmul(z_flat_n, emb_n.t())
            )
            indices = torch.argmin(d.detach(), dim=1)
            z_q_hard = emb_n[indices]  # (B*H*W, C); differentiable wrt embedding.weight

            # STE residual: forward value = z_q_hard, backward dz_q/dz = identity.
            z_q_ste = z_flat_n + (z_q_hard - z_flat_n).detach()

            # Standard VQ-VAE commit loss:
            #   * encoder term pulls z toward the picked codebook entry
            #   * codebook term (weighted by beta) pulls the codebook entry toward z
            commit_loss = ((z_q_hard.detach() - z_flat_n) ** 2).mean() + self.beta * (
                (z_q_hard - z_flat_n.detach()) ** 2
            ).mean()

            z_q = z_q_ste.view(batch_size, height, width, self.embedding_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            indices = indices.view(batch_size, height, width)
            return z_q, indices, commit_loss

        with torch.no_grad():
            d = (
                torch.sum(z_flat_n**2, dim=1, keepdim=True)
                + torch.sum(emb_n**2, dim=1)
                - 2 * torch.matmul(z_flat_n, emb_n.t())
            )
            indices = torch.argmin(d, dim=1)
            z_q = emb_n[indices].view(batch_size, height, width, self.embedding_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            indices = indices.view(batch_size, height, width)
        return z_q, indices

    def get_codebook_entry(self, indices: torch.Tensor, bhwc: list) -> torch.Tensor:
        """
        Get codebook entries by indices

        Args:
            indices: Codebook indices
            bhwc: Target shape [batch, height, width, channel]

        Returns:
            z_q: Quantized features, shape: [B, C, H, W]
        """
        z_q = self.embedding(indices)

        if bhwc is not None:
            batch_size, height, width, channels = bhwc
            # [B, H, W, C] -> [B, C, H, W]
            z_q = z_q.view(batch_size, height, width, channels)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


# ══════════ 依赖闭包抽自 veomni/models/cogview/vq_pre_pool_projector.py （1/1 个定义，裁掉 0 个）══════════

# ── veomni/models/cogview/vq_pre_pool_projector.py:22-40 ──
class VQPrePoolProjector(nn.Module):
    def __init__(self, channels: int, pool_factor: int = 2):
        super().__init__()
        self.pool_factor = pool_factor
        self.proj = nn.Conv2d(channels, channels, kernel_size=pool_factor, stride=pool_factor, bias=True)
        self._init_avg_equivalent()

    def _init_avg_equivalent(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            c_out, c_in, _, _ = self.proj.weight.shape
            assert c_out == c_in, "VQPrePoolProjector expects same in/out channels"
            inv = 1.0 / (self.pool_factor * self.pool_factor)
            for i in range(c_out):
                self.proj.weight[i, i, :, :] = inv
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)



__all__ = ["GlmImageVQVisionModel", "GlmImageVisionIBQ",
           "GlmImageVisionResidualBlock", "VQPrePoolProjector"]


# ════════════════════════════════════════════════════════════════════
#  `modeling_hy.py` 只从 cogview 另外两个文件各要一个类（闭包抽出来的），
#  免得为两个类搬 3043 行。来源见 vision_manifest.json 的 EXTRA_ROOTS。
# ════════════════════════════════════════════════════════════════════

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from .configuration_hy import (
    Glm4vVisionConfig,
    GlmImageVisionConfig,
)
from .modeling_hy_vision import (
    GlmImageModel,
    GlmImageTextRotaryEmbedding,
    GlmImageVQVisionModel,
)
from . import modeling_hy_text as moe_init
from .configuration_hy import Glm4MoeLiteConfig
from .modeling_hy_text import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteExperts,
    Glm4MoeLiteMoE,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteTopkRouter,
)

# ══════════ 依赖闭包抽自 veomni/models/cogview/modeling_glm_image_moe_omni.py （3/8 个定义）══════════

# ── veomni/models/cogview/modeling_glm_image_moe_omni.py:75-133 ──
class GlmImageMoeConfig(PretrainedConfig):
    """Top-level config for GLM Image MoE-Omni.

    - ``text_config``: ``Glm4MoeLiteConfig`` (MLA + MoE backbone).
    - ``vision_config``: ``Glm4vVisionConfig`` (understanding VIT).
    - ``vq_vision_config``: ``GlmImageVisionConfig`` (CLIP-style VQ VIT, optional).
    """

    model_type = "glm_image_moe"
    sub_configs = {"vision_config": Glm4vVisionConfig, "text_config": Glm4MoeLiteConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: None | Glm4MoeLiteConfig = None,
        vision_config: None | Glm4vVisionConfig = None,
        vq_vision_config=None,
        image_token_id=154856,
        image_start_token_id=154830,
        image_end_token_id=154831,
        video_token_id=None,
        video_start_token_id=None,
        video_end_token_id=None,
        # NOTE: init_visual / share_vit_for_vq kept for ckpt-compat; ignored by single-VIT model.
        init_visual=True,
        init_vq=False,
        share_vit_for_vq=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(vq_vision_config, dict):
            self.vq_vision_config = GlmImageVisionConfig(**vq_vision_config)
        else:
            self.vq_vision_config = vq_vision_config  # None disables the VQ path

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_token_id = video_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.init_visual = init_visual
        self.init_vq = init_vq
        self.share_vit_for_vq = share_vit_for_vq

        super().__init__(**kwargs)


# ── veomni/models/cogview/modeling_glm_image_moe_omni.py:141-146 ──
def _resolve_initializer_std(config) -> float:
    """Pick `initializer_range` from whichever sub-config has one."""
    for c in (getattr(config, "text_config", None), config, getattr(config, "vision_config", None)):
        if c is not None and hasattr(c, "initializer_range"):
            return c.initializer_range
    return 0.02


# ── veomni/models/cogview/modeling_glm_image_moe_omni.py:149-205 ──
class GlmImageMoePreTrainedModel(PreTrainedModel):
    config: GlmImageMoeConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm4MoeLiteDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": Glm4MoeLiteDecoderLayer,
        "attentions": Glm4MoeLiteAttention,
    }
    _keep_in_fp32_modules_strict = []

    @torch.no_grad()
    def _init_weights(self, module):
        std = _resolve_initializer_std(self.config)

        if isinstance(module, Glm4MoeLiteTopkRouter):
            moe_init.normal_(module.weight, mean=0.0, std=std)
            moe_init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Glm4MoeLiteExperts):
            moe_init.normal_(module.gate_proj, mean=0.0, std=std)
            moe_init.normal_(module.up_proj, mean=0.0, std=std)
            moe_init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, GlmImageTextRotaryEmbedding):
            cfg = module.config
            base = cfg.rope_parameters["rope_theta"]
            partial = cfg.rope_parameters.get("partial_rotary_factor", 1.0)
            head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            dim = int(head_dim * partial)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq.copy_(inv_freq)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # Row-select (aten.select.int) has no DTensor sharding strategy.
                # Under FSDP2 the flow is meta-init → fully_shard → init_weights,
                # so the weight is a dim0-sharded DTensor here — skip the padding
                # zero-fill in that case (cosmetic only; ckpt loading overwrites
                # it and the padding row never receives gradients anyway).
                from torch.distributed.tensor import DTensor as _DTensor

                if not isinstance(module.weight.data, _DTensor):
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# ══════════ 依赖闭包抽自 veomni/models/cogview/modeling_glm_image_omni_ardiff.py （1/12 个定义）══════════

# ── veomni/models/cogview/modeling_glm_image_omni_ardiff.py:576-588 ──
class _DiffNoAffineRMSNorm(nn.Module):
    """RMSNorm without learnable affine — adaLN 模式下 scale/shift 由 adaLN 提供,
    norm 本身不带 weight(标准 DiT adaLN 做法)。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt)


