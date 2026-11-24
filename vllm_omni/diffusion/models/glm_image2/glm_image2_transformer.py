# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native GLM-Image2 Transfusion decoder for vLLM-Omni.

Ports the diffusion path of transformers ``GlmImage2ForConditionalGeneration``
(``zai-org/GLM-Image-2``): a 47-layer MLA decoder (first layer dense, the rest
64-expert top-4 sigmoid-routed MoE with one shared expert), a SigLIP-style
reference-image ViT (with the legacy 2x2 pooling + ``vq_to_llm_proj`` or the
GLM-V ``cond_vit_aligner``), and the AdaLN diffusion input/output heads. The
final norm and language-model head only serve the thinking mode, which decodes
text token by token against a per-request ``GlmImage2TextCache``.

Attention follows the reference block mask: causal, except that every
contiguous image run (reference VAE, reference ViT, target VAE) is
bidirectional. The target VAE block is always the last run, so everything
before it (text and references, whose diffusion timestep is fixed at zero) is
independent of the denoising step. ``build_prefix_cache`` runs that prefix
once per request and stores each layer's prefix keys/values (decompressed for
short prefixes, as the MLA latent for long ones); every denoising step then
only processes the target tokens, which attend to the whole prefix plus
themselves without any mask. Without CFG parallelism both CFG branches share
one step pass: token-wise layers run over the stacked target tokens and each
branch attends to its own prefix.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import vllm.forward_context as vllm_forward_context
from torch import nn
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.fused_moe import FusedMoE
from vllm_omni.diffusion.models.glm_image2.glm_image2_kernels import fused_rope_kv_supported, query_rope_, stage_kv
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

TEXT_TOKEN, VIT_TOKEN, VAE_TOKEN = 0, 1, 2


@dataclass
class GlmImage2TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    mrope_section: list[int]
    mlp_layer_types: list[str]


@dataclass
class GlmImage2VisionConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    image_size: int
    patch_size: int
    num_channels: int
    layer_norm_eps: float
    pos_embed_interp_mode: str


@dataclass
class GlmImage2ModelConfig:
    text: GlmImage2TextConfig
    vision: GlmImage2VisionConfig
    diff_vae_latent_dim: int
    diff_patch_size: int
    diff_timestep_embed_dim: int
    diff_rope_thw_layout: str
    cond_vit_aligner: str
    cond_merge_size: int
    cond_vit_aligner_context_dim: int
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def latent_dim(self) -> int:
        return self.diff_vae_latent_dim * self.diff_patch_size**2

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GlmImage2ModelConfig:
        text, vision = config["text_config"], config["vq_vision_config"]
        rope = text["rope_parameters"]
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError("GLM-Image2 only supports the default mRoPE type.")
        if config.get("diff_time_embed_mode", "adaln") != "adaln" or not config.get("adaln_fp32", True):
            raise NotImplementedError("GLM-Image2 requires fp32 AdaLN time modulation.")
        if config.get("diff_pred_type", "v") != "v" or not config.get("init_vq", True):
            raise NotImplementedError("GLM-Image2 requires velocity prediction and the reference-image VQ encoder.")
        if not text.get("rope_interleave", True) or config.get("cast_rope_emb_to_input_dtype", False):
            raise NotImplementedError("GLM-Image2 requires interleaved fp32 rotary embeddings.")
        if text.get("q_lora_rank") is None or text.get("attention_bias", False):
            raise NotImplementedError("GLM-Image2 requires low-rank, bias-free MLA projections.")
        if text.get("hidden_act", "silu") != "silu" or vision.get("hidden_act") != "gelu_pytorch_tanh":
            raise NotImplementedError("Unexpected GLM-Image2 activation functions.")
        if config["diff_rope_thw_layout"] not in ("block", "interleave", "qwen3"):
            raise NotImplementedError(f"Unsupported diff_rope_thw_layout {config['diff_rope_thw_layout']!r}.")
        aligner, merge_size = config.get("cond_vit_aligner", "legacy"), config.get("cond_merge_size", 2)
        if aligner not in ("glmv", "legacy") or merge_size < 1 or (aligner == "legacy" and merge_size != 2):
            raise NotImplementedError(f"Unsupported cond_vit_aligner {aligner!r} with cond_merge_size={merge_size}.")
        interpolation = vision.get("pos_embed_interp_mode", "bicubic")
        if interpolation not in ("bilinear", "bicubic"):
            raise NotImplementedError(f"Unsupported pos_embed_interp_mode {interpolation!r}.")
        return cls(
            text=GlmImage2TextConfig(
                hidden_size=text["hidden_size"],
                num_hidden_layers=text["num_hidden_layers"],
                num_attention_heads=text["num_attention_heads"],
                q_lora_rank=text["q_lora_rank"],
                kv_lora_rank=text["kv_lora_rank"],
                qk_nope_head_dim=text["qk_nope_head_dim"],
                qk_rope_head_dim=text["qk_rope_head_dim"],
                v_head_dim=text["v_head_dim"],
                intermediate_size=text["intermediate_size"],
                moe_intermediate_size=text["moe_intermediate_size"],
                n_routed_experts=text["n_routed_experts"],
                n_shared_experts=text["n_shared_experts"],
                num_experts_per_tok=text["num_experts_per_tok"],
                n_group=text["n_group"],
                topk_group=text["topk_group"],
                norm_topk_prob=text["norm_topk_prob"],
                routed_scaling_factor=text["routed_scaling_factor"],
                rms_norm_eps=text["rms_norm_eps"],
                vocab_size=text["vocab_size"],
                rope_theta=rope["rope_theta"],
                mrope_section=list(rope["mrope_section"]),
                mlp_layer_types=list(text["mlp_layer_types"]),
            ),
            vision=GlmImage2VisionConfig(
                hidden_size=vision["hidden_size"],
                num_hidden_layers=vision["num_hidden_layers"],
                num_attention_heads=vision["num_attention_heads"],
                intermediate_size=vision["intermediate_size"],
                image_size=vision["image_size"],
                patch_size=vision["patch_size"],
                num_channels=vision["num_channels"],
                layer_norm_eps=vision["layer_norm_eps"],
                pos_embed_interp_mode=interpolation,
            ),
            diff_vae_latent_dim=config["diff_vae_latent_dim"],
            diff_patch_size=config["diff_patch_size"],
            diff_timestep_embed_dim=config["diff_timestep_embed_dim"],
            diff_rope_thw_layout=config["diff_rope_thw_layout"],
            cond_vit_aligner=aligner,
            cond_merge_size=merge_size,
            cond_vit_aligner_context_dim=config.get("cond_vit_aligner_context_dim", 0),
            raw=config,
        )


# ============================================================ rotary embedding


class GlmImage2RotaryEmbedding(nn.Module):
    """3D (t, h, w) mRoPE: each of the ``rope_dim / 2`` frequencies reads one position axis."""

    def __init__(self, config: GlmImage2ModelConfig):
        super().__init__()
        rope_dim = config.text.qk_rope_head_dim
        sections = config.text.mrope_section
        if config.diff_rope_thw_layout == "block":
            axis_indices = [axis for axis, size in enumerate(sections) for _ in range(size)]
        else:
            temporal, height, width = sections
            if height != width or height < temporal:
                raise ValueError("Interleaved mRoPE requires height == width >= temporal section sizes.")
            spatial = [1, 2] * (height - temporal)
            if config.diff_rope_thw_layout == "qwen3":
                axis_indices = [0, 1, 2] * temporal + spatial
            else:
                axis_indices = spatial + [1, 2, 0] * temporal
        # Computed on CPU to match the reference; the CUDA pow kernel differs by 1 ulp in some entries.
        inv_freq = 1.0 / (config.text.rope_theta ** (torch.arange(0, rope_dim, 2, device="cpu").float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("axis_indices", torch.tensor(axis_indices, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: [3, L] -> fp32 (cos, sin) of shape [L, rope_dim / 2]."""
        positions = position_ids[self.axis_indices.to(position_ids.device)].transpose(0, 1).float()
        frequencies = positions * self.inv_freq.to(positions.device)
        return frequencies.cos(), frequencies.sin()


def apply_rotary_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd channel pairs. x: [L, H, D]; cos/sin: [L, D / 2] (fp32)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).to(x.dtype)


# ================================================================ attention


# Decompressed prefix K/V take ~18x the memory of the MLA latent (H * (D + Dv) vs
# kv_lora_rank + rope per token). Short (text) prefixes keep them to skip the per-step
# kv_b_proj; long ones (reference images) keep the latent and decompress every step.
_MAX_DECOMPRESSED_PREFIX_TOKENS = 1024


@dataclass
class GlmImage2LayerCache:
    """Prefix keys/values of one decoder layer, reused by every denoising step.

    Holds either the decompressed ``key``/``value`` or the MLA ``kv_latent`` plus
    the shared rotary key ``k_rot``.
    """

    prefix_len: int
    key: torch.Tensor | None = None  # [P, H, qk_head_dim], RoPE applied
    value: torch.Tensor | None = None  # [P, H, v_head_dim]
    kv_latent: torch.Tensor | None = None  # [P, kv_lora_rank], after kv_a_layernorm
    k_rot: torch.Tensor | None = None  # [P, 1, qk_rope_head_dim], RoPE applied


@dataclass
class GlmImage2KVBuffers:
    """Per-request ``[P + T, H, D]`` key/value staging shared by all layers.

    Each layer copies its cached prefix rows and writes the target rows in
    place, so a denoising step neither re-decompresses the prefix latent nor
    concatenates prefix and target tensors.
    """

    key: torch.Tensor
    value: torch.Tensor


@dataclass
class GlmImage2TextCache:
    """Keys/values of every decoder layer for incremental (thinking-mode) text decoding."""

    key: torch.Tensor  # [layers, max_tokens, H, qk_head_dim], RoPE applied
    value: torch.Tensor  # [layers, max_tokens, H, v_head_dim]
    length: int = 0


def _dense_block_mask(
    spans: list[tuple[int, int]], query_start: int, num_queries: int, num_keys: int, device: torch.device
) -> torch.Tensor:
    """Boolean [1, 1, Nq, Nk] mask: causal, plus bidirectional attention inside each span."""
    query_index = torch.arange(query_start, query_start + num_queries, device=device)
    key_index = torch.arange(num_keys, device=device)
    allowed = key_index[None, :] <= query_index[:, None]
    for start, end in spans:
        in_query = (query_index >= start) & (query_index < end)
        in_key = (key_index >= start) & (key_index < end)
        allowed |= in_query[:, None] & in_key[None, :]
    return allowed[None, None]


class GlmImage2Attention(nn.Module):
    """DeepSeek-V2 style multi-head latent attention with interleaved mRoPE."""

    def __init__(self, config: GlmImage2TextConfig, quant_config=None, prefix: str = ""):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError(
                f"GLM-Image2 has {config.num_attention_heads} attention heads, "
                f"which is not divisible by tensor_parallel_size={tp_size}."
            )
        self.num_heads = config.num_attention_heads // tp_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        if self.qk_head_dim != self.v_head_dim:
            raise NotImplementedError("GLM-Image2 attention expects equal query/key and value head sizes.")

        self.q_a_proj = ReplicatedLinear(
            config.hidden_size, config.q_lora_rank, bias=False, quant_config=quant_config, prefix=f"{prefix}.q_a_proj"
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=1e-6)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = ColumnParallelLinear(
            config.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.qk_head_dim,
            causal=False,
            softmax_scale=self.qk_head_dim**-0.5,
            num_kv_heads=self.num_heads,
            prefix=f"{prefix}.attn",
            skip_sequence_parallel=True,
        )

    def _block_mask_metadata(
        self, spans: list[tuple[int, int]], query_start: int, num_queries: int, num_keys: int, device: torch.device
    ) -> AttentionMetadata | None:
        query_end = query_start + num_queries
        if query_end == num_keys and any(start <= query_start and end >= query_end for start, end in spans):
            # Every query sits inside one trailing bidirectional block: plain full attention.
            return None
        backend = getattr(self.attn, "attn_backend", None)
        if backend is not None and getattr(backend, "supports_piecewise_spans", False):
            return AttentionMetadata(full_attn_spans=[list(spans)])
        return AttentionMetadata(attn_mask=_dense_block_mask(spans, query_start, num_queries, num_keys, device))

    def _project(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """RoPE'd queries, the normed KV latent, its ``[L, H, nope + v]`` decompression and the raw rotary key."""
        num_tokens = hidden_states.shape[0]
        nope = self.qk_nope_head_dim
        query, _ = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)[0]))
        query = query.view(num_tokens, self.num_heads, self.qk_head_dim)
        fused = fused_rope_kv_supported(query)
        if fused:
            query_rope_(query, nope, cos, sin)
        else:
            query[..., nope:] = apply_rotary_interleave(query[..., nope:], cos, sin)

        compressed_kv, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rot = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # RMSNorm's FP32 conversion materializes the strided latent. Avoid an
        # extra BF16 copy before the reduction on NPU.
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv, _ = self.kv_b_proj(kv_latent)
        return query, kv_latent, kv.view(num_tokens, self.num_heads, nope + self.v_head_dim), k_rot, fused

    def decode_text(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: GlmImage2TextCache, layer: int
    ) -> torch.Tensor:
        """Append causal text tokens to ``cache`` and attend to every cached token."""
        num_tokens = hidden_states.shape[0]
        nope = self.qk_nope_head_dim
        query, _, kv, k_rot, _ = self._project(hidden_states, cos, sin)
        start, end = cache.length, cache.length + num_tokens
        key, value = cache.key[layer, :end], cache.value[layer, :end]
        key[start:, :, :nope] = kv[..., :nope]
        key[start:, :, nope:] = apply_rotary_interleave(k_rot.unsqueeze(1), cos, sin)
        value[start:] = kv[..., nope:]
        metadata = None
        if num_tokens > 1:
            metadata = self._block_mask_metadata([], start, num_tokens, end, hidden_states.device)
        output = self.attn(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), metadata)
        output, _ = self.o_proj(output.reshape(num_tokens, self.num_heads * self.v_head_dim))
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        spans: list[tuple[int, int]],
        caches: list[GlmImage2LayerCache] | None,
        kv_buffers: list[GlmImage2KVBuffers] | None,
        store_cache: bool,
    ) -> tuple[torch.Tensor, GlmImage2LayerCache | None]:
        """Prefix pass (``caches is None``) or a denoising step over one or more CFG branches.

        In a step, ``hidden_states`` stacks the target tokens of ``len(caches)``
        branches; token-wise projections run once over all of them and each
        branch attends to its own prefix plus its own target block (no mask).
        """
        num_tokens = hidden_states.shape[0]
        nope = self.qk_nope_head_dim
        query, kv_latent, kv, k_rot, fused = self._project(hidden_states, cos, sin)
        k_nope, value = kv.split([nope, self.v_head_dim], dim=-1)

        if caches is None:
            k_rot = apply_rotary_interleave(k_rot.unsqueeze(1), cos, sin)
            key = torch.cat((k_nope, k_rot.expand(-1, self.num_heads, -1)), dim=-1)
            value = value.contiguous()
            metadata = self._block_mask_metadata(spans, 0, num_tokens, num_tokens, hidden_states.device)
            output = self.attn(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), metadata)
            new_cache = None
            if store_cache and num_tokens <= _MAX_DECOMPRESSED_PREFIX_TOKENS:
                new_cache = GlmImage2LayerCache(num_tokens, key=key, value=value)
            elif store_cache:
                new_cache = GlmImage2LayerCache(num_tokens, kv_latent=kv_latent, k_rot=k_rot)
        else:
            new_cache = None
            outputs = []
            num_targets = num_tokens // len(caches)
            for index, (cache, buffers) in enumerate(zip(caches, kv_buffers)):
                rows = slice(index * num_targets, (index + 1) * num_targets)
                num_keys = cache.prefix_len + num_targets
                key, branch_value = buffers.key[:num_keys], buffers.value[:num_keys]
                prefix_key, prefix_value = cache.key, cache.value
                if prefix_key is None:
                    # Long prefix: decompress its latent into the staging rows for this step.
                    prefix_kv, _ = self.kv_b_proj(cache.kv_latent)
                    prefix_kv = prefix_kv.view(cache.prefix_len, self.num_heads, nope + self.v_head_dim)
                    key[: cache.prefix_len, :, :nope].copy_(prefix_kv[..., :nope])
                    key[: cache.prefix_len, :, nope:].copy_(cache.k_rot)
                    branch_value[: cache.prefix_len].copy_(prefix_kv[..., nope:])
                    prefix_key, prefix_value = key[:0], branch_value[:0]
                # Stage [cached prefix rows;] target rows into the tail of the buffers.
                staged = cache.prefix_len - prefix_key.shape[0]
                staged_key, staged_value = key[staged:], branch_value[staged:]
                if fused:
                    stage_kv(
                        staged_key, staged_value, prefix_key, prefix_value, kv[rows], k_rot[rows], cos[rows], sin[rows]
                    )
                else:
                    num_prefix = prefix_key.shape[0]
                    staged_key[:num_prefix].copy_(prefix_key)
                    staged_key[num_prefix:, :, :nope].copy_(k_nope[rows])
                    staged_key[num_prefix:, :, nope:].copy_(
                        apply_rotary_interleave(k_rot[rows].unsqueeze(1), cos[rows], sin[rows])
                    )
                    staged_value[:num_prefix].copy_(prefix_value)
                    staged_value[num_prefix:].copy_(value[rows])
                # Target rows sit inside the trailing bidirectional block: unmasked attention.
                outputs.append(self.attn(query[rows].unsqueeze(0), key.unsqueeze(0), branch_value.unsqueeze(0)))
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        output, _ = self.o_proj(output.reshape(num_tokens, self.num_heads * self.v_head_dim))
        return output, new_cache


# ======================================================================= MLP


class GlmImage2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        output, _ = self.down_proj(self.act_fn(gate_up))
        return output


class GlmImage2MoE(nn.Module):
    """Sigmoid-scored, bias-corrected top-k routing plus one always-on shared expert."""

    def __init__(self, config: GlmImage2TextConfig, quant_config=None, prefix: str = ""):
        super().__init__()
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        # The shared expert runs outside FusedMoE: its overlap path enqueues on
        # vLLM's tracked current stream, which diverges from the stream the
        # diffusion worker computes and transports outputs on.
        self.shared_experts = GlmImage2MLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
            # Diffusion workers do not initialise a prefill context-parallel group.
            pcp_size=None if get_current_vllm_config().parallel_config.enable_expert_parallel else 1,
        )

    def forward(self, hidden_states: torch.Tensor, num_branches: int = 1) -> torch.Tensor:
        if num_branches == 1:
            router_logits, _ = self.gate(hidden_states)
        else:
            # The fp32 router GEMM is not row-invariant in M (cuBLAS switches kernels), and
            # near-tied top-k scores can flip experts: route each CFG branch separately so a
            # batched step matches one forward per branch bit for bit.
            router_logits = torch.cat([self.gate(chunk)[0] for chunk in hidden_states.chunk(num_branches)])
        routed = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return routed + self.shared_experts(hidden_states)


class GlmImage2DecoderLayer(nn.Module):
    def __init__(self, config: GlmImage2TextConfig, layer_index: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.self_attn = GlmImage2Attention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        if config.mlp_layer_types[layer_index] == "sparse":
            self.mlp = GlmImage2MoE(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = GlmImage2MLP(
                config.hidden_size, config.intermediate_size, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        spans: list[tuple[int, int]],
        caches: list[GlmImage2LayerCache] | None,
        kv_buffers: list[GlmImage2KVBuffers] | None,
        store_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, GlmImage2LayerCache | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, new_cache = self.self_attn(hidden_states, cos, sin, spans, caches, kv_buffers, store_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if isinstance(self.mlp, GlmImage2MoE):
            return self.mlp(hidden_states, 1 if caches is None else len(caches)), residual, new_cache
        return self.mlp(hidden_states), residual, new_cache

    def decode_text(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: GlmImage2TextCache,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn.decode_text(hidden_states, cos, sin, cache, layer)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


# ===================================================== reference-image ViT


class GlmImage2VisionAttention(nn.Module):
    def __init__(self, config: GlmImage2VisionConfig, prefix: str = ""):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
            skip_sequence_parallel=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = (1, hidden_states.shape[0], self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape)
        key = self.k_proj(hidden_states).view(shape)
        value = self.v_proj(hidden_states).view(shape)
        output = self.attn(query, key, value)
        return self.out_proj(output.reshape(hidden_states.shape[0], -1))


class GlmImage2VisionMLP(nn.Module):
    def __init__(self, config: GlmImage2VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class GlmImage2VisionBlock(nn.Module):
    def __init__(self, config: GlmImage2VisionConfig, prefix: str = ""):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = GlmImage2VisionAttention(config, prefix=f"{prefix}.self_attn")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GlmImage2VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class GlmImage2VisionEmbeddings(nn.Module):
    def __init__(self, config: GlmImage2VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size)
        self.position_embedding = nn.Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor, grid: list[int], merge_size: int) -> torch.Tensor:
        """Embed one image whose patches are packed in ``merge_size`` window order."""
        temporal, height, width = grid
        if temporal != 1:
            raise ValueError("Reference images must have temporal size 1.")
        hidden_states = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        positions = torch.arange(max(height, width), device=hidden_states.device)
        rows, cols = torch.meshgrid(positions[:height], positions[:width], indexing="ij")
        window = (height // merge_size, merge_size, width // merge_size, merge_size)
        rows, cols = rows.reshape(window).transpose(1, 2).flatten(), cols.reshape(window).transpose(1, 2).flatten()
        weight = self.position_embedding.weight
        size = self.config.image_size // self.config.patch_size
        if self.config.pos_embed_interp_mode == "bilinear":
            table = weight.T.reshape(1, -1, size, size).float()
            table = F.interpolate(table, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
            positions = table[0, :, rows, cols].T
        else:
            table = weight.view(size, size, -1).permute(2, 0, 1).unsqueeze(0).float()
            sample_grid = torch.stack((((cols + 0.5) / width) * 2 - 1, ((rows + 0.5) / height) * 2 - 1), dim=-1)
            positions = F.grid_sample(
                table, sample_grid[None, :, None], mode="bicubic", align_corners=False, padding_mode="border"
            )[0, :, :, 0].T
        return hidden_states + positions.to(hidden_states.dtype)


class GlmImage2VisionModel(nn.Module):
    """Reference-image ViT; with legacy pooling it also merges 2x2 windows and L2-normalizes them."""

    def __init__(self, config: GlmImage2VisionConfig, merge_size: int, use_legacy_pooling: bool, prefix: str = ""):
        super().__init__()
        self.merge_size = merge_size
        self.embeddings = GlmImage2VisionEmbeddings(config)
        self.blocks = nn.ModuleList(
            [GlmImage2VisionBlock(config, prefix=f"{prefix}.blocks.{i}") for i in range(config.num_hidden_layers)]
        )
        self.downsample = (
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2) if use_legacy_pooling else None
        )

    def forward(self, pixel_values: torch.Tensor, grid: list[int]) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, grid, self.merge_size)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        if self.downsample is not None:
            windows = hidden_states.view(-1, 2, 2, hidden_states.shape[-1]).permute(0, 3, 1, 2)
            hidden_states = F.normalize(self.downsample(windows).flatten(1), p=2, dim=-1)
        return hidden_states


class GlmImage2VisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.post_projection_norm(self.proj(hidden_states)))
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class GlmImage2VisionAligner(nn.Module):
    """GLM-V aligner: concatenates each merge window's patch features, then projects and merges them."""

    def __init__(self, config: GlmImage2ModelConfig):
        super().__init__()
        self.downsample = nn.Linear(config.vision.hidden_size * config.cond_merge_size**2, config.text.hidden_size)
        self.merger = GlmImage2VisionPatchMerger(
            config.text.hidden_size, config.cond_vit_aligner_context_dim or config.vision.intermediate_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.merger(self.downsample(hidden_states.reshape(-1, self.downsample.in_features)))


# ============================================================ diffusion heads


class GlmImage2TimestepEmbedder(nn.Module):
    def __init__(self, frequency_dim: int, hidden_size: int):
        super().__init__()
        half = frequency_dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device="cpu") / half)
        self.register_buffer("freq", freq, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        angles = timesteps[:, None].float() * self.freq[None].to(timesteps.device)
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))


def _modulate(hidden_states: torch.Tensor, modulation: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 AdaRMSNorm: ``rms(x) * (1 + scale) + shift``."""
    scale, shift = modulation.chunk(2, dim=-1)
    normed = F.rms_norm(hidden_states.float(), (hidden_states.shape[-1],), eps=eps)
    return normed * (1 + scale) + shift


# ================================================================== top level


@dataclass
class GlmImage2TeaCacheState:
    """Per-request TeaCache state: skip the decoder layers while the timestep embedding barely moves.

    The timestep embedding predicts the decoder-residual change far better than
    the modulated latent input (calibration R^2 0.58 vs 0.10), and keeps the
    decision identical on CFG-parallel ranks. A skipped step reuses the previous
    layer-stack residual on top of the new modulated input; the timestep-dependent
    input/output heads still run.
    """

    rel_l1_thresh: float
    coefficients: list[float]
    previous_input: torch.Tensor | None = None
    residual: torch.Tensor | None = None  # fp32 [branches * T, hidden]
    accumulated: float = 0.0
    num_skipped: int = 0

    def should_compute(self, time_embedding: torch.Tensor, force: bool) -> bool:
        previous, self.previous_input = self.previous_input, time_embedding
        if force or previous is None or self.residual is None:
            self.accumulated = 0.0
            return True
        rel_l1 = ((time_embedding - previous).abs().mean() / previous.abs().mean()).item()
        self.accumulated += abs(float(np.polyval(self.coefficients, rel_l1)))
        if self.accumulated < self.rel_l1_thresh:
            self.num_skipped += 1
            return False
        self.accumulated = 0.0
        return True


@dataclass
class GlmImage2PrefixCache:
    """Per-request, per-CFG-branch state reused by every denoising step."""

    layers: list[GlmImage2LayerCache]
    kv_buffers: GlmImage2KVBuffers
    target_cos: torch.Tensor
    target_sin: torch.Tensor
    num_target_tokens: int


class GlmImage2Model(nn.Module):
    def __init__(self, config: GlmImage2ModelConfig, quant_config=None, prefix: str = ""):
        super().__init__()
        text = config.text
        self.embed_tokens = VocabParallelEmbedding(text.vocab_size, text.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleList(
            [
                GlmImage2DecoderLayer(text, index, quant_config=quant_config, prefix=f"{prefix}.layers.{index}")
                for index in range(text.num_hidden_layers)
            ]
        )
        self.rotary_emb = GlmImage2RotaryEmbedding(config)
        self.norm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.vq_visual = GlmImage2VisionModel(
            config.vision,
            merge_size=config.cond_merge_size,
            use_legacy_pooling=config.cond_vit_aligner == "legacy",
            prefix=f"{prefix}.vq_visual",
        )


class GlmImage2Transformer(nn.Module):
    """Diffusion path of ``GlmImage2ForConditionalGeneration`` (checkpoint naming preserved)."""

    # The VQ codebook exists in the checkpoint but is unused for generation.
    _skipped_checkpoint_prefixes = ("model.vq_projector.quantize.",)

    def __init__(self, config: GlmImage2ModelConfig, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        text = config.text
        self.rms_norm_eps = text.rms_norm_eps
        # FusedMoE resolves its layer through vLLM's forward context, which the
        # diffusion runner (vllm_omni.diffusion.forward_context) does not set.
        self._vllm_config = get_current_vllm_config()
        self.model = GlmImage2Model(config, quant_config=quant_config, prefix="model")
        if config.cond_vit_aligner == "glmv":
            self.cond_vit_aligner = GlmImage2VisionAligner(config)
        else:
            self.vq_to_llm_proj = nn.Linear(config.vision.hidden_size, text.hidden_size)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.diff_input_proj = nn.Linear(config.latent_dim, text.hidden_size)
        self.diff_output_proj = nn.Linear(text.hidden_size, config.latent_dim)
        self.diff_time_embed = GlmImage2TimestepEmbedder(config.diff_timestep_embed_dim, text.hidden_size)
        self.diff_adaln_in = nn.Sequential(
            nn.SiLU(), nn.Linear(text.hidden_size, 2 * text.hidden_size, dtype=torch.float32)
        )
        self.diff_adaln_out = nn.Sequential(
            nn.SiLU(), nn.Linear(text.hidden_size, 2 * text.hidden_size, dtype=torch.float32)
        )

    # ---------------------------------------------------------------- inputs

    def _embed_latents(
        self, latents: torch.Tensor, timesteps: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project packed latents and apply per-image AdaLN input modulation (one image if ``lengths`` is None)."""
        time_embedding = self.diff_time_embed(timesteps)
        modulation = self.diff_adaln_in(time_embedding)
        if lengths is None:
            modulation = modulation.expand(latents.shape[0], -1)
        else:
            modulation = modulation.repeat_interleave(lengths, dim=0)
        embeddings = self.diff_input_proj(latents.to(self.diff_input_proj.weight.dtype))
        return _modulate(embeddings.float(), modulation, self.rms_norm_eps).to(embeddings.dtype)

    def _reference_vit_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        features, cursor = [], 0
        for grid in image_grid_thw.tolist():
            count = grid[1] * grid[2]
            features.append(self.model.vq_visual(pixel_values[cursor : cursor + count], grid))
            cursor += count
        features = torch.cat(features)
        if self.config.cond_vit_aligner == "glmv":
            return self.cond_vit_aligner(features)
        return self.vq_to_llm_proj(features.to(self.vq_to_llm_proj.weight.dtype))

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        spans: list[tuple[int, int]],
        prefix_caches: list[GlmImage2PrefixCache] | None,
        store_cache: bool,
    ) -> tuple[torch.Tensor, list[GlmImage2LayerCache]]:
        residual = None
        new_caches = []
        kv_buffers = None if prefix_caches is None else [cache.kv_buffers for cache in prefix_caches]
        for index, layer in enumerate(self.model.layers):
            hidden_states, residual, new_cache = layer(
                hidden_states,
                residual,
                cos,
                sin,
                spans,
                None if prefix_caches is None else [cache.layers[index] for cache in prefix_caches],
                kv_buffers,
                store_cache,
            )
            if new_cache is not None:
                new_caches.append(new_cache)
        # The diffusion head reads the pre-final-norm residual stream.
        return hidden_states + residual, new_caches

    @contextmanager
    def _vllm_forward_context(self):
        with current_omni_platform.set_forward_context(
            None,
            self._vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=None,
        ):
            current_omni_platform.prepare_diffusion_forward_context(self)
            yield

    # --------------------------------------------------------------- forward

    @torch.inference_mode()
    def build_prefix_cache(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        diff_grid_thw: torch.Tensor,
        reference_latents: torch.Tensor | None,
        image_block_spans: list[tuple[int, int]],
        target_start: int,
    ) -> GlmImage2PrefixCache:
        """Run text and reference tokens once; they never attend to the target block."""
        prefix_ids = input_ids[:target_start]
        prefix_types = mm_token_type_ids[:target_start]
        hidden_states = self.model.embed_tokens(prefix_ids)

        vit_mask = prefix_types == VIT_TOKEN
        if bool(vit_mask.any()):
            features = self._reference_vit_features(pixel_values, image_grid_thw)
            hidden_states[vit_mask] = features.to(hidden_states.dtype)
        vae_mask = prefix_types == VAE_TOKEN
        if bool(vae_mask.any()):
            if reference_latents is None:
                raise ValueError("Reference VAE placeholders require reference latents.")
            reference_lengths = diff_grid_thw[:-1].prod(-1).to(hidden_states.device)
            zero_timesteps = torch.zeros(reference_lengths.shape[0], device=hidden_states.device)
            hidden_states[vae_mask] = self._embed_latents(reference_latents, zero_timesteps, reference_lengths).to(
                hidden_states.dtype
            )

        cos, sin = self.model.rotary_emb(position_ids)
        prefix_spans = [(start, min(end, target_start)) for start, end in image_block_spans if start < target_start]
        with self._vllm_forward_context():
            _, caches = self._run_layers(
                hidden_states, cos[:target_start], sin[:target_start], prefix_spans, None, store_cache=True
            )
        attention = self.model.layers[0].self_attn
        buffer_shape = (input_ids.shape[0], attention.num_heads)
        return GlmImage2PrefixCache(
            layers=caches,
            kv_buffers=GlmImage2KVBuffers(
                key=hidden_states.new_empty((*buffer_shape, attention.qk_head_dim)),
                value=hidden_states.new_empty((*buffer_shape, attention.v_head_dim)),
            ),
            target_cos=cos[target_start:].contiguous(),
            target_sin=sin[target_start:].contiguous(),
            num_target_tokens=input_ids.shape[0] - target_start,
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prefix_caches: list[GlmImage2PrefixCache],
        teacache: GlmImage2TeaCacheState | None = None,
        force_compute: bool = False,
    ) -> torch.Tensor:
        """Predict the velocity of the packed target latents ``[T, latent_dim]`` for each CFG branch.

        All branches share the latents and timestep; their target tokens are
        stacked so every token-wise layer (projections, MoE) runs once for all
        branches. Returns ``[len(prefix_caches), T, latent_dim]``.
        """
        num_tokens = latents.shape[0]
        if any(num_tokens != cache.num_target_tokens for cache in prefix_caches):
            raise ValueError("Target latent count does not match the prefix cache.")
        num_branches = len(prefix_caches)
        timesteps = timestep.reshape(1).to(latents.device)
        hidden_states = self._embed_latents(latents, timesteps).repeat(num_branches, 1)
        if num_branches == 1:
            cos, sin = prefix_caches[0].target_cos, prefix_caches[0].target_sin
        else:
            cos = torch.cat([cache.target_cos for cache in prefix_caches])
            sin = torch.cat([cache.target_sin for cache in prefix_caches])

        # The decoder layers update their input in place (fused add + RMSNorm), so TeaCache keeps a copy.
        layer_input = hidden_states.clone() if teacache is not None else hidden_states
        if teacache is not None and not teacache.should_compute(self.diff_time_embed(timesteps), force_compute):
            hidden_states = (layer_input.float() + teacache.residual).to(hidden_states.dtype)
        else:
            with self._vllm_forward_context():
                hidden_states, _ = self._run_layers(hidden_states, cos, sin, [], prefix_caches, False)
            if teacache is not None:
                teacache.residual = hidden_states.float() - layer_input.float()
        modulation = self.diff_adaln_out(self.diff_time_embed(timesteps)).expand(hidden_states.shape[0], -1)
        image_states = _modulate(hidden_states.float(), modulation, self.rms_norm_eps).to(hidden_states.dtype)
        velocity = self.diff_output_proj(image_states.to(self.diff_output_proj.weight.dtype))
        return velocity.view(num_branches, num_tokens, -1)

    def new_text_cache(self, max_tokens: int) -> GlmImage2TextCache:
        attention = self.model.layers[0].self_attn
        weight = self.model.embed_tokens.weight
        shape = (len(self.model.layers), max_tokens, attention.num_heads)
        return GlmImage2TextCache(
            key=weight.new_empty((*shape, attention.qk_head_dim)),
            value=weight.new_empty((*shape, attention.v_head_dim)),
        )

    @torch.inference_mode()
    def text_logits(self, input_ids: torch.Tensor, cache: GlmImage2TextCache) -> torch.Tensor:
        """Append text tokens ``[n]`` to ``cache`` with causal attention; return the fp32 next-token logits."""
        start, num_tokens = cache.length, input_ids.shape[0]
        if start + num_tokens > cache.key.shape[1]:
            raise ValueError("The GLM-Image2 text cache is full.")
        positions = torch.arange(start, start + num_tokens, device=input_ids.device).expand(3, -1)
        cos, sin = self.model.rotary_emb(positions)
        hidden_states, residual = self.model.embed_tokens(input_ids), None
        with self._vllm_forward_context():
            for index, layer in enumerate(self.model.layers):
                hidden_states, residual = layer.decode_text(hidden_states, residual, cos, sin, cache, index)
            hidden_states, _ = self.model.norm(hidden_states[-1:], residual[-1:])
        cache.length = start + num_tokens
        return self.lm_head(hidden_states)[0].float()

    # ----------------------------------------------------------------- weights

    def load_weights(self, weights) -> set[str]:
        stacked_params_mapping = [
            # (param_name, checkpoint_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            name = name.replace("model.vq_projector.pre_pool.proj.", "model.vq_visual.downsample.")
            if name.startswith(self._skipped_checkpoint_prefixes):
                continue
            if name.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
                loaded |= self._load_fused_experts(params, name, tensor)
                continue
            for param_name, checkpoint_name, shard_id in stacked_params_mapping:
                target = name.replace(checkpoint_name, param_name)
                if checkpoint_name not in name or target not in params:
                    continue
                param = params[target]
                param.weight_loader(param, tensor, shard_id)
                loaded.add(target)
                break
            else:
                if name not in params:
                    raise KeyError(f"Unexpected GLM-Image2 checkpoint tensor: {name}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
        return loaded

    def _load_fused_experts(self, params: dict[str, nn.Parameter], name: str, tensor: torch.Tensor) -> set[str]:
        """Split the checkpoint's 3D ``[E, ...]`` expert tensors into per-expert FusedMoE shards."""
        prefix = name.rsplit(".", 1)[0]  # model.layers.N.mlp.experts
        intermediate = self.config.text.moe_intermediate_size
        if name.endswith("gate_up_proj"):
            target = f"{prefix}.routed_experts.w13_weight"
            param = params[target]
            for expert_id in range(tensor.shape[0]):
                expert = tensor[expert_id]
                param.weight_loader(param, expert[:intermediate], target, shard_id="w1", expert_id=expert_id)
                param.weight_loader(param, expert[intermediate:], target, shard_id="w3", expert_id=expert_id)
        else:
            target = f"{prefix}.routed_experts.w2_weight"
            param = params[target]
            for expert_id in range(tensor.shape[0]):
                param.weight_loader(param, tensor[expert_id], target, shard_id="w2", expert_id=expert_id)
        return {target}
