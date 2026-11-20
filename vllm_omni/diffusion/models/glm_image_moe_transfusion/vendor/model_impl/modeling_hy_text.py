# ══════════ 搬自 veomni/models/glm4_moe_lite/initialization.py ══════════
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from collections import defaultdict
from contextlib import contextmanager

import torch


# TODO(wwx): 这里的功能是调用 torch 对应的 init 和设置 _hf_init，但其实都没用，择机删了


# Record all the torch primitives in advance, so that we can use them without them being modified when we patch torch
# in context managers
TORCH_INIT_FUNCTIONS = {
    "uniform_": torch.nn.init.uniform_,
    "normal_": torch.nn.init.normal_,
    "constant_": torch.nn.init.constant_,
    "ones_": torch.nn.init.ones_,
    "zeros_": torch.nn.init.zeros_,
    "eye_": torch.nn.init.eye_,
    "dirac_": torch.nn.init.dirac_,
    "xavier_uniform_": torch.nn.init.xavier_uniform_,
    "xavier_normal_": torch.nn.init.xavier_normal_,
    "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
    "kaiming_normal_": torch.nn.init.kaiming_normal_,
    "trunc_normal_": torch.nn.init.trunc_normal_,
    "orthogonal_": torch.nn.init.orthogonal_,
    "sparse_": torch.nn.init.sparse_,
}


def uniform_(
    tensor: torch.Tensor, a: float = 0.0, b: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["uniform_"](tensor, a=a, b=b, generator=generator)
    return tensor


def normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["normal_"](tensor, mean=mean, std=std, generator=generator)
    return tensor


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["constant_"](tensor, val=val)
    return tensor


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["ones_"](tensor)
    return tensor


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["zeros_"](tensor)
    return tensor


def eye_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["eye_"](tensor)
    return tensor


def dirac_(tensor: torch.Tensor, groups: int = 1) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["dirac_"](tensor, groups=groups)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["xavier_uniform_"](tensor, gain=gain, generator=generator)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["xavier_normal_"](tensor, gain=gain, generator=generator)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["kaiming_uniform_"](
            tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator
        )
    return tensor


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["kaiming_normal_"](
            tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator
        )
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["trunc_normal_"](tensor, mean=mean, std=std, a=a, b=b, generator=generator)
    return tensor


def orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["orthogonal_"](tensor, gain=gain, generator=generator)
    return tensor


def sparse_(
    tensor: torch.Tensor, sparsity: float, std: float = 0.01, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["sparse_"](tensor, sparsity=sparsity, std=std, generator=generator)
    return tensor


def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor


# Here, we need to check several modules imported, and hot patch all of them, as sometimes torch does
# something like `from torch.nn.init import xavier_uniform_` in their internals (e.g in torch.nn.modules.activations,
# where MultiHeadAttention lives), so the function name is binded at import time and just doing
# `setattr(torch.nn.init, name, globals()[name])` is thus not enough
# The following list should be enough for all torch versions we work with
TORCH_MODULES_TO_PATCH = (
    "torch.nn.init",
    "torch.nn.modules.activation",
    "torch.nn.modules.transformer",
    "torch.nn.modules.linear",
    "torch.nn.modules.loss",
    "torch.nn.modules.batchnorm",
    "torch.nn.modules.conv",
    "torch.nn.modules.normalization",
    "torch.nn.modules.rnn",
    "torch.nn.modules.sparse",
)


@contextmanager
def guard_torch_init_functions():
    """
    Guard the `torch.nn.init` primitive functions to behave exactly like the functions in this file, i.e. be
    protected against the `_is_hf_initialized` flag to avoid re-init if the param was already loaded.

    Usually, all models are using the init from `transformers` which are already guarded, but just to make extra sure
    and for remote code, we also use this context manager.
    """
    originals = defaultdict(dict)
    try:
        # Replace all torch funcs by the ones in this file
        for module_name in TORCH_MODULES_TO_PATCH:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for func_name in TORCH_INIT_FUNCTIONS.keys():
                    if hasattr(module, func_name):
                        originals[module][func_name] = getattr(module, func_name)
                        setattr(module, func_name, globals()[func_name])
        yield
    finally:
        # Set back the original functions on all modules
        for module, functions in originals.items():
            for func_name, func in functions.items():
                setattr(module, func_name, func)


@contextmanager
def no_init_weights():
    """
    Disable weight initialization both at the torch-level, and at the transformers-level (`init_weights`).
    This is used to speed-up initializing an empty model with deepspeed, as we do not initialize the model on meta device
    with deepspeed, but we still don't need to run expensive weight initializations as we are loading params afterwards.
    """
    from transformers.modeling_utils import PreTrainedModel

    def empty_func(*args, **kwargs):
        pass

    originals = defaultdict(dict)
    try:
        # Replace all torch funcs by empty ones
        for module_name in TORCH_MODULES_TO_PATCH:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for func_name in TORCH_INIT_FUNCTIONS.keys():
                    if hasattr(module, func_name):
                        originals[module][func_name] = getattr(module, func_name)
                        setattr(module, func_name, empty_func)

        # Also patch our own `init_weights`
        original_init_weights = PreTrainedModel.init_weights
        PreTrainedModel.init_weights = empty_func

        yield
    finally:
        # Set back the original torch functions on all modules
        for module, functions in originals.items():
            for func_name, func in functions.items():
                setattr(module, func_name, func)
        # Set back `init_weights`
        PreTrainedModel.init_weights = original_init_weights


@contextmanager
def no_tie_weights():
    """
    Disable weight tying during loading with `from_pretrained`. This is needed as we want to have access to ALL
    weights in the state_dict during `from_pretrained`, and otherwise tying them would remove them from it, as it's
    called in `post_init` when instantiating.
    """
    from transformers.modeling_utils import PreTrainedModel

    def empty_func(*args, **kwargs):
        pass

    try:
        original_tie_weights = PreTrainedModel.tie_weights
        PreTrainedModel.tie_weights = empty_func

        yield
    finally:
        # Set back the original
        PreTrainedModel.tie_weights = original_tie_weights

# ══════════ 搬自 veomni/models/glm4_moe_lite/generic.py ══════════
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic utilities
"""

from contextlib import nullcontext

import torch


# TODO(wwx): transformers 5+ generic 里的函数，但是看上去都没用，等后面确定没用了就删
def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
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


def is_flash_attention_requested(config=None, requested_attention_implementation: str | None = None):
    """
    Checks whether some flavor of flash attention is requested or not.

    This is checked against one of the two arguments, i.e. either the `config` or the directly passed value
    `requested_attention_implementation`. Otherwise, an error will be raised (ambiguity).

    The different versions of flash attention are usually
    - Implementations based on the original flash attention repo: https://github.com/Dao-AILab/flash-attention
    - Kernels implementations such as: https://huggingface.co/kernels-community/vllm-flash-attn3
    """
    if config is not None and requested_attention_implementation is not None:
        raise ValueError(
            "Requested attention implementation is ambiguous: "
            "Please pass either the config or the name of the attention implementation, not both."
        )

    if config is not None:
        checked_attention_implementation = config._attn_implementation
    else:
        checked_attention_implementation = requested_attention_implementation

    return "flash" in checked_attention_implementation

# ══════════ 搬自 veomni/models/glm4_moe_lite/npu_patch.py ══════════
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu

from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.group_gemm.kernel.npu_group_gemm import GmmFunction
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.npu_patch import npu_fused_operator

import sys as _sys_self; modeling_glm4_moe_lite = _sys_self.modules[__name__]


def glm4_moe_lite_naive_moe_forward_npu(
    self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> torch.Tensor:
    w1 = self.gate_up_proj.transpose(1, 2).to(hidden_states.dtype)
    w2 = self.down_proj.transpose(1, 2).to(hidden_states.dtype)

    permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, top_k_index.to(torch.int32))
    tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)

    gate_up_res = GmmFunction.apply(permuted_tokens, w1, tokens_per_expert)
    act_res = torch_npu.npu_swiglu(gate_up_res)
    down_res = GmmFunction.apply(act_res, w2, tokens_per_expert)

    final_hidden_states = torch_npu.npu_moe_token_unpermute(down_res, row_ids_map, probs=top_k_weights)

    return final_hidden_states


def apply_glm4moe_npu_patch():
    # Patches for GLM4 MoE Model
    modeling_glm4_moe_lite.Glm4MoeLiteRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    modeling_glm4_moe_lite.apply_rotary_pos_emb = npu_fused_operator.apply_rotary_pos_emb_npu

# ══════════ 搬自 veomni/models/glm4_moe_lite/modeling_glm4_moe_lite.py ══════════
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/glm4_moe_lite/modular_glm4_moe_lite.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_glm4_moe_lite.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from collections.abc import Callable
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    can_return_tuple,
)
from transformers.utils.generic import check_model_inputs

import sys as _sys_self; init = _sys_self.modules[__name__]
from .configuration_hy import Glm4MoeLiteConfig
# （同文件前半段已定义；原 from veomni.models.glm4_moe_lite.generic import is_flash_attention_requested, maybe_autocast）

from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops import fused_moe_forward
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.fused_moe import apply_veomni_fused_moe_patch
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.fused_moe.sonic_moe import sonic_moe_forward
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as logging
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.base import (
    is_liger_kernel_available,
    is_torch_npu_available,
)


logger = logging.get_logger(__name__)

MODEL_START_DOCSTRING = """
"""


class Glm4MoeLiteRotaryEmbedding(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Glm4MoeLiteConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
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
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GlmImageTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

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

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # e.g., x.shape = torch.Size([1, 389, 4096]) (bs, seq_len, hidden_size)
        # e.g., position_ids.shape = torch.Size([3, 1, 389]) (3, bs, seq_len))

        # In contrast to other models, GLM_IMAGE different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions) # e.g., shape = torch.Size([3, 1, 1, 389])

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling  # e.g., shape = torch.Size([3, 1, 389, 64])
            sin = emb.sin() * self.attention_scaling  # e.g., shape = torch.Size([3, 1, 389, 64])

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
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


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    Applies interleaved Rotary Position Embedding to the query and key tensors.

    DeepSeek lays the rotary dimensions out in interleaved pairs `(x0, x1), (x2, x3), ...`, each rotated by a
    single frequency. We compute that rotation directly on the even/odd slices instead of de-interleaving with a
    `view`/`transpose`/`reshape`; the output is bit-identical to the de-interleaved `rotate_half` formulation while
    avoiding the extra contiguous copy.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    # `cos`/`sin` are `cat(freqs, freqs)`; the first half holds the per-pair angle.
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)

    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class Glm4MoeLiteAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.rope_parameters = config.rope_parameters

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = Glm4MoeLiteRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = Glm4MoeLiteRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def _split_kv_b_weight(self):
        """Split ``kv_b_proj.weight`` into the K/V up-projection matrices for the
        MLA matrix-absorption path. Recomputed live each call (cheap; robust to
        FSDP (un)sharding / dtype casting).

        Returns W_UK ``[n_heads, qk_nope_head_dim, kv_lora_rank]`` and
        W_UV ``[n_heads, v_head_dim, kv_lora_rank]``.
        """
        w = self.kv_b_proj.weight
        # FSDP2(fully_shard)下 .weight 是 per-param DTensor。即便 layer 已 unshard
        # (Replicate),手动 `.view()` + 下游 einsum 仍走 DTensor dispatch,把 local
        # activation 也卷入分布式语义 → 每步每层多次 redistribute 检查/通信,且随卡数
        # 放大(实测 absorb 4卡 ~1.1s/tok → 8卡 ~18.7s/tok,17×;naive 的 F.linear 有
        # FSDP 优化路径,不受影响)。Replicate 下每卡都持全量,取 .to_local() 让 absorb 的
        # view/einsum 全部走纯本地 tensor —— 数值等价,消除该 per-step 分布式开销。
        if hasattr(w, "to_local"):
            w = w.to_local()
        w = w.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        W_UK = w[:, : self.qk_nope_head_dim, :]
        W_UV = w[:, self.qk_nope_head_dim :, :]
        return W_UK, W_UV

    def _apply_rope_qk(self, q_rot, k_rot, position_embeddings):
        """RoPE on q_rot / k_rot — npu path (optional deinterleave, then mrope).
        Shared by the full & absorb routes so they can never diverge."""
        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        return q_rot, k_rot

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # Inference (use_cache=True): store the compressed latent in the KV cache
        # and run attention in latent space via matrix absorption — numerically
        # equivalent, but caches kv_lora_rank+qk_rope_head_dim per token instead of
        # the full per-head K/V. Training (use_cache=False) keeps the original
        # naive full-K/V path → zero numerical change.
        # Inference (use_cache=True): store the compressed latent in the KV cache
        # and run attention in latent space via matrix absorption — numerically
        # equivalent, but caches kv_lora_rank+qk_rope_head_dim per token instead of
        # the full per-head K/V. Training (use_cache=False) keeps the original
        # naive full-K/V path → zero numerical change.
        # decode(use_cache)**默认走 naive full-K/V**(_forward_full):NPU 上 MLA 矩阵吸收
        # (absorb)的 per-head einsum 严重退化 —— 实测 30B 8卡 t2i decode absorb 18.7s/tok
        # vs naive 0.73s/tok(25×)。根因两层:① `einsum("bhqd,hdc->bhqc", ...)` 把一个大
        # Linear 解压拆成 20 个 head 的 M=1 小 GEMM,昇腾 cube 利用率塌方 + einsum 的
        # permute/reshape/ND↔NZ format 开销(微基准:单 einsum 比 bmm 慢 5×);② `_split_kv_b_weight`
        # 每步直接读 `kv_b_proj.weight` 绕过 FSDP forward hook,破坏 all-gather/compute
        # overlap。naive 用一个大 nn.Linear 解压 + FlashAttention(两个 NPU 优化原语),与
        # absorb 数值等价(_forward_full 即训练同款路径,KV cache 仅多 ~230MB)。
        # 仅当 GPU + 长序列 + 需省 KV-cache 显存时,显式 VEOMNI_AR_USE_ABSORB=1 启用 absorb。
        # TODO(wwx): 改成入参而不是环境变量
        import os as _os

        _use_absorb = _os.environ.get("VEOMNI_AR_USE_ABSORB", "0") == "1"
        if use_cache and past_key_values is not None and _use_absorb:
            return self._forward_absorb(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
        return self._forward_full(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    def _forward_full(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # ===== original naive full-K/V path (training & no-cache inference) =====
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = self._apply_rope_qk(q_rot, k_rot, position_embeddings)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _forward_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # ===== MLA latent-cache + matrix-absorption path (inference) =====
        # Cache stores, per token: the normalized latent c = kv_a_layernorm(k_pass)
        # of size kv_lora_rank, and the RoPE'd k_rot of size qk_rope_head_dim
        # (both shared across heads → MQA). Attention is computed in latent space.
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        # --- query: identical to the naive path ---
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)  # [B, H, Lq, qk_head_dim]
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- compressed kv: keep the latent, do NOT decompress to full K/V ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_new = self.kv_a_layernorm(k_pass)  # [B, Lq, kv_lora_rank] (kv_b input, normalized)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)  # [B, 1, Lq, rope]

        cos, sin = position_embeddings
        q_rot, k_rot = self._apply_rope_qk(q_rot, k_rot, position_embeddings)  # k_rot stored RoPE'd

        # Store latent as "key" [B, 1, L, kv_lora_rank] and roped k_rot as "value"
        # [B, 1, L, qk_rope_head_dim] (single shared head ⇒ MQA across heads).
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        c_cache, krot_cache = past_key_values.update(c_new.unsqueeze(1), k_rot, self.layer_idx, cache_kwargs)
        c_cache = c_cache.squeeze(1)  # [B, Lk, kv_lora_rank]
        krot_cache = krot_cache.squeeze(1)  # [B, Lk, qk_rope_head_dim]
        kv_len = c_cache.shape[1]

        # --- absorbed attention scores ---
        W_UK, W_UV = self._split_kv_b_weight()
        W_UK = W_UK.to(q_pass.dtype)  # [H, qk_nope_head_dim, kv_lora_rank]
        W_UV = W_UV.to(q_pass.dtype)  # [H, v_head_dim, kv_lora_rank]

        # q_pass·W_UK absorbs the K up-projection into the query (latent space).
        q_pass_absorbed = torch.einsum("bhqd,hdc->bhqc", q_pass, W_UK)  # [B, H, Lq, c]
        attn_nope = torch.einsum("bhqc,bkc->bhqk", q_pass_absorbed, c_cache.to(q_pass.dtype))
        attn_rope = torch.einsum("bhqd,bkd->bhqk", q_rot, krot_cache.to(q_rot.dtype))
        attn_weights = (attn_nope + attn_rope) * self.scaling  # [B, H, Lq, Lk]

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_len]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pass.dtype)

        # weighted sum in latent space, then absorb the V up-projection.
        out_latent = torch.einsum("bhqk,bkc->bhqc", attn_weights, c_cache.to(q_pass.dtype))  # [B,H,Lq,c]
        out = torch.einsum("bhqc,hdc->bhqd", out_latent, W_UV)  # [B, H, Lq, v_head_dim]

        attn_output = out.transpose(1, 2).reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Glm4MoeLiteMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Glm4MoeLiteTopkRouter(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.e_score_correction_bias = nn.Buffer(torch.zeros((self.n_routed_experts), dtype=torch.float32))

        # for loss free balancing
        # TODO(wwx): 不知道设置成 int32/int64 会咋样，抄的 megatron 用的是 float32
        # https://github.com/NVIDIA/Megatron-LM/blob/b964d1630ffe32736579677efc3b3327210987d0/megatron/core/transformer/moe/router.py#L187
        # 因为 aux loss based 方案也要统计 moe stats，也需要用到这个，所以 buffer 总建立
        self.router_bias_update_speed = self.config.router_bias_update_speed
        self.local_tokens_per_expert = nn.Buffer(torch.zeros((self.n_routed_experts), dtype=torch.float32))
        self.local_vision_tokens_per_expert = nn.Buffer(torch.zeros((self.n_routed_experts), dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor

        if self.training and torch.is_grad_enabled():
            # CP padding must not affect load balancing or the router bias update.
            token_mask = getattr(self, "_token_mask_for_stats", None)
            counted_indices = topk_indices if token_mask is None else topk_indices[token_mask]
            flat_topk_indices = counted_indices.flatten()
            self.local_tokens_per_expert.scatter_add_(
                0,
                flat_topk_indices,
                torch.ones(
                    flat_topk_indices.shape, device=flat_topk_indices.device, dtype=self.local_tokens_per_expert.dtype
                ),
            )
        if (self.training and torch.is_grad_enabled() and getattr(self, "_stats_enabled", True)
                and hasattr(self, "_vision_mask_for_stats")):
            flat_vision_topk_indices = topk_indices[self._vision_mask_for_stats].flatten()
            self.local_vision_tokens_per_expert.scatter_add_(
                0,
                flat_vision_topk_indices,
                torch.ones(
                    flat_vision_topk_indices.shape,
                    device=flat_vision_topk_indices.device,
                    dtype=self.local_vision_tokens_per_expert.dtype,
                ),
            )
        from .modeling_hy_extras import record_router_stats

        record_router_stats(self, topk_indices, getattr(self, "_token_mask_for_stats", None))
        return router_logits, topk_weights, topk_indices

    def update_router_bias(self, global_tokens_per_expert: torch.Tensor | None = None, dp_group=None) -> torch.Tensor:
        if global_tokens_per_expert is None:
            logger.warning_once(
                "you are using per layer allreduce update, this may cause slowdown, consider using all layers coalesced all reduce"
            )
            global_tokens_per_expert = self.local_tokens_per_expert.clone()
            dist.all_reduce(global_tokens_per_expert, op=dist.ReduceOp.SUM, group=dp_group)

        target = global_tokens_per_expert.mean()
        err = target - global_tokens_per_expert
        self.e_score_correction_bias.add_(self.router_bias_update_speed * torch.sign(err))
        return global_tokens_per_expert

    def zero_local_tokens_per_expert(self) -> None:
        self.local_tokens_per_expert.zero_()
        self.local_vision_tokens_per_expert.zero_()


class Glm4MoeLiteRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """
        Glm4MoeLiteRMSNorm is equivalent to T5LayerNorm -> nn.RMSNorm
        """
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.elementwise_affine:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm4MoeLiteExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        if config._moe_implementation == "sonic":
            self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        else:
            self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
            self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

        if config._moe_implementation == "fused":
            self.forward = self.fused_forward
        elif config._moe_implementation == "sonic":
            self.forward = self.sonic_forward
        else:
            self.forward = self.eager_forward

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert top_k_weights is not None and top_k_index is not None, (
            "top_k_weights and top_k_index must be provided when expert_idx is None"
        )

        out = fused_moe_forward(
            module=self,
            num_experts=self.num_experts,  # self.num_experts
            routing_weights=top_k_weights,  # shape: (batch * sequence_length, self.top_k)
            selected_experts=top_k_index,  # shape: (batch * sequence_length, self.top_k)
            hidden_states=hidden_states,  # shape: (batch_size * sequence_length, hidden_dim)
            fc1_1_weight=self.gate_proj,
            fc1_2_weight=self.up_proj,
            fc2_weight=self.down_proj,
        )
        # out shape: (batch_size * sequence_length, hidden_dim)
        return out

    def sonic_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return sonic_moe_forward(
            hidden_states=hidden_states,
            gate_up_weight=self.gate_up_proj,
            down_weight=self.down_proj,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=self.num_experts,
            training=self.training,
        )

    def eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = nn.functional.linear(current_state, self.gate_proj[expert_idx])
            up = nn.functional.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Glm4MoeLiteMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config  # type: Glm4MoeLiteConfig
        self.experts = Glm4MoeLiteExperts(config)
        self.gate = Glm4MoeLiteTopkRouter(config)
        self.shared_experts = Glm4MoeLiteMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits, topk_weights, topk_indices = self.gate(hidden_states)
        aux_loss = None  # TODO(wwx): 确认影响面，之前的 aux_loss 的实现是错的，logit.softmax，而且暂时用不到先 hardcode 成 None
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # TODO(wwx): 如果是 sonic moe, topk_weights 可以不做 dtype 下转，看看是否有必要
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights.to(hidden_states.dtype)).view(
            *orig_shape
        )
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states, aux_loss


class Glm4MoeLiteDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4MoeLiteAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm4MoeLiteMoE(config)
        else:
            self.mlp = Glm4MoeLiteMLP(config)

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
        **kwargs: Unpack[TransformersKwargs],
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


class Glm4MoeLitePreTrainedModel(PreTrainedModel):
    config: Glm4MoeLiteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm4MoeLiteDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # need pytorch >= 2.9.0
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Glm4MoeLiteDecoderLayer,
        "attentions": Glm4MoeLiteAttention,
    }
    _keep_in_fp32_modules_strict = []  # instead of ["e_score_correction_bias"] # to adapt 4.56.2

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Glm4MoeLiteTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Glm4MoeLiteExperts):
            if hasattr(module, "gate_up_proj"):
                init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            else:
                init.normal_(module.gate_proj, mean=0.0, std=self.config.initializer_range)
                init.normal_(module.up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class Glm4MoeLiteModel(Glm4MoeLitePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.47.*"]

    def __init__(self, config: Glm4MoeLiteConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm4MoeLiteDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.org_rotary_emb = Glm4MoeLiteRotaryEmbedding(config=config)

        self.rotary_emb = GlmImageTextRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            # position_ids=position_ids,
            position_ids=None,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Standalone (non-omni) path: aux_loss collection isn't wired through
        # this CausalLM, so we simply discard it here. The omni path
        # (GlmImageMoeTextModel) collects and propagates aux_loss properly.
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, _ = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def collect_global_tokens_per_expert(
        self, group=None, add_vision_tokens: bool = False, coalesced: bool = False
    ) -> list[torch.Tensor]:
        global_tokens_per_expert_list = []

        # MARK: coalesced 在 30B 7.5 ms -> 6ms，8H100，华为未知（理论上卡越多应该越有优势？）
        if coalesced:
            for layer in self.layers:
                if isinstance(layer.mlp, Glm4MoeLiteMoE):
                    global_tokens_per_expert_list.append(layer.mlp.gate.local_tokens_per_expert.clone())
            if add_vision_tokens:
                for layer in self.layers:
                    if isinstance(layer.mlp, Glm4MoeLiteMoE):
                        global_tokens_per_expert_list.append(layer.mlp.gate.local_vision_tokens_per_expert.clone())
            dist.all_reduce_coalesced(global_tokens_per_expert_list, group=group)
        else:
            for layer in self.layers:
                if isinstance(layer.mlp, Glm4MoeLiteMoE):
                    global_tokens_per_expert = layer.mlp.gate.local_tokens_per_expert.clone()
                    dist.all_reduce(global_tokens_per_expert, op=dist.ReduceOp.SUM)
                    global_tokens_per_expert_list.append(global_tokens_per_expert)
            if add_vision_tokens:
                for layer in self.layers:
                    if isinstance(layer.mlp, Glm4MoeLiteMoE):
                        global_tokens_per_expert = layer.mlp.gate.local_vision_tokens_per_expert.clone()
                        dist.all_reduce(global_tokens_per_expert, op=dist.ReduceOp.SUM)
                        global_tokens_per_expert_list.append(global_tokens_per_expert)

        return global_tokens_per_expert_list

    def update_router_bias(self, global_tokens_per_expert_list: list[torch.Tensor]) -> None:
        sparse_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer.mlp, Glm4MoeLiteMoE):
                global_tokens_per_expert_list[sparse_layer_idx]
                layer.mlp.gate.update_router_bias(global_tokens_per_expert_list[sparse_layer_idx])
                sparse_layer_idx += 1

    def zero_local_tokens_per_expert(self) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, Glm4MoeLiteMoE):
                layer.mlp.gate.zero_local_tokens_per_expert()


class Glm4MoeLiteForCausalLM(Glm4MoeLitePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm4MoeLiteModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Glm4MoeLiteForCausalLM

        >>> model = Glm4MoeLiteForCausalLM.from_pretrained("meta-glm4_moe_lite/Glm4MoeLite-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-glm4_moe_lite/Glm4MoeLite-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if is_liger_kernel_available():
    from copy import copy

    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    class Glm4MoeLiteLigerRMSNorm(LigerRMSNorm):
        """Keep the native optional-affine contract when using CUDA Liger."""

        def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
            super().__init__(hidden_size, eps=eps, in_place=False, elementwise_affine=elementwise_affine)

    Glm4MoeLiteRMSNorm = Glm4MoeLiteLigerRMSNorm

    class Glm4MoeLiteLigerSwiGLUMLP(LigerSwiGLUMLP):
        """Liger SwiGLU MLP with Glm4MoeLiteMLP-compatible signature."""

        def __init__(self, config, intermediate_size=None):
            if intermediate_size is None:
                super().__init__(config)
            else:
                liger_config = copy(config)
                liger_config.intermediate_size = intermediate_size
                super().__init__(liger_config)
                self.config = config

    Glm4MoeLiteMLP = Glm4MoeLiteLigerSwiGLUMLP
    logger.info_rank0("Apply liger kernel to glm4_moe_lite.")


if is_torch_npu_available():
    from .modeling_hy_text import apply_glm4moe_npu_patch

    # MARK: npu 的 rmsnorm fused 比 triton-ascend + liger 快，所以后 patch，覆盖掉
    apply_glm4moe_npu_patch()
apply_veomni_fused_moe_patch()

__all__ = [
    "Glm4MoeLitePreTrainedModel",
    "Glm4MoeLiteModel",
]

