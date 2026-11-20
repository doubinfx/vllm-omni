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


# This api can improve performance on ASCEND NPU
def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    if self.weight is not None:
        if x.dtype != self.weight.dtype:
            x = x.to(
                self.weight.dtype
            )  # TODO(wwx): 这里有额外的 memory foot print，找华为的问问有没有在里面 fuse 的，但是现在走不到
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]
    else:
        # no parameter rms norm
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return x.to(x_dtype)


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_vision_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q_4d = q.unsqueeze(0).float().contiguous()
    k_4d = k.unsqueeze(0).float().contiguous()
    cos_4d = cos.unsqueeze(0).unsqueeze(2).float()
    sin_4d = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed_4d = torch_npu.npu_rotary_mul(q_4d, cos_4d, sin_4d)
    k_embed_4d = torch_npu.npu_rotary_mul(k_4d, cos_4d, sin_4d)
    q_embed = q_embed_4d.squeeze(0).to(orig_q_dtype)
    k_embed = k_embed_4d.squeeze(0).to(orig_k_dtype)
    q_embed = q_embed.reshape(orig_q_shape)
    k_embed = k_embed.reshape(orig_k_shape)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb_npu(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """NPU-fused drop-in for modeling_glm_image_omni.apply_multimodal_rotary_pos_emb.

    Mirrors the original mrope mixing on cos/sin, then fuses rotate_half + mul + add
    into a single npu_rotary_mul call. Assumes full rotation (cos rotary dim == q head dim);
    the partial-rotation wrapper apply_multimodal_rotary_pos_emb_glm slices first.
    """
    mrope_section = mrope_section * 2
    cos = (
        torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
        .unsqueeze(unsqueeze_dim)
        .to(q.dtype)
    )
    sin = (
        torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
        .unsqueeze(unsqueeze_dim)
        .to(q.dtype)
    )
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_hf_npu(q, k, cos, sin, unsqueeze_dim=1):
    """NPU-fused drop-in for modeling_glm_image_omni.apply_rotary_pos_emb_hf.

    Handles partial rotation when cos.shape[-1] < q.shape[-1] by slicing the rotated
    portion, applying npu_rotary_mul, then concatenating the pass-through tail back.
    """
    cos = cos.unsqueeze(unsqueeze_dim).to(q.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(q.dtype)
    rotary_dim = cos.shape[-1]
    if rotary_dim == q.shape[-1]:
        return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)
    q_rot = q[..., :rotary_dim].contiguous()
    k_rot = k[..., :rotary_dim].contiguous()
    q_pass = q[..., rotary_dim:]
    k_pass = k[..., rotary_dim:]
    q_embed = torch.cat([torch_npu.npu_rotary_mul(q_rot, cos, sin), q_pass], dim=-1)
    k_embed = torch.cat([torch_npu.npu_rotary_mul(k_rot, cos, sin), k_pass], dim=-1)
    return q_embed, k_embed
