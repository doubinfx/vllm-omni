# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused GLM-Image2 RoPE / K-V staging kernels must match eager bit for bit."""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.glm_image2.glm_image2_kernels import query_rope_, stage_kv
from vllm_omni.diffusion.models.glm_image2.glm_image2_transformer import apply_rotary_interleave
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

HEADS, NOPE, ROPE, VALUE, LATENT = 20, 192, 64, 256, 512


def _rotary(num_tokens: int, column_major: bool) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.randint(0, 5000, (num_tokens, 1), device="cuda").float()
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, ROPE, 2, device="cuda").float() / ROPE))
    angles = positions * inv_freq
    if column_major:
        # The prefix pass derives cos/sin from transposed position ids.
        angles = angles.t().contiguous().t()
    return angles.cos(), angles.sin()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("num_tokens", [1, 7, 4096])
@pytest.mark.parametrize("column_major", [False, True])
def test_query_rope_matches_eager(num_tokens, column_major):
    torch.manual_seed(0)
    cos, sin = _rotary(num_tokens, column_major)
    query = (torch.randn(num_tokens, HEADS, NOPE + ROPE, device="cuda") * 3).bfloat16()
    expected = query.clone()
    expected[..., NOPE:] = apply_rotary_interleave(expected[..., NOPE:], cos, sin)
    query_rope_(query, NOPE, cos, sin)
    torch.testing.assert_close(query, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("prefix_len,num_tokens", [(0, 7), (1, 1), (211, 4096), (1024, 1024)])
def test_stage_kv_matches_eager(prefix_len, num_tokens):
    torch.manual_seed(0)
    # Second CFG branch of a stacked step: row-sliced, strided inputs.
    cos, sin = _rotary(2 * num_tokens, column_major=False)
    rows = slice(num_tokens, 2 * num_tokens)
    kv = torch.randn(2 * num_tokens, HEADS * (NOPE + VALUE), device="cuda").bfloat16()
    kv = kv.view(2 * num_tokens, HEADS, NOPE + VALUE)[rows]
    k_rot = torch.randn(2 * num_tokens, LATENT + ROPE, device="cuda").bfloat16()[rows, LATENT:]
    prefix_key = torch.randn(prefix_len, HEADS, NOPE + ROPE, device="cuda").bfloat16()
    prefix_value = torch.randn(prefix_len, HEADS, VALUE, device="cuda").bfloat16()

    rotated = apply_rotary_interleave(k_rot.unsqueeze(1), cos[rows], sin[rows])
    expected_key = torch.cat((prefix_key, torch.cat((kv[..., :NOPE], rotated.expand(-1, HEADS, -1)), dim=-1)))
    expected_value = torch.cat((prefix_value, kv[..., NOPE:]))

    key = torch.full((prefix_len + num_tokens + 3, HEADS, NOPE + ROPE), float("nan"), device="cuda").bfloat16()
    value = torch.full((prefix_len + num_tokens + 3, HEADS, VALUE), float("nan"), device="cuda").bfloat16()
    num_keys = prefix_len + num_tokens
    stage_kv(key[:num_keys], value[:num_keys], prefix_key, prefix_value, kv, k_rot, cos[rows], sin[rows])
    torch.testing.assert_close(key[:num_keys], expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value[:num_keys], expected_value, rtol=0, atol=0)
    # Rows past P + T are left untouched.
    assert key[num_keys:].isnan().all() and value[num_keys:].isnan().all()
