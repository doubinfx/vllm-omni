# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import torch

from vllm_omni.platforms.npu.models import glm_image2


def test_repeat_rotary_tables_interleaves_frequency_pairs():
    cos = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    sin = torch.tensor([[4.0, 5.0, 6.0, 7.0]])

    expanded_cos, expanded_sin = glm_image2._repeat_rotary_tables(cos, sin)

    assert expanded_cos.tolist() == [[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]
    assert expanded_sin.tolist() == [[4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0]]


def test_npu_rotary_input_uses_bsnd_layout(monkeypatch):
    calls = []

    def rotary(hidden, cos, sin, *, rotary_mode):
        calls.append((hidden, cos, sin, rotary_mode))
        return hidden

    monkeypatch.setattr(
        glm_image2,
        "_get_torch_npu",
        lambda: SimpleNamespace(npu_rotary_mul=rotary),
    )
    hidden = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    cos = torch.randn(3, 8)
    sin = torch.randn(3, 8)

    output = glm_image2._apply_npu_rotary_interleave(hidden, cos, sin)

    assert output.shape == hidden.shape
    layout, cos_arg, sin_arg, mode = calls[0]
    assert layout.shape == (1, 3, 4, 8)
    assert cos_arg.shape == (1, 3, 1, 8)
    assert sin_arg.shape == (1, 3, 1, 8)
    assert mode == "interleave"


def test_patch_is_scoped_to_glm_image2_pipeline(monkeypatch):
    monkeypatch.setattr(glm_image2, "_install_a2_rotary_patches", lambda: None)

    assert glm_image2.apply_glm_image2_npu_patches(object(), object()) is False
