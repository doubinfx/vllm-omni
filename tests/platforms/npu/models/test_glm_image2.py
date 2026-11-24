# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.models import glm_image2


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


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


def test_a2_dense_attention_uses_native_fusion(monkeypatch):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    calls = []
    output = torch.randn(1, 4, 2, 8)
    monkeypatch.setattr(glm_image2, "_is_a2", lambda: True)
    monkeypatch.setattr(
        glm_image2,
        "_get_torch_npu",
        lambda: SimpleNamespace(
            npu_fusion_attention=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or (output,)
            )
        ),
    )
    impl = FlashAttentionImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=0.25,
        causal=False,
    )
    query = torch.randn(1, 4, 2, 8)
    keep_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    keep_mask[..., -1] = False

    result = glm_image2._forward_a2_dense_attention(
        impl,
        query,
        query,
        query,
        AttentionMetadata(attn_mask=keep_mask),
    )

    assert result is output
    _, kwargs = calls[0]
    assert kwargs["input_layout"] == "BSND"
    assert kwargs["sparse_mode"] == 1
    assert torch.equal(kwargs["atten_mask"], ~keep_mask)


@pytest.mark.skipif(
    not _npu_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize(
    ("hidden_size", "rows"),
    [(512, 1), (768, 1), (2048, 1), (2048, 64)],
)
def test_exact_a2_rms_norm_matches_vllm_ir(hidden_size, rows):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    with set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device="npu"))
    ):
        module = RMSNorm(hidden_size, eps=1e-6)
    module.to("npu")
    module.weight.data = module.weight.data.to(torch.bfloat16)
    torch.manual_seed(2026 + hidden_size + rows)
    hidden = torch.randn(rows, hidden_size, dtype=torch.bfloat16, device="npu")
    residual = torch.randn_like(hidden)

    expected = module.forward_native(hidden)
    expected_add, expected_residual = module.forward_native(
        hidden,
        residual.clone(),
    )
    module._a2_rms_norm_fallback = module.forward_native
    actual = glm_image2._apply_exact_a2_rms_norm(module, hidden)
    actual_add, actual_residual = glm_image2._apply_exact_a2_rms_norm(
        module,
        hidden,
        residual.clone(),
    )

    assert torch.equal(actual, expected)
    assert torch.equal(actual_add, expected_add)
    assert torch.equal(actual_residual, expected_residual)
