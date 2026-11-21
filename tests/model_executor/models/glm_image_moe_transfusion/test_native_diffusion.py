from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.glm_image_moe_transfusion import native_diffusion


def test_standard_conv_internal_format_restores_previous_value(monkeypatch):
    values = []

    class FakeConfig:
        def __setattr__(self, name, value):
            assert name == "allow_internal_format"
            values.append(value)

    monkeypatch.setattr(
        native_diffusion.torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True,
            config=FakeConfig(),
        ),
    )
    monkeypatch.setattr(
        native_diffusion,
        "torch_npu",
        SimpleNamespace(
            _C=SimpleNamespace(_npu_getOption=lambda name: b"enable" if name == "ALLOW_INTERNAL_FORMAT" else b"")
        ),
    )

    with native_diffusion._standard_conv_internal_format():
        assert values == [False]

    assert values == [False, True]


def test_batched_packed_attention_mask_isolates_requests_and_padding():
    mask = native_diffusion._batched_packed_attention_mask(
        [2, 3],
        padded_text_len=3,
        image_len=4,
        device=torch.device("cpu"),
    )
    masked = torch.finfo(torch.bfloat16).min

    assert mask.shape == (2, 1, 7, 7)
    # Request 0's image sees its real text and image, but not request 0's
    # padding token or any request that shares the physical tensor.
    assert torch.all(mask[0, 0, 3:, :2] == 0)
    assert mask[0, 0, 3:, 2].eq(masked).all()
    assert mask[0, 0, :3, 3:].eq(masked).all()
    # Padding rows remain finite and cannot create cross-request attention.
    assert mask[0, 0, 2, 2] == 0
    assert mask[0, 0, 2, :2].eq(masked).all()


def test_select_mrope_preserves_batch_dimension():
    position_ids = torch.zeros((3, 2, 7), dtype=torch.long)
    image_mask = torch.zeros((2, 7), dtype=torch.bool)
    image_mask[:, 3:] = True

    cos, sin = native_diffusion._select_mrope(
        position_ids,
        image_mask,
        inv_freq=torch.ones(6),
        sections=[2, 2, 2],
        layout="qwen3",
    )

    assert cos.shape == (2, 7, 12)
    assert sin.shape == (2, 7, 12)
