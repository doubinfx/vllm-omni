from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.glm_image_moe_transfusion import native_diffusion
from vllm_omni.model_executor.models.glm_image_moe_transfusion import ascend_compat


def test_standard_conv_internal_format_restores_previous_value(monkeypatch):
    values = []

    class FakeConfig:
        def __setattr__(self, name, value):
            assert name == "allow_internal_format"
            values.append(value)

    monkeypatch.setattr(
        ascend_compat.torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True,
            config=FakeConfig(),
        ),
    )
    monkeypatch.setattr(
        ascend_compat,
        "_get_torch_npu",
        lambda: SimpleNamespace(
            _C=SimpleNamespace(
                _npu_getOption=lambda name: b"enable" if name == "ALLOW_INTERNAL_FORMAT" else b""
            )
        ),
    )
    monkeypatch.setattr(ascend_compat, "is_ascend_a2", lambda: True)

    with ascend_compat.standard_conv_internal_format():
        assert values == [False]

    assert values == [False, True]


def test_standard_conv_internal_format_is_a2_only(monkeypatch):
    values = []

    class FakeConfig:
        def __setattr__(self, name, value):
            assert name == "allow_internal_format"
            values.append(value)

    monkeypatch.setattr(
        ascend_compat.torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, config=FakeConfig()),
    )
    monkeypatch.setattr(ascend_compat, "is_ascend_a2", lambda: False)

    with ascend_compat.standard_conv_internal_format():
        assert values == []

    assert values == []


def test_packed_attention_mask_covers_text_and_image_rows():
    mask = native_diffusion._packed_attention_mask(
        text_len=3,
        image_len=4,
        device=torch.device("cpu"),
    )
    masked = torch.finfo(torch.bfloat16).min

    assert mask.shape == (1, 1, 7, 7)
    assert torch.all(mask[0, 0, 3:, :3] == 0)
    assert torch.all(mask[0, 0, 3:, 3:] == 0)
    assert mask[0, 0, :3, 3:].eq(masked).all()


def test_select_mrope_returns_single_request_tables():
    position_ids = torch.zeros((3, 7), dtype=torch.long)
    image_mask = torch.zeros((7,), dtype=torch.bool)
    image_mask[3:] = True

    cos, sin = native_diffusion._select_mrope(
        position_ids,
        image_mask,
        inv_freq=torch.ones(6),
        sections=[2, 2, 2],
        layout="qwen3",
    )

    assert cos.shape == (1, 7, 12)
    assert sin.shape == (1, 7, 12)
