from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.glm_image_moe_transfusion import a2_kernels, native_diffusion
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
            _C=SimpleNamespace(_npu_getOption=lambda name: b"enable" if name == "ALLOW_INTERNAL_FORMAT" else b"")
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


@pytest.mark.cpu
def test_a2_fused_kernels_are_disabled_by_environment(monkeypatch):
    monkeypatch.setenv(a2_kernels.A2_FUSED_OPS_ENV, "0")
    monkeypatch.setattr(a2_kernels, "is_ascend_a2", lambda: True)

    assert not a2_kernels.use_a2_fused_attention()
    assert not a2_kernels.use_a2_fused_moe()


@pytest.mark.cpu
def test_a2_fa_vendor_requires_an_ascend910b_forward_kernel(monkeypatch, tmp_path):
    vendor = tmp_path / "custom_transformer"
    for relative in a2_kernels.A2_FA_VENDOR_FILES:
        path = vendor / relative
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(a2_kernels.A2_FA_VENDOR_ENV, str(vendor))

    assert a2_kernels.registered_a2_fa_vendor() == vendor.resolve()


@pytest.mark.cpu
def test_explicit_a2_fa_vendor_rejects_an_a3_only_package(monkeypatch, tmp_path):
    vendor = tmp_path / "custom_transformer"
    (vendor / "op_api" / "lib").mkdir(parents=True)
    (vendor / "op_api" / "lib" / "libcust_opapi.so").touch()
    monkeypatch.setenv(a2_kernels.A2_FA_VENDOR_ENV, str(vendor))

    with pytest.raises(RuntimeError, match="ascend910b"):
        a2_kernels.registered_a2_fa_vendor()


@pytest.mark.cpu
def test_module_adapters_are_scoped_to_enabled_kernels():
    model = object.__new__(native_diffusion.NativeGlmImageMoeDiffusion)
    model.fused_attention = True
    model.fused_moe = False
    attention = torch.nn.Linear(4, 4)
    mlp = torch.nn.Linear(4, 4)
    layer = SimpleNamespace(self_attn=attention, mlp=mlp)
    model.backbone = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    original_attention_forward = attention.forward.__func__
    original_mlp_forward = mlp.forward.__func__
    model._install_module_adapters()

    assert attention.forward.__func__ is not original_attention_forward
    assert mlp.forward.__func__ is original_mlp_forward


@pytest.mark.cpu
def test_module_adapters_leave_both_paths_untouched_when_disabled():
    model = object.__new__(native_diffusion.NativeGlmImageMoeDiffusion)
    model.fused_attention = False
    model.fused_moe = False
    attention = torch.nn.Linear(4, 4)
    mlp = torch.nn.Linear(4, 4)
    layer = SimpleNamespace(self_attn=attention, mlp=mlp)
    model.backbone = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    original_attention_forward = attention.forward.__func__
    original_mlp_forward = mlp.forward.__func__
    model._install_module_adapters()

    assert attention.forward.__func__ is original_attention_forward
    assert mlp.forward.__func__ is original_mlp_forward
