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


def test_batched_attention_mask_pads_text_without_cross_request_rows():
    mask = native_diffusion._packed_attention_mask_batch(
        text_lengths=[2, 3],
        image_len=2,
        device=torch.device("cpu"),
    )
    masked = torch.finfo(torch.bfloat16).min

    assert mask.shape == (2, 1, 5, 5)
    assert mask[0, 0, 3:, :2].eq(0).all()
    assert mask[0, 0, 3:, 2].eq(masked).all()
    assert mask[1, 0, 3:, :3].eq(0).all()
    assert mask[0, 0, 2, 2].eq(0)


def test_batched_positions_keep_request_local_text_lengths():
    model = object.__new__(native_diffusion.NativeGlmImageMoeDiffusion)
    model.extra_device = torch.device("cpu")
    layouts = [
        native_diffusion.GlmImageTokenLayout(
            input_ids=(1, 2, 3),
            anchor=2,
            prefix_len=2,
            grid_h=1,
            grid_w=1,
            user_text="short",
        ),
        native_diffusion.GlmImageTokenLayout(
            input_ids=(1, 2, 4, 3),
            anchor=3,
            prefix_len=3,
            grid_h=1,
            grid_w=1,
            user_text="longer",
        ),
    ]

    positions, image_mask, padded_text_len, image_len = model._batch_position_tensors(layouts)

    assert padded_text_len == 3
    assert image_len == 1
    assert positions.shape == (2, 3, 4)
    assert positions[0, 0, -1].item() == 2
    assert positions[1, 0, -1].item() == 3
    assert image_mask[:, -1].all()
    assert not image_mask[:, :padded_text_len].any()


def test_adaln_broadcasts_request_local_conditioning():
    hidden = torch.randn((2, 5, 8), dtype=torch.float32)
    scale = torch.randn((2, 8), dtype=torch.float32)
    shift = torch.randn((2, 8), dtype=torch.float32)

    batched = native_diffusion._adaln_apply(
        hidden,
        scale.unsqueeze(1),
        shift.unsqueeze(1),
        1e-5,
    )
    separate = torch.cat(
        [
            native_diffusion._adaln_apply(
                hidden[index : index + 1],
                scale[index : index + 1],
                shift[index : index + 1],
                1e-5,
            )
            for index in range(2)
        ]
    )

    assert torch.allclose(batched, separate)


def test_a2_rotary_splits_unsupported_batch_input(monkeypatch):
    calls = []

    def rotary(hidden, cos, sin, *, rotary_mode):
        calls.append((hidden.shape, cos.shape, sin.shape, rotary_mode))
        return hidden

    monkeypatch.setattr(
        a2_kernels,
        "_get_npu",
        lambda: SimpleNamespace(npu_rotary_mul=rotary),
    )
    hidden = torch.randn((2, 3, 4, 8))
    cos = torch.randn((2, 3, 8))
    sin = torch.randn((2, 3, 8))

    output = a2_kernels._apply_rotary(hidden, cos, sin, interleaved=False)

    assert output.shape == hidden.shape
    assert len(calls) == 2
    assert all(shape[0] == 1 for shape, _, _, _ in calls)


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
