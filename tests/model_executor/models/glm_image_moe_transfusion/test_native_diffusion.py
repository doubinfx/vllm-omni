from types import SimpleNamespace

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
