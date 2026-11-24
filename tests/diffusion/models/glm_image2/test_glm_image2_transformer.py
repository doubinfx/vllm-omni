# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

from vllm_omni.diffusion.models.glm_image2.glm_image2_transformer import (
    GlmImage2Transformer,
)


def test_forward_context_uses_platform_dispatch(monkeypatch):
    from vllm_omni.diffusion.models.glm_image2 import glm_image2_transformer

    calls = []
    platform = SimpleNamespace(
        set_forward_context=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or nullcontext()
        ),
        prepare_diffusion_forward_context=lambda model: calls.append(
            ("prepare", model)
        ),
    )
    monkeypatch.setattr(
        glm_image2_transformer,
        "current_omni_platform",
        platform,
        raising=False,
    )
    model = object.__new__(GlmImage2Transformer)
    model._vllm_config = object()

    with model._vllm_forward_context():
        pass

    assert len(calls) == 2
    args, kwargs = calls[0]
    assert args == (None, model._vllm_config)
    assert kwargs["cudagraph_runtime_mode"] == glm_image2_transformer.CUDAGraphMode.NONE
    assert kwargs["batch_descriptor"] is None
    assert calls[-1] == ("prepare", model)


def test_incomplete_vllm_context_is_replaced_by_platform_context(monkeypatch):
    from contextlib import nullcontext

    from vllm_omni.diffusion.models.glm_image2 import glm_image2_transformer

    calls = []
    platform = SimpleNamespace(
        set_forward_context=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or nullcontext()
        ),
        prepare_diffusion_forward_context=lambda model: calls.append(
            ("prepare", model)
        ),
    )
    monkeypatch.setattr(
        glm_image2_transformer,
        "current_omni_platform",
        platform,
        raising=False,
    )
    monkeypatch.setattr(
        glm_image2_transformer.vllm_forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(moe_comm_type="allgather"),
    )
    model = object.__new__(GlmImage2Transformer)
    model._vllm_config = object()

    with model._vllm_forward_context():
        pass

    assert len(calls) == 2
    assert calls[-1] == ("prepare", model)
