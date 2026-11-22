# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.model_loader import diffusers_loader


def test_loader_skips_generic_materialization_for_model_owned_weights(monkeypatch):
    model = torch.nn.Linear(2, 2)
    model.loads_own_weights = True
    model.train()

    loader = object.__new__(diffusers_loader.DiffusersPipelineLoader)
    loader.od_config = SimpleNamespace(dtype=torch.float32)
    loader.parallel_config = SimpleNamespace(use_hsdp=False)
    loader.quant_config = None
    monkeypatch.setattr(loader, "_init_from_load_format", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        loader,
        "load_weights",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic loader must not run")),
    )

    loaded = loader.load_model(load_device="cpu", load_format="custom_pipeline")

    assert loaded is model
    assert not model.training
