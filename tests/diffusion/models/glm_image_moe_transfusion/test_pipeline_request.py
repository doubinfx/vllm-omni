# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.glm_image_moe_transfusion.pipeline import (
    GlmImageMoeTransfusionPipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.model = SimpleNamespace(backbone=SimpleNamespace(dtype=torch.float32))

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return torch.zeros((1, 3, 8, 8), dtype=torch.float32)


def _request(prompt: str, *, seed: int, num_outputs: int = 1) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": prompt, "modalities": ["image"]},
        sampling_params=OmniDiffusionSamplingParams(
            height=128,
            width=128,
            num_inference_steps=1,
            num_outputs_per_prompt=num_outputs,
            seed=seed,
        ),
        request_id=prompt,
    )


def test_pipeline_executes_one_request_per_forward() -> None:
    pipeline = object.__new__(GlmImageMoeTransfusionPipeline)
    pipeline.generator = RecordingGenerator()
    pipeline.device = torch.device("cpu")

    output = pipeline.forward(DiffusionRequestBatch(requests=[_request("prompt", seed=10)]))

    assert not getattr(GlmImageMoeTransfusionPipeline, "supports_request_batch", False)
    assert output.output.shape == (1, 3, 8, 8)
    assert pipeline.generator.calls == [
        {
            "prompt": "prompt",
            "height": 128,
            "width": 128,
            "num_steps": 1,
            "seed": 10,
            "generated_ids": [],
        }
    ]


def test_pipeline_rejects_multiple_requests() -> None:
    pipeline = object.__new__(GlmImageMoeTransfusionPipeline)
    pipeline.generator = RecordingGenerator()
    pipeline.device = torch.device("cpu")

    with pytest.raises(ValueError, match="one request per forward"):
        pipeline.forward(
            DiffusionRequestBatch(
                requests=[
                    _request("first", seed=10),
                    _request("second", seed=20),
                ]
            )
        )


def test_pipeline_rejects_multiple_outputs_per_request() -> None:
    pipeline = object.__new__(GlmImageMoeTransfusionPipeline)
    pipeline.generator = RecordingGenerator()
    pipeline.device = torch.device("cpu")

    with pytest.raises(ValueError, match="num_outputs_per_prompt=1"):
        pipeline.forward(
            DiffusionRequestBatch(
                requests=[_request("prompt", seed=10, num_outputs=2)]
            )
        )
