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
        self.model = SimpleNamespace(
            backbone=SimpleNamespace(dtype=torch.float32),
        )

    def generate_batch(self, prompts, **kwargs):
        self.calls.append({"prompts": prompts, **kwargs})
        samples = len(prompts)
        return torch.arange(samples * 3 * 8 * 8, dtype=torch.float32).reshape(samples, 3, 8, 8)


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


def test_pipeline_batches_requests_and_splits_outputs() -> None:
    pipeline = object.__new__(GlmImageMoeTransfusionPipeline)
    pipeline.generator = RecordingGenerator()
    pipeline.device = torch.device("cpu")
    batch = DiffusionRequestBatch(
        requests=[
            _request("first", seed=10, num_outputs=2),
            _request("second", seed=20, num_outputs=2),
        ]
    )

    outputs = pipeline.forward(batch)

    assert GlmImageMoeTransfusionPipeline.supports_request_batch is True
    assert len(outputs) == 2
    assert [output.output.shape[0] for output in outputs] == [2, 2]
    call = pipeline.generator.calls[0]
    assert call["prompts"] == ["first", "first", "second", "second"]
    assert call["seeds"] == [10, 11, 20, 21]
    assert call["height"] == 128
    assert call["width"] == 128
    assert call["num_steps"] == 1


def test_pipeline_rejects_mixed_request_execution_shapes() -> None:
    pipeline = object.__new__(GlmImageMoeTransfusionPipeline)
    pipeline.generator = RecordingGenerator()
    pipeline.device = torch.device("cpu")
    second = _request("second", seed=20)
    second.sampling_params.num_inference_steps = 2

    with pytest.raises(ValueError, match="common size and inference-step"):
        pipeline.forward(
            DiffusionRequestBatch(
                requests=[
                    _request("first", seed=10),
                    second,
                ]
            )
        )
