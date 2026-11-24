# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GLM-Image2 TeaCache skip decisions."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.backend import (
    CUSTOM_TEACACHE_ENABLERS,
    TeaCacheBackend,
)
from vllm_omni.diffusion.cache.teacache.config import (
    _MODEL_COEFFICIENTS,
    TeaCacheConfig,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.glm_image2.glm_image2_transformer import GlmImage2TeaCacheState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _embedding(scale: float) -> torch.Tensor:
    return torch.full((1, 8), scale)


def test_skips_until_accumulated_distance_reaches_threshold():
    # Identity rescale: the accumulated distance is the summed relative L1 change.
    state = GlmImage2TeaCacheState(rel_l1_thresh=0.25, coefficients=[1.0, 0.0])
    assert state.should_compute(_embedding(1.0), force=False)  # first step always runs
    state.residual = torch.zeros(1)
    decisions = [state.should_compute(_embedding(1.0 + 0.1 * step), force=False) for step in range(1, 6)]
    # Relative changes 0.1, 0.09, 0.083 accumulate past 0.25 on the third step, which resets.
    assert decisions == [False, False, True, False, False]
    assert state.num_skipped == 4


def test_forced_step_computes_and_resets():
    state = GlmImage2TeaCacheState(rel_l1_thresh=10.0, coefficients=[1.0, 0.0])
    state.should_compute(_embedding(1.0), force=False)
    state.residual = torch.zeros(1)
    assert not state.should_compute(_embedding(1.5), force=False)
    assert state.should_compute(_embedding(2.0), force=True)
    assert state.accumulated == 0.0


def test_glm_image2_teacache_is_registered():
    assert "GlmImage2Pipeline" in _MODEL_COEFFICIENTS
    pipeline = SimpleNamespace()
    CUSTOM_TEACACHE_ENABLERS["GlmImage2Pipeline"](pipeline, SimpleNamespace(rel_l1_thresh=0.1, coefficients=None))
    assert pipeline._tea_cache_config.rel_l1_thresh == 0.1
    assert pipeline._tea_cache_config.coefficients == _MODEL_COEFFICIENTS["GlmImage2Pipeline"]


def test_refresh_accepts_loop_managed_glm_image2_state():
    class GlmImage2Pipeline:
        pass

    pipeline = GlmImage2Pipeline()
    pipeline._tea_cache_config = TeaCacheConfig(
        transformer_type="GlmImage2Pipeline",
        rel_l1_thresh=0.2,
    )
    backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.2))

    backend.refresh(pipeline, num_inference_steps=4)
