# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_extras import get_extra_body_params, get_image_entry_stage_resolver
from vllm_omni.model_extras.glm_image_moe_transfusion import resolve_image_entry_stage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("request_args", "num_stages", "default_stage_id", "expected"),
    [
        ({}, 2, 0, 0),
        ({"think": True}, 2, 0, 0),
        ({"think": False}, 2, 0, 1),
        ({"extra_args": {"think": False}}, 2, 0, 1),
        ({"think": False}, 1, 0, 0),
    ],
)
def test_resolve_image_entry_stage(request_args, num_stages, default_stage_id, expected) -> None:
    assert resolve_image_entry_stage(request_args, num_stages, default_stage_id) == expected


def test_resolve_image_entry_stage_rejects_missing_ar_stage() -> None:
    with pytest.raises(ValueError, match="two-stage"):
        resolve_image_entry_stage({"think": True}, 1, 0)


@pytest.mark.parametrize("think", ["false", 0, 1])
def test_resolve_image_entry_stage_requires_boolean(think) -> None:
    with pytest.raises(ValueError, match="boolean"):
        resolve_image_entry_stage({"think": think}, 2, 0)


def test_resolve_image_entry_stage_rejects_duplicate_values() -> None:
    with pytest.raises(ValueError, match="more than once"):
        resolve_image_entry_stage({"think": False, "extra_args": {"think": True}}, 2, 0)


@pytest.mark.parametrize(
    "model_class_name",
    ["GlmImageMoeTransfusionForCausalLM", "GlmImageMoeTransfusionPipeline"],
)
def test_glm_image_moe_transfusion_registers_think_routing(model_class_name) -> None:
    assert get_extra_body_params(model_class_name) == frozenset({"think"})
    assert get_image_entry_stage_resolver(model_class_name) is resolve_image_entry_stage
