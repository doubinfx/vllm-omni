# SPDX-License-Identifier: Apache-2.0
"""Serving metadata for native GLM-Image HY/MoE pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_entry_stage_id(num_stages: int, request: Mapping[str, Any]) -> int:
    """Resolve thinking requests to AR or direct requests to Diffusion."""
    think = request.get("think")
    if think is None:
        return 0
    if think:
        if num_stages < 2:
            raise ValueError("think=true requires the GLM-Image thinking service")
        return 0
    return 1 if num_stages > 1 else 0


__all__ = ["resolve_entry_stage_id"]
