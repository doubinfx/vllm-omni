# SPDX-License-Identifier: Apache-2.0
"""Checkpoint views for GLM-Image HY/MoE stages."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_ar_model_root(
    model: str,
    revision: str | None,
    task_type: str | None,
) -> str:
    del revision, task_type
    configured = os.environ.get("GLM_IMAGE_MOE_AR_MODEL")
    if configured:
        return configured

    root = Path(model).resolve()
    candidate = root.parent / "ar_model"
    if (candidate / "config.json").is_file():
        return str(candidate)
    return str(root)


__all__ = ["resolve_ar_model_root"]
