# SPDX-License-Identifier: Apache-2.0
"""Model-specific prompt handling for native GLM-Image HY/MoE."""

from __future__ import annotations

from typing import Any

from vllm_omni.model_executor.models.glm_image_moe_transfusion.prompt import (
    build_ar_prefix_ids,
)


def build_text_to_image_tokenization(
    prompt: str,
    tokenizer: Any,
    height: int | None,
    width: int | None,
) -> dict[str, list[int]]:
    """Build the exact AR prefix used by the native GLM-Image pipeline."""
    token_ids, _ = build_ar_prefix_ids(
        tokenizer,
        prompt,
        height or 1024,
        width or 1024,
    )
    return {"prompt_token_ids": token_ids}
