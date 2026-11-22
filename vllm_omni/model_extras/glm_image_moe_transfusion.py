# SPDX-License-Identifier: Apache-2.0
"""Model-specific prompt handling for native GLM-Image HY/MoE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_executor.models.glm_image_moe_transfusion.prompt import (
    ASSISTANT_TOKEN_ID,
    BOI_TOKEN_ID,
    EOP_TOKEN_ID,
    SOP_TOKEN_ID,
    USER_TOKEN_ID,
    build_ar_prefix_ids,
)

_EXPECTED_TOKENS = {
    USER_TOKEN_ID: "<|user|>",
    ASSISTANT_TOKEN_ID: "<|assistant|>",
    BOI_TOKEN_ID: "<|begin_of_image|>",
    SOP_TOKEN_ID: "<sop>",
    EOP_TOKEN_ID: "<eop>",
}

GLM_IMAGE_MOE_TRANSFUSION_EXTRA_BODY_PARAMS = frozenset({"think"})


def resolve_image_entry_stage(
    request_args: Mapping[str, object],
    num_stages: int,
    default_stage_id: int,
) -> int:
    """Route thinking requests through AR and direct requests to diffusion."""
    sources: list[object] = []
    for container in (
        request_args,
        request_args.get("extra_args"),
        request_args.get("extra_params"),
    ):
        if isinstance(container, Mapping) and "think" in container:
            sources.append(container["think"])
    if len(sources) > 1:
        raise ValueError("think was provided more than once")
    if not sources or sources[0] is None:
        return default_stage_id

    think = sources[0]
    if not isinstance(think, bool):
        raise ValueError("think must be a boolean")
    if think:
        if num_stages < 2:
            raise ValueError("think=true requires a two-stage GLM-Image service")
        return 0
    return 1 if num_stages > 1 else 0


def validate_glm_image_moe_tokenizer(tokenizer: Any) -> None:
    """Fail loudly if the tokenizer no longer matches the native token layout."""
    mismatches = {
        token_id: (token, tokenizer.convert_ids_to_tokens(token_id))
        for token_id, token in _EXPECTED_TOKENS.items()
        if tokenizer.convert_ids_to_tokens(token_id) != token
    }
    if mismatches:
        details = ", ".join(
            f"{token_id}: expected {expected!r}, got {actual!r}" for token_id, (expected, actual) in mismatches.items()
        )
        raise ValueError(f"GLM-Image tokenizer special-token mismatch: {details}")


def build_text_to_image_tokenization(
    prompt: str,
    tokenizer: Any,
    height: int | None,
    width: int | None,
) -> dict[str, list[int]]:
    """Build the exact AR prefix used by the native GLM-Image pipeline."""
    validate_glm_image_moe_tokenizer(tokenizer)
    token_ids, _ = build_ar_prefix_ids(
        tokenizer,
        prompt,
        height or 1024,
        width or 1024,
    )
    return {"prompt_token_ids": token_ids}
