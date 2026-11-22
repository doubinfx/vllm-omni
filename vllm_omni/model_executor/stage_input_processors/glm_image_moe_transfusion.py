# SPDX-License-Identifier: Apache-2.0
"""AR-to-diffusion handoff for GLM-Image HY/MoE thinking mode."""

from __future__ import annotations

from typing import Any


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    if isinstance(value, list):
        return _as_mapping(value[0]) if value else {}
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def ar2diffusion(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> dict[str, Any] | None:
    """Pass the sampled thinking tokens to the native diffusion stage."""
    del streaming_context, requires_multimodal_data
    if not source_outputs:
        return None
    output = source_outputs[0].outputs[0]
    generated_token_ids = [
        int(token) for token in getattr(output, "cumulative_token_ids", ())
    ]
    original = _as_mapping(prompt)
    processor_kwargs = original.get("mm_processor_kwargs") or {}

    def dimension(name: str, default: int) -> int:
        for value in (
            processor_kwargs.get(f"target_{name[0]}"),
            original.get(name),
        ):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return default

    return {
        "prompt": str(original.get("prompt") or ""),
        "height": dimension("height", 1024),
        "width": dimension("width", 1024),
        "extra": {"generated_token_ids": generated_token_ids},
    }


__all__ = ["ar2diffusion"]
