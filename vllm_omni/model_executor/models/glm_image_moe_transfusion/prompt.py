"""Token-layout contract shared by the GLM-Image AR and diffusion stages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

USER_TOKEN_ID = 154827
ASSISTANT_TOKEN_ID = 154828
BOI_TOKEN_ID = 154830
EOP_TOKEN_ID = 154825
GEN_IMAGE_REQUEST_TOKEN_ID = 154857
SOP_TOKEN_ID = 154824

LATENT_SCALE = 16
RESOLUTION_ALIGNMENT = 32

THINKING_TEMPLATE = (
    "Generate an image based on the user's request. Before generating the image, "
    "carefully analyze the request, plan the composition, visual elements, style, "
    "and constraints, and resolve any ambiguities or inconsistencies. Explicitly "
    "present your analysis and generation plan before producing the final image:"
)
IMAGE_REQUEST_TEMPLATE = "Please generate an image according to the text below."


@dataclass(frozen=True)
class GlmImageTokenLayout:
    """Packed token layout consumed by the native diffusion stage."""

    input_ids: tuple[int, ...]
    anchor: int
    prefix_len: int
    grid_h: int
    grid_w: int
    user_text: str


def _encode(tokenizer: Any, text: str) -> list[int]:
    if callable(tokenizer):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return list(tokenizer.encode(text, add_special_tokens=False))


def _validate_resolution(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if height % RESOLUTION_ALIGNMENT or width % RESOLUTION_ALIGNMENT:
        raise ValueError(f"GLM-Image resolution must be aligned to {RESOLUTION_ALIGNMENT}, got {height}x{width}")


def build_ar_prefix_ids(
    tokenizer: Any,
    prompt: str,
    height: int,
    width: int,
) -> tuple[list[int], str]:
    """Build the exact text prefix consumed by the AR stage."""
    _validate_resolution(height, width)
    user_text = f"{THINKING_TEMPLATE} {prompt} h:{height} w:{width}"
    return [
        USER_TOKEN_ID,
        GEN_IMAGE_REQUEST_TOKEN_ID,
        *_encode(tokenizer, user_text),
        ASSISTANT_TOKEN_ID,
    ], user_text


def build_direct_prefix_ids(
    tokenizer: Any,
    prompt: str,
    height: int,
    width: int,
) -> tuple[list[int], str]:
    """Build the non-thinking T2I prefix used by the diffusion-only path."""
    _validate_resolution(height, width)
    user_text = f"{IMAGE_REQUEST_TEMPLATE} {prompt} h:{height} w:{width}"
    return [
        USER_TOKEN_ID,
        GEN_IMAGE_REQUEST_TOKEN_ID,
        *_encode(tokenizer, user_text),
        ASSISTANT_TOKEN_ID,
    ], user_text


def trim_ar_output_ids(generated_ids: Iterable[int]) -> list[int]:
    """Remove any model-written shape/image tail before diffusion packing."""
    ids = [int(token_id) for token_id in generated_ids]
    if SOP_TOKEN_ID in ids:
        return ids[: ids.index(SOP_TOKEN_ID)]
    if USER_TOKEN_ID in ids:
        return ids[: ids.index(USER_TOKEN_ID)]
    while ids and ids[-1] in (BOI_TOKEN_ID, EOP_TOKEN_ID):
        ids.pop()
    return ids


def build_diffusion_layout(
    tokenizer: Any,
    prompt: str,
    height: int,
    width: int,
    generated_ids: Iterable[int] = (),
) -> GlmImageTokenLayout:
    """Pack the text context around the generated image-latent rows.

    The final ``<|user|>`` is the conversation terminator. The diffusion rows
    are inserted immediately before it by the native pipeline.
    """
    thinking_ids = list(generated_ids)
    if thinking_ids:
        prefix_ids, user_text = build_ar_prefix_ids(tokenizer, prompt, height, width)
    else:
        prefix_ids, user_text = build_direct_prefix_ids(tokenizer, prompt, height, width)
    grid_h, grid_w = height // LATENT_SCALE, width // LATENT_SCALE
    shape_ids = _encode(tokenizer, f"<sop>{grid_h} {grid_w}<eop>")
    tokens = [
        *prefix_ids,
        *trim_ar_output_ids(thinking_ids),
        *shape_ids,
        BOI_TOKEN_ID,
        USER_TOKEN_ID,
    ]
    return GlmImageTokenLayout(
        input_ids=tuple(tokens),
        anchor=len(tokens) - 1,
        prefix_len=len(prefix_ids),
        grid_h=grid_h,
        grid_w=grid_w,
        user_text=user_text,
    )
