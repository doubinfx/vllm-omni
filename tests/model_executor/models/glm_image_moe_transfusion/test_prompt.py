from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.glm_image_moe_transfusion.prompt import (
    ASSISTANT_TOKEN_ID,
    BOI_TOKEN_ID,
    EOP_TOKEN_ID,
    GEN_IMAGE_REQUEST_TOKEN_ID,
    IMAGE_REQUEST_TEMPLATE,
    SOP_TOKEN_ID,
    THINKING_TEMPLATE,
    USER_TOKEN_ID,
    build_ar_prefix_ids,
    build_diffusion_layout,
    build_direct_prefix_ids,
    trim_ar_output_ids,
    validate_glm_image_moe_tokenizer,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.calls.append(text)
        if text == "<sop>32 32<eop>":
            return [SOP_TOKEN_ID, 32, 32, EOP_TOKEN_ID]
        if text.startswith(THINKING_TEMPLATE):
            return [100, 101]
        return [1000 + ord(char) for char in text]


def test_build_ar_prefix_matches_training_layout() -> None:
    tokenizer = RecordingTokenizer()
    ids, user_text = build_ar_prefix_ids(
        tokenizer,
        "a red cube",
        height=512,
        width=512,
    )

    assert ids == [
        USER_TOKEN_ID,
        GEN_IMAGE_REQUEST_TOKEN_ID,
        100,
        101,
        ASSISTANT_TOKEN_ID,
    ]
    assert user_text == f"{THINKING_TEMPLATE} a red cube h:512 w:512"


def test_build_diffusion_layout_replaces_model_shape_tail() -> None:
    tokenizer = RecordingTokenizer()
    generated = [200, 201, SOP_TOKEN_ID, 9, BOI_TOKEN_ID]
    layout = build_diffusion_layout(
        tokenizer,
        "a red cube",
        height=512,
        width=512,
        generated_ids=generated,
    )

    assert layout.input_ids == (
        USER_TOKEN_ID,
        GEN_IMAGE_REQUEST_TOKEN_ID,
        100,
        101,
        ASSISTANT_TOKEN_ID,
        200,
        201,
        SOP_TOKEN_ID,
        32,
        32,
        EOP_TOKEN_ID,
        BOI_TOKEN_ID,
        USER_TOKEN_ID,
    )
    assert layout.anchor == len(layout.input_ids) - 1
    assert layout.prefix_len == 5
    assert (layout.grid_h, layout.grid_w) == (32, 32)


def test_build_diffusion_layout_uses_direct_prefix_without_thinking() -> None:
    tokenizer = RecordingTokenizer()
    layout = build_diffusion_layout(
        tokenizer,
        "a red cube",
        height=512,
        width=512,
    )

    assert layout.input_ids[:2] == (USER_TOKEN_ID, GEN_IMAGE_REQUEST_TOKEN_ID)
    assert layout.input_ids[-3:] == (EOP_TOKEN_ID, BOI_TOKEN_ID, USER_TOKEN_ID)
    assert tokenizer.calls == [
        f"{IMAGE_REQUEST_TEMPLATE} a red cube h:512 w:512",
        "<sop>32 32<eop>",
    ]
    assert not any(call.startswith(THINKING_TEMPLATE) for call in tokenizer.calls)


def test_build_direct_prefix_uses_resolution_suffix() -> None:
    tokenizer = RecordingTokenizer()
    ids, user_text = build_direct_prefix_ids(
        tokenizer,
        "a red cube",
        height=512,
        width=512,
    )

    assert ids[:2] == [USER_TOKEN_ID, GEN_IMAGE_REQUEST_TOKEN_ID]
    assert ids[-1] == ASSISTANT_TOKEN_ID
    assert user_text == f"{IMAGE_REQUEST_TEMPLATE} a red cube h:512 w:512"


def test_trim_ar_output_removes_trailing_image_markers() -> None:
    assert trim_ar_output_ids([10, 11, EOP_TOKEN_ID, BOI_TOKEN_ID]) == [10, 11]


def test_trim_ar_output_removes_conversation_terminator() -> None:
    assert trim_ar_output_ids([10, 11, USER_TOKEN_ID, 12]) == [10, 11]


@pytest.mark.parametrize(
    ("height", "width"),
    [(0, 512), (512, 0), (504, 512), (512, 500)],
)
def test_resolution_alignment(height: int, width: int) -> None:
    with pytest.raises(ValueError):
        build_ar_prefix_ids(RecordingTokenizer(), "prompt", height, width)


def test_tokenizer_special_ids_must_match_layout() -> None:
    expected = {
        SOP_TOKEN_ID: "<sop>",
        EOP_TOKEN_ID: "<eop>",
        USER_TOKEN_ID: "<|user|>",
        ASSISTANT_TOKEN_ID: "<|assistant|>",
        BOI_TOKEN_ID: "<|begin_of_image|>",
    }

    class Tokenizer:
        def convert_ids_to_tokens(self, token_id: int):
            return expected.get(token_id)

    validate_glm_image_moe_tokenizer(Tokenizer())

    class DriftedTokenizer(Tokenizer):
        def convert_ids_to_tokens(self, token_id: int):
            return "wrong-token"

    with pytest.raises(ValueError, match="special-token mismatch"):
        validate_glm_image_moe_tokenizer(DriftedTokenizer())
