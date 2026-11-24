# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GLM-Image2 thinking: reasoning sampling, stop rules and the packed image request."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.glm_image2.glm_image2_processor import THINKING_TEMPLATE, GlmImage2Processor
from vllm_omni.diffusion.models.glm_image2.glm_image2_thinking import (
    BOI_ID,
    EOP_ID,
    SOP_ID,
    THINK_END_ID,
    THINK_START_ID,
    USER_ID,
    generate_reasoning,
    sample_next_token,
    split_reasoning,
)
from vllm_omni.diffusion.models.glm_image2.pipeline_glm_image2 import (
    GlmImage2Pipeline,
    get_glm_image2_post_process_func,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

ASSISTANT_ID, GEN_REQUEST_ID, VAE_ID = 154828, 154857, 154856


class _ScriptedDecoder:
    """Emits ``script`` one token per ``text_logits`` call through one-hot logits."""

    def __init__(self, script: list[int]):
        self.script, self.calls, self.max_tokens = script, [], None
        self.model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1)))

    def new_text_cache(self, max_tokens: int):
        self.max_tokens = max_tokens
        return SimpleNamespace(length=0)

    def text_logits(self, input_ids: torch.Tensor, cache) -> torch.Tensor:
        self.calls.append(input_ids.tolist())
        logits = torch.full((160_000,), -1e4)
        logits[self.script[len(self.calls) - 1]] = 1e4
        return logits


class _CharTokenizer:
    """One id per character, except the special tokens."""

    specials = {"<sop>": SOP_ID, "<eop>": EOP_ID, "<|begin_of_image|>": BOI_ID, "<|user|>": USER_ID}

    def encode(self, text, is_pretokenized=False, add_special_tokens=False):
        ids = []
        for word in text if is_pretokenized else [text]:
            for piece in re.split(f"({'|'.join(map(re.escape, self.specials))})", word):
                ids += [self.specials[piece]] if piece in self.specials else [ord(c) for c in piece]
        return SimpleNamespace(ids=ids)


def _processor() -> GlmImage2Processor:
    processor = object.__new__(GlmImage2Processor)
    processor.tokenizer = _CharTokenizer()
    processor.patch_size, processor.merge_size, processor.vae_compression_ratio = 16, 2, 16
    processor.min_pixels, processor.max_pixels = 32 * 32, 2048 * 2048
    processor.vae_token_id, processor.gen_request_token_id = VAE_ID, GEN_REQUEST_ID
    processor.user_token_id, processor.assistant_token_id = USER_ID, ASSISTANT_ID
    return processor


def test_sample_next_token_is_seeded_and_truncated():
    logits = torch.linspace(0, 8, 100)
    draws = [sample_next_token(logits, torch.Generator().manual_seed(seed)) for seed in range(20)]
    assert draws == [sample_next_token(logits, torch.Generator().manual_seed(seed)) for seed in range(20)]
    assert min(draws) >= 50  # top-k 50 keeps the largest half
    assert sample_next_token(logits, torch.Generator(), temperature=0) == 99


def test_top_k_sampling_preserves_existing_token_sequence():
    def reference(logits, generator):
        logits = logits.float() / 0.6
        threshold = torch.topk(logits, 50).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = logits.softmax(-1)
        ordered, order = torch.sort(probs, descending=True)
        ordered = ordered.masked_fill(
            ordered.cumsum(-1) - ordered > 0.9,
            0,
        )
        probs = torch.zeros_like(probs).index_copy(0, order, ordered)
        return int(
            torch.multinomial(
                probs / probs.sum(),
                1,
                generator=generator,
            )
        )

    logits = torch.randn(160_000)
    for seed in range(20):
        expected = reference(
            logits,
            torch.Generator().manual_seed(seed),
        )
        actual = sample_next_token(
            logits,
            torch.Generator().manual_seed(seed),
        )
        assert actual == expected


def test_generate_reasoning_stops_before_the_model_size_tokens():
    script = [THINK_START_ID, 11, THINK_END_ID, 12, 13, SOP_ID, 99]
    decoder = _ScriptedDecoder(script)
    assert generate_reasoning(decoder, [1, 2, 3], seed=0, max_new_tokens=16) == script[:5]
    assert decoder.calls[0] == [1, 2, 3] and all(len(call) == 1 for call in decoder.calls[1:])
    assert decoder.max_tokens == 3 + 16


def test_generate_reasoning_strips_trailing_image_markers():
    decoder = _ScriptedDecoder([11, THINK_END_ID, 12, EOP_ID, BOI_ID])
    assert generate_reasoning(decoder, [1], seed=0) == [11, THINK_END_ID, 12]


def test_generate_reasoning_respects_the_token_budget():
    assert generate_reasoning(_ScriptedDecoder([11] * 8), [1], seed=0, max_new_tokens=4) == [11] * 4


def test_split_reasoning_separates_the_final_caption():
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: " ".join(map(str, ids)))
    assert split_reasoning([THINK_START_ID, 1, 2, THINK_END_ID, 3, 4], tokenizer) == ("1 2", "3 4")
    assert split_reasoning([1, 2], tokenizer) == ("1 2", "")


def test_thinking_prefix_wraps_the_request_with_the_target_size():
    request = f"{THINKING_TEMPLATE} a cat h:992 w:992"
    expected = [USER_ID, GEN_REQUEST_ID, *map(ord, request), ASSISTANT_ID]
    assert _processor().thinking_prefix("a cat", 1000, 1000) == expected


def test_thinking_inputs_append_the_target_block():
    inputs = _processor().thinking_inputs([USER_ID, 7, 8], 64, 96)
    text = [USER_ID, 7, 8, SOP_ID, ord("4"), ord(" "), ord("6"), EOP_ID, BOI_ID]
    start = len(text)
    assert inputs.input_ids.tolist() == text + [VAE_ID] * 24
    assert inputs.target_start == start
    assert inputs.image_block_spans() == [(start, start + 24)]
    assert inputs.diff_grid_thw.tolist() == [[1, 4, 6]] and inputs.is_ref_vae.tolist() == [False]
    assert inputs.position_ids[:, :start].tolist() == [list(range(start))] * 3
    assert inputs.position_ids[:, start:].tolist() == [
        [start] * 24,
        [start + row for row in range(4) for _ in range(6)],
        [start + col for _ in range(4) for col in range(6)],
    ]
    assert inputs.pixel_values.shape == (0, 768) and not inputs.reference_images


def test_post_process_keeps_the_thinking_text():
    post_process = get_glm_image2_post_process_func(None)
    image = torch.zeros(1, 3, 4, 4)
    assert [item.size for item in post_process(image)] == [(4, 4)]
    metadata = {"text": {"think_text": "plan", "text_output": "caption"}}
    output = post_process({"payload": {"image": image}, "metadata": metadata})
    assert output["metadata"] == metadata
    assert [item.size for item in output["payload"]["image"]] == [(4, 4)]


@pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
def test_npu_post_process_matches_float_transfer_path():
    post_process = get_glm_image2_post_process_func(None)
    torch.manual_seed(2026)
    image = torch.rand(2, 3, 32, 32, dtype=torch.float32) * 2 - 1
    device_image = image.to("npu")
    expected = (
        (image / 2 + 0.5)
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .numpy()
        * 255
    ).round().astype("uint8")
    actual = np.asarray(post_process(device_image)[0])
    assert np.array_equal(actual, expected[0])


@pytest.mark.parametrize("guidance_scale, prompt_format", [(4.0, "plain"), (1.0, "calico_structure")])
def test_thinking_rejects_unsupported_requests_as_client_errors(guidance_scale, prompt_format):
    pipeline = GlmImage2Pipeline.__new__(GlmImage2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.default_num_inference_steps, pipeline.default_guidance_scale, pipeline.default_sampler = 4, 1.0, "ode"
    params = SimpleNamespace(
        height=64,
        width=64,
        num_inference_steps=None,
        guidance_scale=guidance_scale,
        guidance_scale_provided=True,
        extra_args={"enable_thinking": True, "prompt_format": prompt_format},
        seed=0,
    )
    request = SimpleNamespace(prompts=["a cat"], sampling_params=params, is_dummy_run=lambda: False)
    with pytest.raises(OmniClientError, match="thinking"):
        pipeline.forward(request)
