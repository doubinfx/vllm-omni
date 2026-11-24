# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GLM-Image2 thinking mode: the decoder samples a chain of thought and a final
caption, then generates the image conditioned on them.

The checkpoint marks the chain with its training-time ``<think>``/``</think>``
ids (151350/151351), which the shipped tokenizer decodes as ordinary text.
"""

from __future__ import annotations

import torch

THINK_START_ID = 151350
THINK_END_ID = 151351
SOP_ID = 154824
EOP_ID = 154825
BOI_ID = 154830
USER_ID = 154827
MAX_NEW_TOKENS = 4072


def sample_next_token(
    logits: torch.Tensor,
    generator: torch.Generator,
    *,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """Sample on CPU so a seed reproduces the chain on any device."""
    logits = logits.detach().float().cpu()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    candidate_ids = None
    if 0 < top_k < logits.numel():
        candidates, candidate_ids = torch.topk(logits, top_k)
    else:
        candidates = logits
    probs = candidates.softmax(-1)
    if 0 < top_p < 1:
        ordered, order = torch.sort(probs, descending=True)
        ordered = ordered.masked_fill(ordered.cumsum(-1) - ordered > top_p, 0)
        if candidate_ids is not None:
            probs = torch.zeros_like(logits).index_copy(
                0,
                candidate_ids[order],
                ordered,
            )
            candidate_ids = None
        else:
            probs = torch.zeros_like(probs).index_copy(0, order, ordered)
    elif candidate_ids is not None:
        probs = torch.zeros_like(logits).index_copy(
            0,
            candidate_ids,
            probs,
        )
        candidate_ids = None
    sampled = int(torch.multinomial(probs / probs.sum(), 1, generator=generator))
    return int(candidate_ids[sampled]) if candidate_ids is not None else sampled


@torch.inference_mode()
def generate_reasoning(
    transformer, prefix_ids: list[int], *, seed: int, max_new_tokens: int = MAX_NEW_TOKENS
) -> list[int]:
    """Sample the reasoning ids that precede the image block.

    Stops before the model's own ``<sop>H W<eop>``: the requested size, not the
    model's, decides the canvas.
    """
    device = transformer.model.embed_tokens.weight.device
    cache = transformer.new_text_cache(len(prefix_ids) + max_new_tokens)
    generator = torch.Generator().manual_seed(seed)
    logits = transformer.text_logits(torch.tensor(prefix_ids, device=device), cache)
    generated: list[int] = []
    while len(generated) < max_new_tokens:
        token = sample_next_token(logits, generator)
        if token == SOP_ID:
            break
        generated.append(token)
        if token in (BOI_ID, USER_ID):
            break
        logits = transformer.text_logits(torch.tensor([token], device=device), cache)
    while generated and generated[-1] in (BOI_ID, EOP_ID, USER_ID):
        generated.pop()
    return generated


def split_reasoning(reasoning_ids: list[int], tokenizer) -> tuple[str, str]:
    """Decoded chain of thought and final caption; the caption is empty if the chain never closed."""
    end = reasoning_ids.index(THINK_END_ID) if THINK_END_ID in reasoning_ids else len(reasoning_ids)
    start = 1 if reasoning_ids[:1] == [THINK_START_ID] else 0
    thought = tokenizer.decode(reasoning_ids[start:end], skip_special_tokens=False).strip()
    caption = tokenizer.decode(reasoning_ids[end + 1 :], skip_special_tokens=False).strip()
    return thought, caption
