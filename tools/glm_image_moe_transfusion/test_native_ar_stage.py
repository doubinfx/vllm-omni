#!/usr/bin/env python3
"""Smoke test the native GLM-Image AR stage with the training prefix."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


USER_TOKEN_ID = 154827
ASSISTANT_TOKEN_ID = 154828
GEN_IMAGE_REQUEST_TOKEN_ID = 154857
BOI_TOKEN_ID = 154830

THINKING_TEMPLATE = (
    "Generate an image based on the user's request. Before generating the image, "
    "carefully analyze the request, plan the composition, visual elements, style, "
    "and constraints, and resolve any ambiguities or inconsistencies. Explicitly "
    "present your analysis and generation plan before producing the final image:"
)


def main() -> None:
    model_dir = Path("/mnt/share/d00911077/glm-image/native_stages_v2/ar")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    prompt = "一只在雪地里打滚的柴犬，浅景深"
    body = f"{THINKING_TEMPLATE} {prompt} h:512 w:512"
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    prefix_ids = [
        USER_TOKEN_ID,
        GEN_IMAGE_REQUEST_TOKEN_ID,
        *body_ids,
        ASSISTANT_TOKEN_ID,
    ]
    print("prefix_ids:", prefix_ids)
    print("prefix_text:", tokenizer.decode(prefix_ids, skip_special_tokens=False))

    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=1,
        enforce_eager=True,
        gpu_memory_utilization=0.55,
        disable_log_stats=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prefix_ids}],
        SamplingParams(
            max_tokens=32,
            temperature=0.0,
            top_p=0.9,
            top_k=50,
            stop_token_ids=[BOI_TOKEN_ID, USER_TOKEN_ID],
        ),
    )
    for output in outputs:
        print("output:", output.outputs[0].text)
        print("token_ids:", output.outputs[0].token_ids)


if __name__ == "__main__":
    main()
