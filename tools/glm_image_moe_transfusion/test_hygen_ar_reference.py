#!/usr/bin/env python3
"""Generate one Hygen AR trace for native-stage parity checks."""

from __future__ import annotations

import os

from hygen import Inferencer
from hygen import generation_hy


def main() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    devices = list(range(len(visible.split(",")))) if visible else [4, 5, 6, 7]
    inferencer = Inferencer.from_spec(
        "dmd2_lyzsft10000_s2000",
        devices=devices,
        ckpt="/mnt/share/d00911077/glm-image/g/dmd2_lyzsft10000_s2000",
    )
    prompt = "一只在雪地里打滚的柴犬，浅景深"
    prefix, body = inferencer.transform.ar_prefix_ids(
        prompt,
        512,
        512,
        mode="ttg",
        with_res=True,
    )
    config = inferencer._ar_config(1234, {"max_new_tokens": 32, "temperature": 0.0})
    raw, info = inferencer.decoder.generate(
        prefix,
        config,
        tokenizer=inferencer.rt.tokenizer,
    )
    trimmed = generation_hy.trim_generated(raw)
    print("body:", body)
    print("prefix_ids:", prefix.tolist())
    print("raw_ids:", raw)
    print("trimmed_ids:", trimmed)
    print("text:", inferencer.rt.tokenizer.decode(trimmed, skip_special_tokens=False))
    print("info:", info)


if __name__ == "__main__":
    main()
