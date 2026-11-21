#!/usr/bin/env python3
"""Prepare vLLM-native stage assets from the full GLM-Image HY/MoE checkpoint.

This tool does not copy model implementation code. It only derives the
configuration, tokenizer files, and AR weight index needed by the standard
vLLM Glm4MoeLiteForCausalLM model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenization_glm4moe.py",
    "chat_template.jinja",
    "scheduler_config.json",
    "config.json",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_ar_config(model_config: dict, tokenizer_config: dict) -> dict:
    text_config = dict(model_config["text_config"])
    text_config.update(
        {
            "_name_or_path": "",
            "architectures": ["GlmImageMoeTransfusionForCausalLM"],
            "model_type": "glm4_moe_lite",
            "torch_dtype": "bfloat16",
            "dtype": "bfloat16",
            "tie_word_embeddings": bool(model_config.get("tie_word_embeddings", False)),
            "eos_token_id": tokenizer_config.get("eos_token_id", text_config.get("eos_token_id")),
            "pad_token_id": tokenizer_config.get("pad_token_id", tokenizer_config.get("eos_token_id")),
            "bos_token_id": tokenizer_config.get("bos_token_id"),
        }
    )
    return text_config


def filtered_ar_index(index: dict) -> dict:
    weight_map = index.get("weight_map", {})
    ar_weight_map = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith("model.") or name == "lm_head.weight"
    }
    if not ar_weight_map:
        raise RuntimeError("no AR weights found in checkpoint index")
    return {
        "metadata": {"total_size": sum(0 for _ in ar_weight_map)},
        "weight_map": ar_weight_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--effective-config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_config = load_json(args.effective_config)
    tokenizer_config = load_json(args.tokenizer / "tokenizer_config.json")
    index = load_json(args.checkpoint / "model.safetensors.index.json")

    ar_dir = args.output / "ar"
    ar_dir.mkdir(parents=True, exist_ok=True)
    write_json(ar_dir / "config.json", build_ar_config(model_config, tokenizer_config))

    for filename in TOKENIZER_FILES:
        source = args.tokenizer / filename
        if source.is_file():
            link_or_copy(source, ar_dir / filename)

    for shard_name in sorted(set(index.get("weight_map", {}).values())):
        link_or_copy(args.checkpoint / shard_name, ar_dir / shard_name)

    write_json(
        ar_dir / "model.safetensors.index.json",
        filtered_ar_index(index),
    )
    print(f"prepared AR stage at {ar_dir}")


if __name__ == "__main__":
    main()
