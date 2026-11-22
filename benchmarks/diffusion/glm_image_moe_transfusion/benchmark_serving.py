# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark GLM-Image T2I serving across direct and thinking modes."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def _request(
    *,
    base_url: str,
    prompt: str,
    size: int,
    steps: int,
    seed: int,
    max_tokens: int | None,
) -> tuple[float, dict[str, object], Image.Image]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "size": f"{size}x{size}",
        "num_inference_steps": steps,
        "seed": seed,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/images/generations",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    latency = time.perf_counter() - started
    image_bytes = base64.b64decode(data["data"][0]["b64_json"])
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        if image.size != (size, size) or image.format != "PNG":
            raise RuntimeError(f"unexpected image shape/format: {image.size} {image.format}")
        result_image = image.copy()
    return latency, data, result_image


def _image_stats(image: Image.Image) -> dict[str, float | list[int] | str | bool]:
    array = np.asarray(image, dtype=np.float32)
    return {
        "size": list(image.size),
        "mode": image.mode,
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "non_constant": bool(array.std() > 1.0),
        "non_black": bool(array.mean() > 1.0),
    }


def _latency_stats(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_seconds": float(values.mean()),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "std_seconds": float(values.std()),
    }


def _stage_metrics(response: dict[str, object]) -> dict[str, object]:
    metrics = response.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    selected: dict[str, object] = {}
    if "peak_memory_mb" in metrics:
        selected["peak_memory_mb"] = metrics["peak_memory_mb"]
    stages = metrics.get("stage_durations")
    if isinstance(stages, dict):
        selected["stage_durations"] = stages
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("direct", "think"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = args.max_tokens if args.mode == "think" else None
    results: dict[str, object] = {
        "mode": args.mode,
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cases": [],
    }

    for size in args.sizes:
        for warmup in range(args.warmups):
            _request(
                base_url=args.base_url,
                prompt=args.prompt,
                size=size,
                steps=args.steps,
                seed=args.seed,
                max_tokens=max_tokens,
            )
            print(f"{args.mode} {size}x{size} warmup {warmup + 1}/{args.warmups}", flush=True)

        latencies: list[float] = []
        responses: list[dict[str, float | list[int] | str | bool]] = []
        for repeat in range(args.repeats):
            latency, response, image = _request(
                base_url=args.base_url,
                prompt=args.prompt,
                size=size,
                steps=args.steps,
                seed=args.seed,
                max_tokens=max_tokens,
            )
            latencies.append(latency)
            image_path = args.output_dir / f"{args.mode}_{size}_{repeat}.png"
            image.save(image_path)
            responses.append(
                {
                    "latency_seconds": latency,
                    "image_path": str(image_path),
                    "image_stats": _image_stats(image),
                    "metrics": _stage_metrics(response),
                }
            )
            print(f"{args.mode} {size}x{size} repeat {repeat + 1}/{args.repeats}: {latency:.3f} s", flush=True)

        cases = results["cases"]
        assert isinstance(cases, list)
        cases.append(
            {
                "size": size,
                "latency": _latency_stats(latencies),
                "throughput_images_per_second": 1.0 / float(np.mean(latencies)),
                "responses": responses,
            }
        )
        (args.output_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for case in results["cases"]:
        if not math.isfinite(case["latency"]["mean_seconds"]):
            raise RuntimeError(f"non-finite latency: {case}")


if __name__ == "__main__":
    main()
