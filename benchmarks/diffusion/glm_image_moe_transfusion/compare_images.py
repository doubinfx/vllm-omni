# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare paired GLM-Image outputs with PSNR, SSIM and image statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    channels = reference.shape[1]
    window = torch.ones((channels, 1, 11, 11), dtype=torch.float32) / 121.0
    mean_reference = F.conv2d(reference, window, groups=channels)
    mean_candidate = F.conv2d(candidate, window, groups=channels)
    variance_reference = F.conv2d(reference.square(), window, groups=channels) - mean_reference.square()
    variance_candidate = F.conv2d(candidate.square(), window, groups=channels) - mean_candidate.square()
    covariance = F.conv2d(reference * candidate, window, groups=channels) - mean_reference * mean_candidate
    c1 = 6.5025
    c2 = 58.5225
    score = ((2 * mean_reference * mean_candidate + c1) * (2 * covariance + c2)) / (
        (mean_reference.square() + mean_candidate.square() + c1) * (variance_reference + variance_candidate + c2)
    )
    return float(score.mean())


def _case(name: str, reference_path: Path, candidate_path: Path) -> dict[str, object]:
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError(f"{name}: shape mismatch {reference.shape} != {candidate.shape}")
    difference = (reference - candidate).abs()
    mse = float(difference.square().mean())
    return {
        "name": name,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "shape": list(reference.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "mean_abs_error": float(difference.mean()),
        "max_abs_error": float(difference.max()),
        "psnr_db": math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
        "ssim": _ssim(reference, candidate),
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "reference_std": float(reference.std()),
        "candidate_std": float(candidate.std()),
        "candidate_non_constant": bool(candidate.std() > 1.0),
        "candidate_non_black": bool(candidate.mean() > 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("NAME", "REFERENCE", "CANDIDATE"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [_case(name, Path(reference), Path(candidate)) for name, reference, candidate in args.case]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
