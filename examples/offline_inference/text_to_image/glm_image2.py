# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Example script for GLM-Image2 text-to-image and reference-guided generation.

GLM-Image2 is a single-stage diffusion pipeline: a shared MLA/MoE Transfusion
decoder (native vLLM-Omni implementation) denoises packed VAE latents,
optionally conditioned on one or more reference images.

Usage (text-to-image):
    python glm_image2.py \
        --model zai-org/GLM-Image-2 \
        --prompt "A photorealistic mountain landscape at sunset" \
        --output glm_image2_output.png

Usage (reference-guided generation / editing):
    python glm_image2.py \
        --model zai-org/GLM-Image-2 \
        --image input.png \
        --prompt "Convert this image to watercolor style" \
        --output glm_image2_edit.png

Usage (structured caption):
    python glm_image2.py \
        --model zai-org/GLM-Image-2 \
        --prompt-format calico_structure \
        --prompt '{"type": "image_generation", "objects": [...], "text_render": [...]}' \
        --output glm_image2_structured.png

Usage (thinking: plan the image as text first, text-to-image with guidance 1):
    python glm_image2.py \
        --model zai-org/GLM-Image-2 \
        --prompt "An owl balancing a globe on its head" \
        --enable-thinking \
        --output glm_image2_thinking.png

Multi-GPU: the shared ``text_to_image.py`` script exposes
``--cfg-parallel-size 2`` (one CFG branch per GPU) and
``--tensor-parallel-size`` (20 attention heads: 1, 2, 4 or 5).
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with GLM-Image2 (T2I + optional reference images).")
    parser.add_argument(
        "--model",
        default="zai-org/GLM-Image-2",
        help="Model name or local path to a diffusers-format GLM-Image2 checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="a cup of coffee on the table",
        help="Text prompt for image generation (a JSON structured caption with --prompt-format calico_structure).",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=None,
        help="Optional reference image path(s) (PNG, JPG, etc.) for reference-guided generation.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="plain",
        choices=["plain", "calico_structure"],
        help="Prompt format interpreted by the GLM-Image2 processor.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (1.0 disables CFG). Defaults to the checkpoint's generation config.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps. Defaults to the checkpoint's generation config.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["ode", "renoise"],
        help="Euler ODE sampler or the DMD2 renoise sampler. Defaults to the checkpoint's generation config.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Sample a chain of thought and a final caption before the image (text-to-image, guidance 1).",
    )
    parser.add_argument("--height", type=int, default=1024, help="Requested output image height in pixels.")
    parser.add_argument("--width", type=int, default=1024, help="Requested output image width in pixels.")
    parser.add_argument(
        "--output",
        type=str,
        default="glm_image2_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        help="Disable torch.compile and force eager execution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    reference_images = None
    if args.image:
        reference_images = []
        for image_path in args.image:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Reference image not found: {image_path}")
            reference_images.append(Image.open(image_path).convert("RGB"))

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    omni_kwargs = {"model": args.model}
    if args.enforce_eager is not None:
        omni_kwargs["enforce_eager"] = args.enforce_eager
    omni = Omni(**omni_kwargs)

    default = "checkpoint default"
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps or default}")
    print(f"  Guidance scale: {default if args.guidance_scale is None else args.guidance_scale}")
    print(f"  Sampler: {args.sampler or default}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"  Prompt format: {args.prompt_format}")
    if reference_images:
        print(f"  Reference images: {len(reference_images)}")
    print(f"{'=' * 60}\n")

    if reference_images:
        prompt_dict = {
            "prompt": args.prompt,
            "multi_modal_data": {"image": reference_images},
        }
    else:
        prompt_dict = {
            "prompt": args.prompt,
            "modalities": ["image"],
        }

    diffusion_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    diffusion_params.extra_args["prompt_format"] = args.prompt_format
    if args.sampler is not None:
        diffusion_params.extra_args["sampler"] = args.sampler
    if args.enable_thinking:
        diffusion_params.extra_args["enable_thinking"] = True

    generation_start = time.perf_counter()
    outputs = omni.generate(prompt_dict, sampling_params_list=[diffusion_params])
    generation_time = time.perf_counter() - generation_start
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    if not outputs or len(outputs) == 0:
        raise ValueError("No output generated from omni.generate()")

    images = getattr(outputs[0], "images", None)
    if not images:
        raise ValueError("No images found in request_output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"Saved generated image to {output_path}")
    multimodal = getattr(outputs[0], "multimodal_output", None)
    text = ((multimodal.get("metadata") or {}).get("text") or {}) if isinstance(multimodal, dict) else {}
    if text.get("text_output"):
        print(f"Final caption: {text['text_output']}")


if __name__ == "__main__":
    main()
