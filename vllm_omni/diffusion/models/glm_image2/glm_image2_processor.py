# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native GLM-Image2 prompt/reference-image processor.

Ports the inference path of the transformers ``GlmImage2Processor`` /
``GlmImage2ImageProcessor`` without depending on a transformers release that
ships them: the tokenizer is read directly from ``tokenizer.json`` with the
``tokenizers`` library, the prompt layout comes from the checkpoint's chat
template, and reference images are resized with Pillow LANCZOS.

One call prepares one target image: the packed prompt
``<|user|><|gen_request|>[references]caption<|assistant|>...<|vae|>*N`` with
3D (t, h, w) mRoPE positions, the reference ViT patches (in spatial-merge
window order) and VAE images, and the reference-then-target diffusion grids.
The target VAE block is always the last run of the sequence.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import PIL.Image
import regex as re
import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

# Token ids the checkpoint was trained with; the tokenizer must preserve them.
_EXPECTED_SPECIAL_TOKENS = {"<|image|>": 154854, "<|vae|>": 154856, "<|gen_request|>": 154857}

TEXT_TOKEN, VIT_TOKEN, VAE_TOKEN = 0, 1, 2

THINKING_TEMPLATE = (
    "Generate an image based on the user's request. Before generating the image, "
    "carefully analyze the request, plan the composition, visual elements, style, "
    "and constraints, and resolve any ambiguities or inconsistencies. Explicitly "
    "present your analysis and generation plan before producing the final image:"
)


def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 32 * 32, max_pixels: int = 2048 * 2048
) -> tuple[int, int]:
    """Align both sides to ``factor`` and keep the area within ``[min_pixels, max_pixels]``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def find_quoted_spans(prompt: str, structured: bool = False) -> list[tuple[int, int]] | None:
    """Outer quoted spans, excluding contractions and possessive apostrophes."""
    pairs = (("", ""),) if structured else (("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”"))
    openers = {start for start, _ in pairs}
    closers = {end: start for start, end in pairs}
    letters = "a-zA-ZÀ-ɏ̀-ͯͰ-ϿЀ-ӿ"
    word_character = re.compile(f"[{letters}0-9]")
    apostrophes = re.compile(rf"(?<=[{letters}])['’](?=[{letters}])|['’](?=\d{{2}}s\b)")
    skipped = set() if structured else {match.start() for match in apostrophes.finditer(prompt)}
    ambiguous = {"'", '"'}
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    for match in re.finditer(f"[{re.escape(''.join(sorted(openers | set(closers))))}]", prompt):
        index, character = match.start(), match.group()
        if index in skipped:
            continue
        previous = prompt[index - 1] if index else ""
        following = prompt[index + 1] if index + 1 < len(prompt) else ""
        matched = None
        if character in closers and not (character in ambiguous and (not previous or previous.isspace())):
            matched = next((i for i in range(len(stack) - 1, -1, -1) if stack[i][0] == closers[character]), None)
        if matched is not None:
            spans.append((stack[matched][1], index))
            del stack[matched:]
        elif character in openers and (
            character not in ambiguous
            or (not word_character.match(previous or " ") and following and not following.isspace())
        ):
            stack.append((character, index))
        elif structured or not (character in ambiguous or word_character.match(previous or " ")):
            return None
    if any(character not in ambiguous for character, _ in stack):
        return None
    outer: list[tuple[int, int]] = []
    previous_end = -1
    for start, end in sorted(spans):
        if start > previous_end:
            if structured or end > start + 1:
                outer.append((start, end))
            previous_end = end
    return outer


def prepare_caption(caption: str, prompt_format: str) -> str:
    if prompt_format not in ("plain", "calico_structure"):
        raise ValueError("prompt_format must be 'plain' or 'calico_structure'.")
    if prompt_format == "plain":
        return caption
    parsed = json.loads(caption)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("text_render"), list):
        raise ValueError("A structured caption must be a JSON object with a text_render list.")
    parsed.pop("type", None)
    for item in parsed["text_render"]:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            clean = item["text"].replace("", "").replace("", "")
            item["text"] = f"{clean}"
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


@dataclass
class GlmImage2Inputs:
    """Packed single-sample model inputs (all tensors on CPU)."""

    input_ids: torch.Tensor  # [L] int64
    mm_token_type_ids: torch.Tensor  # [L] int64: 0 text, 1 reference ViT, 2 VAE
    position_ids: torch.Tensor  # [3, L] int64
    pixel_values: torch.Tensor  # [sum(Hv*Wv), 3*P*P] float32, patches in spatial-merge window order
    reference_images: list[torch.Tensor]  # normalized [3, H, W] float32 per reference, for the VAE
    image_grid_thw: torch.Tensor  # [num_refs, 3] int64 (ViT patch grids)
    diff_grid_thw: torch.Tensor  # [num_refs + 1, 3] int64 (VAE token grids, target last)
    is_ref_vae: torch.Tensor  # [num_refs + 1] bool

    @property
    def target_start(self) -> int:
        """Index of the first target-image token; everything before it is the reusable prefix."""
        return int(self.input_ids.shape[0] - self.diff_grid_thw[-1].prod().item())

    def image_block_spans(self) -> list[tuple[int, int]]:
        """``[start, end)`` of every contiguous ViT/VAE run (bidirectional attention blocks)."""
        types = self.mm_token_type_ids.tolist()
        spans, cursor = [], 0
        for modality, group in itertools.groupby(types):
            count = len(list(group))
            if modality != TEXT_TOKEN:
                spans.append((cursor, cursor + count))
            cursor += count
        return spans


class GlmImage2Processor:
    def __init__(self, processor_dir: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(processor_dir, "tokenizer.json"))
        with open(os.path.join(processor_dir, "processor_config.json")) as f:
            processor_config = json.load(f)
        image_config = processor_config.get("image_processor")
        if image_config is None:
            with open(os.path.join(processor_dir, "preprocessor_config.json")) as f:
                image_config = json.load(f)
        with open(os.path.join(processor_dir, "chat_template.jinja")) as f:
            environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            self.chat_template = environment.from_string(f.read())
        size = image_config.get("size", {})
        self.patch_size = int(image_config.get("patch_size", 16))
        self.merge_size = int(image_config.get("merge_size", 2))
        self.min_pixels = int(size.get("shortest_edge", 32 * 32))
        self.max_pixels = int(size.get("longest_edge", 2048 * 2048))
        self.image_mean = np.asarray(image_config.get("image_mean", [0.5] * 3), dtype=np.float32).reshape(-1, 1, 1)
        self.image_std = np.asarray(image_config.get("image_std", [0.5] * 3), dtype=np.float32).reshape(-1, 1, 1)
        self.vae_compression_ratio = int(processor_config.get("vae_compression_ratio", 16))
        self.use_shared_image_positions = bool(processor_config.get("use_shared_image_positions", False))

        for token, token_id in _EXPECTED_SPECIAL_TOKENS.items():
            if self.tokenizer.token_to_id(token) != token_id:
                raise ValueError("The tokenizer must preserve the checkpoint's image, VAE and generation-request ids.")
        self.image_token_id = _EXPECTED_SPECIAL_TOKENS["<|image|>"]
        self.vae_token_id = _EXPECTED_SPECIAL_TOKENS["<|vae|>"]
        self.gen_request_token_id = _EXPECTED_SPECIAL_TOKENS["<|gen_request|>"]
        self.user_token_id = self.tokenizer.token_to_id("<|user|>")
        self.assistant_token_id = self.tokenizer.token_to_id("<|assistant|>")
        if (self.patch_size * self.merge_size) % self.vae_compression_ratio:
            raise ValueError("Resize alignment must be divisible by the VAE compression.")

    def get_resize_output_image_size(self, height: int, width: int) -> tuple[int, int]:
        return smart_resize(height, width, self.patch_size * self.merge_size, self.min_pixels, self.max_pixels)

    # ---------------------------------------------------------------- images

    def _preprocess_image(self, image: PIL.Image.Image) -> tuple[np.ndarray, np.ndarray, list[int]]:
        image = image.convert("RGB")
        height, width = self.get_resize_output_image_size(image.height, image.width)
        if (height, width) != (image.height, image.width):
            image = image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) / np.float32(255)
        array = (array - self.image_mean) / self.image_std
        patch, merge = self.patch_size, self.merge_size
        grid_h, grid_w = height // patch, width // patch
        patches = array.reshape(3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6).reshape(grid_h * grid_w, 3 * patch**2)
        return patches, array, [1, grid_h, grid_w]

    # ------------------------------------------------------------------ text

    @staticmethod
    def _split_quoted(text: str, structured: bool) -> list[str]:
        """Quoted spans are tokenized one character at a time (text rendering)."""
        pieces, cursor = [], 0
        for start, end in find_quoted_spans(text, structured) or []:
            quoted = json.loads('"' + text[start + 1 : end] + '"') if structured else text[start : end + 1]
            pieces += [text[cursor:start], *quoted]
            cursor = end + 1
        return pieces + [text[cursor:]]

    def _expand_prompt(
        self, prompt: str, image_grids: list[list[int]], diff_grids: list[list[int]], structured: bool
    ) -> tuple[list[str], dict[int, tuple[int, ...]]]:
        """Split the rendered prompt into tokenizer words; each placeholder word expands to its image run."""
        image_grids, diff_grids = iter(image_grids), iter(diff_grids)
        pieces: list[str] = []
        blocks: dict[int, tuple[int, ...]] = {}
        reference_grid = None
        parts = re.split(r"(<\|vae\|>|<\|image\|>)", prompt)
        for index, part in enumerate(parts):
            temporal_offset = 0
            if part == "<|vae|>":
                _, height, width = next(diff_grids)
                spatial_height, spatial_width = height, width
                if self.use_shared_image_positions and parts[index + 1 : index + 3] == ["", "<|image|>"]:
                    # The reference ViT run reuses this VAE run's spatial frame, one temporal step later.
                    reference_grid, advance = (height, width), 0
                else:
                    advance = max(height, width)
            elif part == "<|image|>":
                _, height, width = next(image_grids)
                advance = max(height, width)
                height, width = height // self.merge_size, width // self.merge_size
                spatial_height, spatial_width = height, width
                if reference_grid is not None:
                    temporal_offset, (spatial_height, spatial_width) = 1, reference_grid
                    advance, reference_grid = max(reference_grid), None
            else:
                pieces.extend(self._split_quoted(part, structured))
                continue
            blocks[len(pieces)] = (height, width, advance, temporal_offset, spatial_height, spatial_width)
            pieces.append(part * (height * width))
        return pieces, blocks

    @staticmethod
    def _position_ids(word_ids: np.ndarray, num_words: int, blocks: dict[int, tuple[int, ...]]) -> np.ndarray:
        positions = np.zeros((3, word_ids.shape[0]), dtype=np.int64)
        bounds = np.searchsorted(word_ids, np.arange(num_words + 1))
        start = 0
        for word, (first, last) in enumerate(zip(bounds[:-1], bounds[1:])):
            count = last - first
            if word not in blocks:
                positions[:, first:last] = start + np.arange(count)
                start += count
                continue
            height, width, advance, temporal_offset, spatial_height, spatial_width = blocks[word]
            if count != height * width:
                raise ValueError("Image placeholder runs do not match the supplied image grids.")
            rows, cols = np.divmod(np.arange(count), width)
            positions[0, first:last] = start + temporal_offset
            positions[1, first:last] = start + rows * spatial_height // height
            positions[2, first:last] = start + cols * spatial_width // width
            start += advance
        return positions

    def __call__(
        self,
        caption: str,
        reference_images: list[PIL.Image.Image] | None,
        height: int,
        width: int,
        prompt_format: str = "plain",
    ) -> GlmImage2Inputs:
        if "<|image|>" in caption or "<|vae|>" in caption:
            raise ValueError("The caption must not contain literal image or VAE placeholder tokens.")
        patches, images, image_grids = [], [], []
        for image in reference_images or []:
            patch_array, array, grid = self._preprocess_image(image)
            patches.append(patch_array)
            images.append(torch.from_numpy(array))
            image_grids.append(grid)
        pixel_values = np.concatenate(patches) if patches else np.empty((0, 3 * self.patch_size**2), dtype=np.float32)

        patch, ratio = self.patch_size, self.vae_compression_ratio
        diff_grids = [[1, h * patch // ratio, w * patch // ratio] for _, h, w in image_grids]
        target_h, target_w = self.get_resize_output_image_size(height, width)
        diff_grids.append([1, target_h // ratio, target_w // ratio])

        content = [{"type": "image", "height": grid[1], "width": grid[2]} for grid in diff_grids[:-1]]
        content.append({"type": "text", "text": prepare_caption(caption, prompt_format)})
        prompt = self.chat_template.render(
            messages=[{"role": "user", "content": content}],
            target_height=diff_grids[-1][1],
            target_width=diff_grids[-1][2],
        )
        pieces, blocks = self._expand_prompt(prompt, image_grids, diff_grids, prompt_format == "calico_structure")
        encoding = self.tokenizer.encode(pieces, is_pretokenized=True, add_special_tokens=False)
        ids = np.asarray(encoding.ids, dtype=np.int64)
        modalities = np.zeros_like(ids)
        modalities[ids == self.image_token_id] = VIT_TOKEN
        modalities[ids == self.vae_token_id] = VAE_TOKEN
        positions = self._position_ids(np.asarray(encoding.word_ids, dtype=np.int64), len(pieces), blocks)
        return GlmImage2Inputs(
            input_ids=torch.from_numpy(ids),
            mm_token_type_ids=torch.from_numpy(modalities),
            position_ids=torch.from_numpy(positions),
            pixel_values=torch.from_numpy(pixel_values),
            reference_images=images,
            image_grid_thw=torch.tensor(image_grids, dtype=torch.int64).reshape(-1, 3),
            diff_grid_thw=torch.tensor(diff_grids, dtype=torch.int64),
            is_ref_vae=torch.tensor([True] * len(image_grids) + [False]),
        )

    def thinking_prefix(self, prompt: str, height: int, width: int) -> list[int]:
        """Token ids that ask the model to reason before generating one ``height`` x ``width`` image."""
        target_h, target_w = self.get_resize_output_image_size(height, width)
        request = f"{THINKING_TEMPLATE} {prompt} h:{target_h} w:{target_w}"
        pieces = self._split_quoted(request, False)
        tokens = self.tokenizer.encode(pieces, is_pretokenized=True, add_special_tokens=False).ids
        return [self.user_token_id, self.gen_request_token_id, *tokens, self.assistant_token_id]

    def thinking_inputs(self, token_ids: list[int], height: int, width: int) -> GlmImage2Inputs:
        """Pack ``thinking_prefix`` plus the sampled reasoning ids with the target image block."""
        target_h, target_w = self.get_resize_output_image_size(height, width)
        grid_h, grid_w = target_h // self.vae_compression_ratio, target_w // self.vae_compression_ratio
        shape = self.tokenizer.encode(f"<sop>{grid_h} {grid_w}<eop><|begin_of_image|>", add_special_tokens=False)
        text = [*token_ids, *shape.ids]
        start, count = len(text), grid_h * grid_w
        rows, cols = np.divmod(np.arange(count), grid_w)
        positions = np.concatenate(
            (
                np.broadcast_to(np.arange(start), (3, start)),
                np.stack((np.full(count, start), start + rows, start + cols)),
            ),
            axis=1,
        )
        modalities = np.zeros(start + count, dtype=np.int64)
        modalities[start:] = VAE_TOKEN
        return GlmImage2Inputs(
            input_ids=torch.tensor(text + [self.vae_token_id] * count, dtype=torch.int64),
            mm_token_type_ids=torch.from_numpy(modalities),
            position_ids=torch.from_numpy(positions),
            pixel_values=torch.empty((0, 3 * self.patch_size**2), dtype=torch.float32),
            reference_images=[],
            image_grid_thw=torch.empty((0, 3), dtype=torch.int64),
            diff_grid_thw=torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64),
            is_ref_vae=torch.tensor([False]),
        )
