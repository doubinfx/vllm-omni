"""Native diffusion inference for the GLM-Image HY/MoE checkpoint.

The text backbone is Transformers' standard GLM4-MoE-lite implementation.
Only the flow-matching input/output projections and time-conditioning adapters
are GLM-Image specific.
"""

from __future__ import annotations

import json
import logging
import math
import re
from functools import wraps
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
)

from vllm_omni.diffusion.models.glm_image_moe_transfusion.a2_kernels import (
    a2_attention_forward,
    a2_moe_forward,
    use_a2_fused_attention,
    use_a2_fused_moe,
)
from vllm_omni.model_executor.models.glm_image_moe_transfusion.ascend_compat import (
    configure_a2_diffusion,
    standard_conv_internal_format,
)
from vllm_omni.model_executor.models.glm_image_moe_transfusion.prompt import (
    GlmImageTokenLayout,
    build_diffusion_layout,
)

logger = logging.getLogger(__name__)


def _remap_flux2_vae_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate the checkpoint's legacy Flux VAE keys to current Diffusers."""
    remapped: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        original = name
        name = name.replace(".nin_shortcut.", ".conv_shortcut.")
        name = re.sub(r"^(encoder|decoder)\.mid\.block_1\.", r"\1.mid_block.resnets.0.", name)
        name = re.sub(r"^(encoder|decoder)\.mid\.block_2\.", r"\1.mid_block.resnets.1.", name)
        name = name.replace(".mid.attn_1.norm.", ".mid_block.attentions.0.group_norm.")
        name = name.replace(".mid.attn_1.q.", ".mid_block.attentions.0.to_q.")
        name = name.replace(".mid.attn_1.k.", ".mid_block.attentions.0.to_k.")
        name = name.replace(".mid.attn_1.v.", ".mid_block.attentions.0.to_v.")
        name = name.replace(".mid.attn_1.proj_out.", ".mid_block.attentions.0.to_out.0.")
        name = re.sub(
            r"^encoder\.down\.(\d+)\.block\.(\d+)\.",
            r"encoder.down_blocks.\1.resnets.\2.",
            name,
        )
        name = re.sub(
            r"^encoder\.down\.(\d+)\.downsample\.conv\.",
            r"encoder.down_blocks.\1.downsamplers.0.conv.",
            name,
        )
        name = name.replace("encoder.norm_out.", "encoder.conv_norm_out.")
        name = name.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        if name.startswith("encoder.quant_conv."):
            name = name.removeprefix("encoder.")
        if name.startswith("decoder.post_quant_conv."):
            name = name.removeprefix("decoder.")

        match = re.match(r"^decoder\.up\.(\d+)\.(block\.\d+|upsample\.conv)\.", name)
        if match:
            old_level = int(match.group(1))
            new_level = 3 - old_level
            suffix = match.group(2)
            if suffix.startswith("block."):
                block = suffix.split(".", 1)[1]
                replacement = f"decoder.up_blocks.{new_level}.resnets.{block}."
            else:
                replacement = f"decoder.up_blocks.{new_level}.upsamplers.0.conv."
            name = name[: match.start()] + replacement + name[match.end() :]

        if name in remapped:
            raise RuntimeError(f"duplicate Flux2 VAE key after remap: {original}")
        if name.endswith((".to_q.weight", ".to_k.weight", ".to_v.weight", ".to_out.0.weight")):
            tensor = tensor.squeeze(-1).squeeze(-1)
        remapped[name] = tensor
    return remapped


def _rms_norm_no_affine(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


def _adaln_apply(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normed = _rms_norm_no_affine(x, eps).to(x.dtype)
    return torch.addcmul(shift, normed, 1 + scale)


class _TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int = 2048, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def _mrope_axis_pattern(section: list[int], layout: str) -> list[int]:
    """Build the frequency-to-position-axis map used by GLM-Image."""
    if layout == "block":
        return [axis for axis, width in enumerate(section) for _ in range(width)] * 2

    t_count, h_count, w_count = section
    total = t_count + h_count + w_count
    axis_per_pair = [0] * total
    high_count = (h_count - t_count) + (w_count - t_count)
    h_left, w_left = h_count - t_count, w_count - t_count
    for index in range(high_count):
        if index % 2 == 0 and h_left:
            axis_per_pair[index] = 1
            h_left -= 1
        elif index % 2 == 1 and w_left:
            axis_per_pair[index] = 2
            w_left -= 1
        elif h_left:
            axis_per_pair[index] = 1
            h_left -= 1
        else:
            axis_per_pair[index] = 2
            w_left -= 1
    for index in range(3 * t_count):
        axis_per_pair[high_count + index] = (0, 1, 2)[index % 3]

    if layout == "hw_thw":
        pairs = axis_per_pair
    elif layout == "qwen3":
        pairs = axis_per_pair[high_count:] + axis_per_pair[:high_count]
    else:
        raise ValueError(f"unsupported GLM-Image MRoPE layout: {layout}")
    return pairs + pairs


def _select_mrope(
    position_ids: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    inv_freq: torch.Tensor,
    sections: list[int],
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one T/H/W axis per rotary dimension and return [1,L,D] tables."""
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.to(position_ids.device)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    text_axis = _mrope_axis_pattern(sections, "block")
    image_axis = _mrope_axis_pattern(sections, layout)
    axis = torch.where(
        image_mask.to(cos.device)[:, None],
        torch.tensor(image_axis, dtype=torch.long, device=cos.device)[None, :],
        torch.tensor(text_axis, dtype=torch.long, device=cos.device)[None, :],
    )
    # Ascend's gather on a permuted [3,L,D] table can collapse every frequency
    # to the first column. Select the three axes explicitly instead.
    cos = cos.permute(1, 0, 2)
    sin = sin.permute(1, 0, 2)
    cos = torch.where(
        axis == 0,
        cos[:, 0, :],
        torch.where(axis == 1, cos[:, 1, :], cos[:, 2, :]),
    )
    sin = torch.where(
        axis == 0,
        sin[:, 0, :],
        torch.where(axis == 1, sin[:, 1, :], sin[:, 2, :]),
    )
    return cos.unsqueeze(0).contiguous(), sin.unsqueeze(0).contiguous()


def _packed_attention_mask_batch(
    text_lengths: list[int],
    image_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build one padded mask while keeping every request's text isolated."""
    if not text_lengths or any(text_len <= 0 for text_len in text_lengths):
        raise ValueError(f"text lengths must be positive, got {text_lengths}")
    if image_len <= 0:
        raise ValueError(f"image length must be positive, got {image_len}")

    batch_size = len(text_lengths)
    padded_text_len = max(text_lengths)
    length = padded_text_len + image_len
    allowed = torch.zeros((batch_size, length, length), dtype=torch.bool, device=device)
    for batch_index, text_len in enumerate(text_lengths):
        text_indices = torch.arange(text_len, device=device)
        allowed[batch_index, :text_len, :text_len] = text_indices[:, None] >= text_indices[None, :]
        if text_len < padded_text_len:
            padding = torch.arange(text_len, padded_text_len, device=device)
            allowed[batch_index, padding, padding] = True
        allowed[batch_index, padded_text_len:, :text_len] = True
        allowed[batch_index, padded_text_len:, padded_text_len:] = True

    mask = torch.full(
        (batch_size, 1, length, length),
        torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16,
        device=device,
    )
    mask.masked_fill_(allowed[:, None], 0)
    return mask


def _packed_attention_mask(text_len: int, image_len: int, device: torch.device) -> torch.Tensor:
    """Text is causal; generated latent rows see text and all latent rows."""
    return _packed_attention_mask_batch([text_len], image_len, device)


class NativeGlmImageMoeDiffusion(nn.Module):
    """Flow-matching model over the shared GLM4-MoE-lite/HY checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        *,
        stage_config: str,
        device: torch.device | None = None,
        max_memory_gib: int = 50,
    ) -> None:
        super().__init__()
        self.checkpoint = Path(checkpoint)
        self.stage_config = stage_config
        config = Glm4MoeLiteConfig.from_pretrained(stage_config)
        config._attn_implementation = "sdpa"
        max_memory = {index: f"{max_memory_gib}GiB" for index in range(torch.npu.device_count())}
        self.backbone = Glm4MoeLiteForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            generation_config=GenerationConfig(),
        ).eval()
        self.config = config
        self.rms_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.latent_channels = 32
        self.patch_size = 2
        self.anchor_patch_dim = self.latent_channels * self.patch_size**2
        self.extra_device = device or torch.device("npu:0")

        self.diff_input_proj = nn.Linear(self.anchor_patch_dim, self.hidden_size)
        self.diff_output_proj = nn.Linear(self.hidden_size, self.anchor_patch_dim)
        self.diff_time_embed = _TimestepEmbedding(self.hidden_size, 256)
        self.diff_adaln_in = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size))
        self.diff_adaln_out = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size))
        for module in (
            self.diff_input_proj,
            self.diff_output_proj,
            self.diff_time_embed,
            self.diff_adaln_in,
            self.diff_adaln_out,
        ):
            module.to(self.extra_device, dtype=torch.bfloat16)
        self._load_extra_weights()
        self._restore_fp32_router_weights()

        rope_parameters = config.rope_parameters
        self.inv_freq = 1.0 / (
            rope_parameters["rope_theta"]
            ** (
                torch.arange(
                    0,
                    config.head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.extra_device,
                )
                / config.head_dim
            )
        )
        self.rope_sections = list(rope_parameters["mrope_section"])
        self.rope_layout = "qwen3"
        self.fused_attention = use_a2_fused_attention()
        self.fused_moe = use_a2_fused_moe()
        if self.fused_attention or self.fused_moe:
            self._install_module_adapters()
        logger.info(
            "Native GLM-Image A2 kernels: attention=%s moe=%s",
            self.fused_attention,
            self.fused_moe,
        )

    def _install_module_adapters(self) -> None:
        for layer in self.backbone.model.layers:
            if self.fused_attention:
                self._wrap_attention(layer.self_attn)
            if self.fused_moe:
                self._wrap_mlp(layer.mlp)

    def _wrap_attention(self, attention: nn.Module) -> None:
        original_forward = attention.forward

        @wraps(original_forward)
        def attention_forward(
            module: nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None = None,
            past_key_values: object | None = None,
            use_cache: bool | None = False,
            **kwargs: object,
        ):
            del use_cache, kwargs
            if attention_mask is None:
                raise RuntimeError("A2 fused attention requires a packed attention mask")
            if past_key_values is not None:
                raise RuntimeError("A2 fused attention does not support KV cache")
            npu_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask < 0
            return a2_attention_forward(
                module,
                hidden_states,
                npu_mask,
                position_embeddings,
            )

        attention.forward = MethodType(attention_forward, attention)

    def _wrap_mlp(self, mlp: nn.Module) -> None:
        original_forward = mlp.forward

        @wraps(original_forward)
        def mlp_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            if hasattr(module, "experts"):
                return a2_moe_forward(module, hidden_states)
            return original_forward(hidden_states)

        mlp.forward = MethodType(mlp_forward, mlp)

    def _load_extra_weights(self) -> None:
        index = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for name, shard in weight_map.items():
            if not name.startswith("diff_"):
                continue
            with safe_open(self.checkpoint / shard, framework="pt", device="cpu") as handle:
                state[name] = handle.get_tensor(name)
        missing, unexpected = self.load_state_dict(state, strict=False)
        ignored = {name for name in missing if name.startswith("backbone.") or ".weight" not in name}
        real_missing = set(missing) - ignored
        if real_missing or unexpected:
            raise RuntimeError(
                f"GLM-Image diffusion projection mismatch: missing={sorted(real_missing)} "
                f"unexpected={sorted(unexpected)}"
            )

    def _restore_fp32_router_weights(self) -> None:
        """Keep router parameters exact; the shared backbone otherwise runs BF16."""
        index = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        for layer_idx, layer in enumerate(self.backbone.model.layers):
            if not hasattr(layer.mlp, "gate"):
                continue
            gate = layer.mlp.gate
            weight_name = f"model.layers.{layer_idx}.mlp.gate.weight"
            bias_name = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            with safe_open(
                self.checkpoint / weight_map[weight_name],
                framework="pt",
                device="cpu",
            ) as handle:
                weight = handle.get_tensor(weight_name)
            with safe_open(
                self.checkpoint / weight_map[bias_name],
                framework="pt",
                device="cpu",
            ) as handle:
                bias = handle.get_tensor(bias_name)
            gate.weight.data = weight.to(device=gate.weight.device, dtype=torch.float32)
            gate.e_score_correction_bias.data = bias.to(
                device=gate.e_score_correction_bias.device,
                dtype=torch.float32,
            )

    def _position_tensors(self, layout: GlmImageTokenLayout) -> tuple[torch.Tensor, torch.Tensor, int]:
        text_len = layout.anchor
        image_len = layout.grid_h * layout.grid_w
        text_positions = torch.arange(text_len, device=self.extra_device).view(1, -1).expand(3, -1)

        h_coord = torch.arange(layout.grid_h, device=self.extra_device, dtype=torch.float32)
        w_coord = torch.arange(layout.grid_w, device=self.extra_device, dtype=torch.float32)
        image_positions = torch.stack(
            (
                torch.full(
                    (layout.grid_h * layout.grid_w,),
                    float(text_len),
                    dtype=torch.float32,
                    device=self.extra_device,
                ),
                (text_len + h_coord[:, None]).expand(layout.grid_h, layout.grid_w).reshape(-1),
                (text_len + w_coord[None, :]).expand(layout.grid_h, layout.grid_w).reshape(-1),
            )
        )
        position_ids = torch.cat((text_positions, image_positions), dim=1).to(torch.long)
        image_mask = torch.cat(
            (
                torch.zeros(text_len, dtype=torch.bool, device=self.extra_device),
                torch.ones(image_len, dtype=torch.bool, device=self.extra_device),
            )
        )
        return position_ids, image_mask, image_len

    def _batch_position_tensors(
        self,
        layouts: list[GlmImageTokenLayout],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        position_rows: list[torch.Tensor] = []
        image_masks: list[torch.Tensor] = []
        image_len: int | None = None
        for layout in layouts:
            position_ids, image_mask, current_image_len = self._position_tensors(layout)
            if image_len is None:
                image_len = current_image_len
            elif current_image_len != image_len:
                raise ValueError(
                    "GLM-Image request batching requires one image resolution; "
                    f"got image lengths {image_len} and {current_image_len}"
                )
            position_rows.append(position_ids)
            image_masks.append(image_mask)

        assert image_len is not None
        texts = [layout.anchor for layout in layouts]
        padded_text_len = max(texts)
        length = padded_text_len + image_len
        positions = torch.zeros(
            (len(layouts), 3, length),
            dtype=torch.long,
            device=self.extra_device,
        )
        image_mask = torch.zeros(
            (len(layouts), length),
            dtype=torch.bool,
            device=self.extra_device,
        )
        for batch_index, (text_len, text_positions, current_image_mask) in enumerate(
            zip(texts, position_rows, image_masks, strict=True)
        ):
            positions[batch_index, :, :text_len] = text_positions[:, :text_len]
            positions[batch_index, :, padded_text_len:] = text_positions[:, text_len:]
            image_mask[batch_index, padded_text_len:] = current_image_mask[text_len:]
        return positions, image_mask, padded_text_len, image_len

    def predict_velocity_batch(
        self,
        layouts: list[GlmImageTokenLayout],
        latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if not layouts:
            raise ValueError("GLM-Image request batch must not be empty")
        batch_size = len(layouts)
        if latents.shape[0] != batch_size:
            raise ValueError(f"latent batch {latents.shape[0]} != request batch {batch_size}")

        text_lengths = [layout.anchor for layout in layouts]
        padded_text_len = max(text_lengths)
        input_ids = torch.zeros(
            (batch_size, padded_text_len),
            dtype=torch.long,
            device=self.extra_device,
        )
        for batch_index, layout in enumerate(layouts):
            text_len = layout.anchor
            input_ids[batch_index, :text_len] = torch.tensor(
                layout.input_ids[:text_len],
                dtype=torch.long,
                device=self.extra_device,
            )
        text_hidden = self.backbone.model.embed_tokens(input_ids)

        position_ids, image_mask, padded_text_len, image_len = self._batch_position_tensors(layouts)
        if latents.shape[1] != image_len:
            raise ValueError(f"latent rows {latents.shape[1]} != image grid {image_len}")
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError(f"timestep shape {tuple(timestep.shape)} != ({batch_size},)")
        time_embedding = self.diff_time_embed(timestep.float() * 1000.0)
        scale, shift = self.diff_adaln_in(time_embedding).chunk(2, dim=-1)
        image_hidden = self.diff_input_proj(latents)
        image_hidden = _adaln_apply(
            image_hidden,
            scale.unsqueeze(1),
            shift.unsqueeze(1),
            self.rms_eps,
        )

        hidden_states = torch.cat((text_hidden, image_hidden), dim=1)
        rope_tables = [
            _select_mrope(
                position_ids[batch_index],
                image_mask[batch_index],
                inv_freq=self.inv_freq,
                sections=self.rope_sections,
                layout=self.rope_layout,
            )
            for batch_index in range(batch_size)
        ]
        cos = torch.stack([table[0].squeeze(0) for table in rope_tables])
        sin = torch.stack([table[1].squeeze(0) for table in rope_tables])
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        attention_mask = _packed_attention_mask_batch(
            text_lengths,
            image_len,
            self.extra_device,
        )

        for layer in self.backbone.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )
        hidden_states = self.backbone.model.norm(hidden_states)
        image_indices = torch.arange(
            padded_text_len,
            padded_text_len + image_len,
            device=hidden_states.device,
        )
        # A2 cannot lower a BF16 basic slice here, so use index_select rather
        # than ``hidden_states[:, -image_len:]``.
        image_hidden = torch.index_select(hidden_states, 1, image_indices).to(self.extra_device)
        output_scale, output_shift = self.diff_adaln_out(time_embedding).chunk(2, dim=-1)
        image_hidden = _adaln_apply(
            image_hidden,
            output_scale.unsqueeze(1),
            output_shift.unsqueeze(1),
            self.rms_eps,
        )
        return self.diff_output_proj(image_hidden)

    def predict_velocity(
        self,
        layout: GlmImageTokenLayout,
        latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_velocity_batch(
            [layout],
            latents.unsqueeze(0),
            timestep,
        ).squeeze(0)


class NativeGlmImageGenerator:
    """End-to-end native text-to-image generation for `think=False`."""

    def __init__(
        self,
        *,
        checkpoint: str,
        stage_config: str,
        tokenizer_path: str,
        vae_path: str,
        device: torch.device | None = None,
        max_memory_gib: int = 50,
        vae_device: str = "npu:0",
    ) -> None:
        configure_a2_diffusion()
        self.model = NativeGlmImageMoeDiffusion(
            checkpoint,
            stage_config=stage_config,
            device=device,
            max_memory_gib=max_memory_gib,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        vae_dir = Path(vae_path)
        vae_config = json.loads((vae_dir / "config.json").read_text())
        patch_size = int(vae_config.get("patch_size", 2))
        self.vae = AutoencoderKLFlux2(
            in_channels=int(vae_config.get("in_channels", 3)),
            out_channels=int(vae_config.get("out_ch", 3)),
            block_out_channels=tuple(
                int(vae_config.get("ch", 128)) * int(multiplier)
                for multiplier in vae_config.get("ch_mult", (1, 2, 4, 4))
            ),
            layers_per_block=int(vae_config.get("num_res_blocks", 2)),
            latent_channels=int(vae_config.get("z_channels", 32)),
            norm_num_groups=32,
            sample_size=int(vae_config.get("resolution", 256)),
            batch_norm_eps=float(vae_config.get("bn_eps", 1e-4)),
            patch_size=(patch_size, patch_size),
        )
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(dtype=torch.bfloat16)
        missing, unexpected = self.vae.load_state_dict(
            _remap_flux2_vae_state(load_file(vae_dir / "model.safetensors")),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(f"Flux2 VAE weight mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
        self.vae_device = torch.device(vae_device)
        self.vae.to(self.vae_device).eval()

    def _timesteps(self, sequence_length: int, steps: int) -> torch.Tensor:
        alpha = math.exp(1.15 + (1.8 - 1.15) * ((sequence_length - 4096) / (16384 - 4096)))
        raw = torch.linspace(1.0, 0.0, steps + 1)
        return alpha * raw / (1 + (alpha - 1) * raw)

    def _decode_latents(self, latents: torch.Tensor, layout: GlmImageTokenLayout) -> torch.Tensor:
        packed = latents.view(
            1,
            layout.grid_h,
            layout.grid_w,
            self.model.latent_channels,
            self.model.patch_size,
            self.model.patch_size,
        )
        latents = (
            packed.permute(0, 3, 1, 4, 2, 5)
            .reshape(
                1,
                self.model.latent_channels,
                layout.grid_h * self.model.patch_size,
                layout.grid_w * self.model.patch_size,
            )
            .to(self.vae_device, self.vae.dtype)
        )
        p = self.model.patch_size
        _, channels, latent_h, latent_w = latents.shape
        patched = (
            latents.view(1, channels, latent_h // p, p, latent_w // p, p)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(1, channels * p * p, latent_h // p, latent_w // p)
        )
        mean = self.vae.bn.running_mean.to(patched.device, patched.dtype).view(1, -1, 1, 1)
        std = (
            (self.vae.bn.running_var.to(patched.device, torch.float32) + self.vae.bn.eps)
            .sqrt()
            .to(patched.dtype)
            .view(1, -1, 1, 1)
        )
        patched = patched * std + mean
        latents = (
            patched.view(1, channels, p, p, latent_h // p, latent_w // p)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(1, channels, latent_h, latent_w)
        )
        logger.info(
            "Native GLM-Image VAE decode: shape=%s dtype=%s device=%s",
            tuple(latents.shape),
            latents.dtype,
            latents.device,
        )
        with standard_conv_internal_format():
            decoded = self.vae.decode(latents, return_dict=False)[0]
        if not bool(torch.isfinite(decoded).all()):
            raise RuntimeError("Flux2 VAE produced non-finite image pixels")
        return decoded.clamp(-1.0, 1.0)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        *,
        height: int = 512,
        width: int = 512,
        num_steps: int = 4,
        seeds: list[int] | None = None,
        generated_ids: list[list[int]] | None = None,
        generators: list[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        if not prompts:
            raise ValueError("GLM-Image request batch must not be empty")
        batch_size = len(prompts)
        seeds = seeds or [0] * batch_size
        generated_ids = generated_ids or [[] for _ in prompts]
        generators = generators or [None] * batch_size
        if len(seeds) != batch_size or len(generated_ids) != batch_size or len(generators) != batch_size:
            raise ValueError("GLM-Image batch seeds, generated IDs, and generators must match prompts")

        layouts = [
            build_diffusion_layout(self.tokenizer, prompt, height, width, request_ids)
            for prompt, request_ids in zip(prompts, generated_ids, strict=True)
        ]
        image_len = layouts[0].grid_h * layouts[0].grid_w
        if any(current.grid_h * current.grid_w != image_len for current in layouts):
            raise ValueError("GLM-Image request batching requires one image resolution")

        request_generators: list[torch.Generator] = []
        for seed, generator in zip(seeds, generators, strict=True):
            if generator is None:
                generator = torch.Generator(device=self.model.extra_device).manual_seed(seed)
            elif generator.device != self.model.extra_device:
                raise ValueError(
                    f"generator device {generator.device} does not match {self.model.extra_device}"
                )
            request_generators.append(generator)

        latents = torch.stack(
            [
                torch.randn(
                    (image_len, self.model.anchor_patch_dim),
                    generator=generator,
                    device=self.model.extra_device,
                    dtype=torch.bfloat16,
                )
                for generator in request_generators
            ],
            dim=0,
        )
        timesteps = self._timesteps(image_len, num_steps)
        for step in range(num_steps):
            current = timesteps[step].to(self.model.extra_device)
            next_step = timesteps[step + 1].to(self.model.extra_device)
            velocity = self.model.predict_velocity_batch(
                layouts,
                latents,
                current.view(1).expand(batch_size),
            )
            # GLM-Image's DMD2 student uses the renoise sampler:
            # x0 = x_t - t*v, then x_t-1 = (1-t_next)*x0 + t_next*eps.
            x0 = latents.float() - current.float() * velocity.float()
            noise = torch.stack(
                [
                    torch.randn(
                        (image_len, self.model.anchor_patch_dim),
                        generator=generator,
                        device=self.model.extra_device,
                        dtype=torch.float32,
                    )
                    for generator in request_generators
                ],
                dim=0,
            )
            latents = ((1.0 - next_step.float()) * x0 + next_step.float() * noise).to(latents.dtype)
        return torch.cat(
            [
                self._decode_latents(latents[batch_index : batch_index + 1], layout)
                for batch_index, layout in enumerate(layouts)
            ],
            dim=0,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        height: int = 512,
        width: int = 512,
        num_steps: int = 4,
        seed: int = 1234,
        generated_ids: list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.generate_batch(
            [prompt],
            height=height,
            width=width,
            num_steps=num_steps,
            seeds=[seed],
            generated_ids=[generated_ids or []],
            generators=[generator],
        )
