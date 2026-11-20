"""【自动生成，勿手改】Flux2 VAE —— eval 路径用到的全部。

重新生成 / 校验：
    python3 model_impl/tools/gen_vae.py
    python3 model_impl/tools/gen_vae.py --check     # 上游改了没
    python3 model_impl/tools/gen_vae.py --verify    # 新旧 VAE 的 encode/decode 是否逐位相同

逐字抽自 veomni 的两个文件（见末尾来源表），**函数体一字未改**。
裁掉了 `CogviewVAE` / `AbstractRegularizer` / `DiagonalGaussianRegularizer`
—— flux2 那边引用 0 次，而 eval 用的是 `Flux2VAE`。
"""

from __future__ import annotations

from abc import abstractmethod  # noqa: F401
from typing import Any, Callable, Optional, Tuple, Union  # noqa: F401

import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from . import base as logging          # get_logger 同名，调用处 `logging.get_logger` 照常
from .configuration_hy import Flux2VAEConfig


def overwrite_pretrained_config(config, _logger=None, **kwargs):
    """逐字抄自 `veomni/utils/model_utils.py:102-112`（12 行，不值得为它搬整个模块）。"""
    logger_to_use = _logger if _logger else logging.get_logger(__name__)
    for key, value in kwargs.items():
        if key in config:
            old_vale = getattr(config, key)
            logger_to_use.warning(f"overwrite **{key}** from {old_vale} to {value}")
            setattr(config, key, value)
        else:
            logger_to_use.warning(f"the kwargs key **{key}** **not** in the origin config, direct add")
            setattr(config, key, value)
    return config


logger = logging.get_logger(__name__)


# ── 逐字抽自 veomni/models/cogview/vae/modeling_cogview_vae.py:19-32 ──
class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            if self.with_conv
            else nn.Identity()
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


# ── 逐字抽自 veomni/models/cogview/vae/modeling_cogview_vae.py:35-50 ──
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        # no asymmetric padding in torch conv, must do it ourselves
        self.pre_conv = nn.ZeroPad2d((0, 1, 0, 1)) if self.with_conv else nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = (
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            if self.with_conv
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        x = self.conv(x)
        return x


# ── 逐字抽自 veomni/models/cogview/vae/modeling_cogview_vae.py:53-102 ──
class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# ── 逐字抽自 veomni/models/cogview/vae/modeling_cogview_vae.py:105-202 ──
class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


# ── 逐字抽自 veomni/models/cogview/vae/modeling_cogview_vae.py:205-341 ──
class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.gradient_checkpointing = gradient_checkpointing

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logger.debug(f"Working with {self.z_shape=} i.e. {np.prod(self.z_shape)} dimensions.")

        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = make_conv_cls(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def _make_resblock(self) -> Callable:
        return ResnetBlock

    def _make_conv(self) -> Callable:
        return torch.nn.Conv2d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def _forward_block(self, block, h, temb, **kwargs):
        """Helper function for gradient checkpointing."""
        return block(h, temb, **kwargs)

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.gradient_checkpointing and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(
                self._forward_block, self.mid.block_1, h, temb, use_reentrant=False, **kwargs
            )
            h = torch.utils.checkpoint.checkpoint(
                self._forward_block, self.mid.block_2, h, temb, use_reentrant=False, **kwargs
            )
        else:
            h = self.mid.block_1(h, temb, **kwargs)
            h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if self.gradient_checkpointing and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(
                        self._forward_block, self.up[i_level].block[i_block], h, temb, use_reentrant=False, **kwargs
                    )
                else:
                    h = self.up[i_level].block[i_block](h, temb, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


# ── 逐字抽自 veomni/models/cogview/vae/modeling_cogview_vae.py:344-391 ──
class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, return_gaussians: bool = False):
        gaussians = torch.randn_like(self.mean)
        x = self.mean + self.std * gaussians
        if return_gaussians:
            return x, gaussians
        return x

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


# ── 逐字抽自 veomni/models/cogview/vae/modeling_flux2_vae.py:59-84 ──
class AttnBlock(nn.Module):
    """LDM/BFL bottleneck self-attention (single head, 1×1 conv q/k/v/proj_out)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        # (b, c, h, w) -> (b, 1 head, h*w tokens, c). SDPA's default 1/sqrt(c)
        # scale matches the original `w_ * c**-0.5` softmax attention.
        q = q.view(b, 1, c, h * w).transpose(-1, -2).contiguous()
        k = k.view(b, 1, c, h * w).transpose(-1, -2).contiguous()
        v = v.view(b, 1, c, h * w).transpose(-1, -2).contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        return h_.transpose(-1, -2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


# ── 逐字抽自 veomni/models/cogview/vae/modeling_flux2_vae.py:87-122 ──
class Flux2Encoder(Encoder):
    """CogView/LDM ``Encoder`` + bottleneck attention + ``quant_conv``."""

    def __init__(self, *, ch, ch_mult, num_res_blocks, dropout, in_channels, resolution, z_channels):
        super().__init__(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=True,
        )
        block_in = ch * ch_mult[-1]
        self.mid.attn_1 = AttnBlock(block_in)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return self.quant_conv(h)


# ── 逐字抽自 veomni/models/cogview/vae/modeling_flux2_vae.py:125-160 ──
class Flux2Decoder(Decoder):
    """CogView/LDM ``Decoder`` + bottleneck attention + ``post_quant_conv``."""

    def __init__(self, *, ch, out_ch, ch_mult, num_res_blocks, dropout, in_channels, resolution, z_channels):
        super().__init__(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
        )
        block_in = ch * ch_mult[-1]
        self.mid.attn_1 = AttnBlock(block_in)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None

        h = self.conv_in(self.post_quant_conv(z))

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


# ── 逐字抽自 veomni/models/cogview/vae/modeling_flux2_vae.py:163-267 ──
class Flux2VAE(PreTrainedModel):
    """FLUX.2 VAE. ``encode`` / ``decode`` mirror ``QwenVAE2D``'s contract:
    ``encode(x: (b, 3, H, W) in [-1, 1]) -> (b, z, H/8, W/8)`` normalized latent,
    ``decode(z) -> (b, 3, H, W)`` image tensor (not a dataclass)."""

    config_class = Flux2VAEConfig
    _supports_gradient_checkpointing = False

    def __init__(self, config: Flux2VAEConfig, **kwargs):
        config = overwrite_pretrained_config(config, logger, **kwargs)
        super().__init__(config)
        self.config: Flux2VAEConfig

        self.z_channels = config.z_channels
        self.patch_size = config.patch_size

        self.encoder = Flux2Encoder(
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            z_channels=config.z_channels,
        )
        self.decoder = Flux2Decoder(
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            z_channels=config.z_channels,
        )
        # Frozen latent normalizer. Kept as a real BatchNorm2d module so that the
        # checkpoint's ``bn.running_mean`` / ``bn.running_var`` /
        # ``bn.num_batches_tracked`` buffers load by name; the forward path below
        # applies the statistics manually so it can never fall into train-mode
        # batch statistics or mutate the running stats.
        self.bn = nn.BatchNorm2d(
            config.z_channels * config.patch_size**2,
            eps=config.bn_eps,
            affine=False,
            track_running_stats=True,
        )

        # conv-stack compression only; the token grid is a further `patch_size`×
        # coarser (see module docstring).
        self.spatial_compression_ratio = 2 ** (len(config.ch_mult) - 1)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(b, c, h, w) -> (b, c*p*p, h/p, w/p), channel-major (c, ph, pw)."""
        p = self.patch_size
        b, c, h, w = x.shape
        assert h % p == 0 and w % p == 0, f"latent {h}x{w} not divisible by patch_size={p}"
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4)  # b, c, ph, pw, h/p, w/p
        return x.reshape(b, c * p * p, h // p, w // p)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Exact inverse of :meth:`_patchify`."""
        p = self.patch_size
        b, cp, h, w = x.shape
        c = cp // (p * p)
        x = x.view(b, c, p, p, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)  # b, c, h, ph, w, pw
        return x.reshape(b, c, h * p, w * p)

    def _bn_stats(self, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.bn.running_mean.to(ref.device, ref.dtype).view(1, -1, 1, 1)
        std = (self.bn.running_var.to(ref.device, torch.float32) + self.bn.eps).sqrt().to(ref.dtype)
        return mean, std.view(1, -1, 1, 1)

    def encode(self, x: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        assert x.ndim == 4, f"Expected 4D input (b, c, h, w), got {x.ndim}D"
        h = self.encoder(x)
        if self.config.sample_posterior:
            z = DiagonalGaussianDistribution(h).sample()
        else:
            z = torch.chunk(h, 2, dim=1)[0]  # posterior mean — FLUX.2's default

        z = self._patchify(z)
        mean, std = self._bn_stats(z)
        z = (z - mean) / std

        if self.config.unpatchify_latents:
            z = self._unpatchify(z)
        return z

    def decode(self, z: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        assert z.ndim == 4, f"Expected 4D input (b, c, h, w), got {z.ndim}D"
        if self.config.unpatchify_latents:
            z = self._patchify(z)

        mean, std = self._bn_stats(z)
        z = z * std + mean
        z = self._unpatchify(z)

        dec = self.decoder(z)
        return torch.clamp(dec, min=-1.0, max=1.0)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.encode(x)
        return ModelOutput(recons=self.decode(z), z=z)




#: 与上游 `veomni/models/cogview/vae/modeling_flux2_vae.py:270` 一致 —— loader 按这个名字取类
ModelClass = Flux2VAE

__all__ = ["Flux2VAE", "Flux2Encoder", "Flux2Decoder", "AttnBlock",
           "Encoder", "Decoder", "ResnetBlock", "Upsample", "Downsample",
           "DiagonalGaussianDistribution", "ModelClass"]

# ── 来源表（sha256 是上游原文的，供 --check 比对）──────────────────────────
_SOURCES = {
    "Upsample": {
        "source": "veomni/models/cogview/vae/modeling_cogview_vae.py:19-32",
        "sha256_upstream": "060da569781bc7a4d3d2d263a1428e05c6244a01e9f93db42dc83a3933915197"
    },
    "Downsample": {
        "source": "veomni/models/cogview/vae/modeling_cogview_vae.py:35-50",
        "sha256_upstream": "be64c42fb5b7a67acac471023dc5392e225946a200c877514490cabd88b43e80"
    },
    "ResnetBlock": {
        "source": "veomni/models/cogview/vae/modeling_cogview_vae.py:53-102",
        "sha256_upstream": "1a9f4d4afdc6eba0f2fac796b38f2223c9b2a28e24dcac4624c62d77b2ddd491"
    },
    "Encoder": {
        "source": "veomni/models/cogview/vae/modeling_cogview_vae.py:105-202",
        "sha256_upstream": "8b22d9b08eaf18c3333c0a92f3e10b395dc80130589b9835a0c1a90a287ee42d"
    },
    "Decoder": {
        "source": "veomni/models/cogview/vae/modeling_cogview_vae.py:205-341",
        "sha256_upstream": "d0f5838718825e48de7594a0a0fcb69ff6bb30e948926e772f20cfd27ad6ed55"
    },
    "DiagonalGaussianDistribution": {
        "source": "veomni/models/cogview/vae/modeling_cogview_vae.py:344-391",
        "sha256_upstream": "ac75928b7237d52d9dde047b5c35c0e14f552a4202bddaa706b5c0886a6ffd49"
    },
    "AttnBlock": {
        "source": "veomni/models/cogview/vae/modeling_flux2_vae.py:59-84",
        "sha256_upstream": "66b8d8bcc5c6efb0f58574627637e20bbc31cb69bbab994cfa3d518cc0825832"
    },
    "Flux2Encoder": {
        "source": "veomni/models/cogview/vae/modeling_flux2_vae.py:87-122",
        "sha256_upstream": "a3340cb8b9d8e88c67d35147711fd2c6dc811f50268bce2fecf6df7e43f5cf4a"
    },
    "Flux2Decoder": {
        "source": "veomni/models/cogview/vae/modeling_flux2_vae.py:125-160",
        "sha256_upstream": "44d393e76c624f4e4317fb9794c5db53777f5adbd49e414a630f264056e49d86"
    },
    "Flux2VAE": {
        "source": "veomni/models/cogview/vae/modeling_flux2_vae.py:163-267",
        "sha256_upstream": "83117a512ba4c3f94bd21cd3acee3c74d4aaf819a56ec7d99298502348c0b8aa"
    }
}
