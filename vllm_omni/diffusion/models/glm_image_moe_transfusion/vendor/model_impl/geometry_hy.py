# 逐字节搬自 `veomni/models/transfusion/hy/geometry.py`。

# 为什么搬：token 布局要用的网格换算，make_hy_transform_factory 里调
#
# ⚠ 除 import 行（见 gen_extras.py 的 VENDOR_REWRITES），**一个字都没改** ——
# 逐字节复制是"数值不可能漂"的全部依据。重新生成（必须在 gen_* 之后）：
#     bash model_impl/tools/regen.sh

"""Resolve HY token geometry without changing the VAE or the ViT encoder."""


def configure_vae_geometry(args):
    """Flux2's exposed latent is H/8; diffusion patching sets the token stride.

    vae_compression_ratio is the *token* stride used by saved model configs,
    rather than the convolutional downsampling inside the VAE.
    """
    if args.diff_patch_size < 1:
        raise ValueError("diff_patch_size must be a positive integer")
    if not args.diff_enabled or args.diff_space != "vae" or not args.vae_ckpt_path:
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.vae_ckpt_path, trust_remote_code=True)
    if config.model_type != "flux2_vae":
        return
    stride = 2 ** (len(config.ch_mult) - 1)
    channels = config.z_channels
    if not config.unpatchify_latents:
        stride *= config.patch_size
        channels *= config.patch_size**2
    if args.diff_vae_latent_dim != channels:
        raise ValueError(
            f"diff_vae_latent_dim={args.diff_vae_latent_dim} disagrees with Flux2 encode channels={channels}"
        )
    args.vae_compression_ratio = stride * args.diff_patch_size


def configure_transform_geometry(options, vae_stride, cond_merge_size):
    """Apply model geometry after per-source options have been merged.

    Keep resize alignment and ViT merging independent of diffusion patching.
    Source YAMLs may specify resize budgets, but cannot change latent geometry.
    """
    options = dict(options)
    options.update(vae_latent_scale=vae_stride, cond_latent_scale=vae_stride,
                   vae_compression_ratio=vae_stride)
    if cond_merge_size is not None:
        options["cond_merge_size"] = cond_merge_size
        options["latent_scale"] = int(options.get("vit_patch_size", 16)) * cond_merge_size
    return options
