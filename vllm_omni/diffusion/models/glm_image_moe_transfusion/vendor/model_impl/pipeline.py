"""【自动生成，勿手改】扩散段与 collator —— 逐字抽自 veomni 仓库，见 model_impl/tools/vendor_tasks.py。

重新生成：
    python3 model_impl/tools/vendor_tasks.py
校验与上游是否仍一致：
    python3 model_impl/tools/vendor_tasks.py --check

这些函数是**实跑验证过**的数值配方（renoise 的 x0 公式、参考行/目标行顺序、packed 布局），
逐字复制是为了让"数值漂移"这件事在结构上不可能发生。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # noqa: F401

import torch
from einops import rearrange
from transformers import AutoTokenizer, PretrainedConfig  # noqa: F401

if TYPE_CHECKING:  # 只在 `_denoise_and_decode` 的签名注解里出现，运行时不需要真类
    MyModelArguments = Any
    MyInferArguments = Any

# 下面这些是被抽出来的那几段在**上游模块作用域**里用到的名字
# （靠 model_impl/tools/vendor_tasks.py 的扫描列出来，不是手挑的）
# ── 已搬进 model_impl 的 ──────────────────────────────────────────────────────────
from .processing_hy import (  # noqa: F401
    DataCollator,
    Glm4TransfusionImageTransform,
    Glm4TransfusionThinkGenTransform,
)
from .processing_hy import len2culen  # noqa: F401  （2026-11-17 并入 processing_hy）

# `MyCollator._collate_native_features` 用它兜住没列进 concat/list 的键。
# 上游是 `veomni/data/data_collator.py` re-export 的 torch 原版。
from torch.utils.data._utils.collate import default_collate  # noqa: F401
from .schedulers_hy import FlowMatchNativeResolutionScheduler

# ── eval 走不到的 format / CP：用会 raise 的占位 ────────────────────────────
#   `make_transform_factory` 要在**函数签名和分支里**提到这些名字，但 eval 只走
#   `wds_image_with_meta_json` 那一个分支（用户约束：只做 eval 的适配）。
#   做成占位而不是搬进来：真走到会**响亮地** raise，而不是悄悄用上训练数据管线。
from .base import _eval_only_placeholder as _p  # noqa: F401

FFRSampleTransform = _p("FFRSampleTransform", "ffrecord 数据源（训练用）")
ImageProcessingConfig = _p("ImageProcessingConfig", "ffrecord 的图像配置", dataclass_like=True)
GLMBinaryTextSampleTransform = _p("GLMBinaryTextSampleTransform", "glm_binary 数据源")
LMDBTextSampleTransform = _p("LMDBTextSampleTransform", "lmdb 数据源")
ParquetTextSampleTransform = _p("ParquetTextSampleTransform", "parquet_text 数据源")
Glm4ARImageUndTransform = _p("Glm4ARImageUndTransform", "understanding 数据源")
LYZ_WDS_FORMATS = {}          # lyz 的 format 注册表；eval 不按 format 分发
cp_broadcast_ = _p("cp_broadcast_", "序列并行（单进程走不到）")
cp_broadcast_inputs = _p("cp_broadcast_inputs", "序列并行")
cp_vae_encode = _p("cp_vae_encode", "序列并行")
enable_transfusion_cp = _p("enable_transfusion_cp", "序列并行")


# ── 逐字抽自 tasks/transfusion/train.py:175-179 ──
_THINKGEN_MODES = {
    "wds_t2i_prompted_image_with_meta_json": "t2i",
    "wds_thinkgen_image_with_meta_json": "ttg",
    "wds_thinkonly_with_meta_json": "to",
}


# ── 逐字抽自 tasks/transfusion/train.py:182-293 ──
def make_transform_factory(
    tokenizer: AutoTokenizer,
    ffr_image_config: Optional[ImageProcessingConfig] = None,
    image_token_id: int = 0,
    image_start_token_id: int = 0,
    image_end_token_id: int = 0,
    lang_vocab_size: int = 0,
    vq_compression_ratio: int = 16,
    vae_compression_ratio: int = 16,
    vae_token_id: int | None = None,
):

    def _get_compatible_kwargs(cls, kw: dict, other_filter_names: tuple[str, ...]) -> dict:
        import dataclasses

        names = {f.name for f in dataclasses.fields(cls)}
        for other_name in other_filter_names:
            if other_name in names:
                names.remove(other_name)
        return {k: v for k, v in kw.items() if k in names}

    def factory(kwargs: dict, dataset_format: str = "webdataset_with_meta_json"):
        if dataset_format == "ffrecord":
            return FFRSampleTransform(
                image_config=ffr_image_config or ImageProcessingConfig(),
                tokenizer=kwargs.get("tokenizer"),
                seed=kwargs.get("seed", 1234),
                image_token_id=image_token_id,
                **_get_compatible_kwargs(FFRSampleTransform, kwargs, ("tokenizer", "seed")),
            )
        elif dataset_format == "lmdb":
            return LMDBTextSampleTransform(
                tokenizer=kwargs.get("tokenizer"),
                seed=kwargs.get("seed", 1234),
                **_get_compatible_kwargs(LMDBTextSampleTransform, kwargs, ("tokenizer", "seed")),
            )
        elif dataset_format == "glm_binary":
            return GLMBinaryTextSampleTransform(
                tokenizer=kwargs.get("tokenizer"),
                seed=kwargs.get("seed", 1234),
                **_get_compatible_kwargs(GLMBinaryTextSampleTransform, kwargs, ("tokenizer", "seed")),
            )
        elif dataset_format == "parquet_text":
            return ParquetTextSampleTransform(
                tokenizer=kwargs.get("tokenizer"),
                seed=kwargs.get("seed", 1234),
                **_get_compatible_kwargs(ParquetTextSampleTransform, kwargs, ("tokenizer", "seed")),
            )
        elif dataset_format in ("wds_und_image_with_meta_json", "parquet_und_image_with_meta"):
            return Glm4ARImageUndTransform(
                boi_token=image_start_token_id,
                eoi_token=image_end_token_id,
                image_token_id=image_token_id,
                **_get_compatible_kwargs(
                    Glm4ARImageUndTransform, kwargs, ("seed", "image_token_id", "boi_token", "eoi_token")
                ),
            )
        elif dataset_format in _THINKGEN_MODES:
            # luyizhuo SFT formats (CoT distillation line), one class three modes:
            #   t2i = request template + caption (no AR loss)
            #   ttg = think + final + shape loss, then diffusion
            #   to  = the ttg text stopping at <boi> (image never read)
            # ⚠ NATIVE LAYOUT ONLY. This transform emits block_range / inline <|vae|> tokens and
            # no token_type_ids, so under transfusion_variant=hy the collator would route these
            # items to _collate_native_features and the packed batch would be inconsistent.
            # The HY port of this transform is still TODO (dmd/PORT_TO_POSTTRAIN.md section 4);
            # the DMD trainers raise on such items rather than mixing layouts.
            return Glm4TransfusionThinkGenTransform(
                mode=_THINKGEN_MODES[dataset_format],
                gen_boi_token=image_start_token_id,
                gen_eoi_token=image_end_token_id,
                gen_placeholder_token=image_token_id,
                vq_compression_ratio=vq_compression_ratio,
                vae_compression_ratio=vae_compression_ratio,
                **_get_compatible_kwargs(
                    Glm4TransfusionThinkGenTransform,
                    kwargs,
                    ("seed", "mode", "gen_boi_token", "gen_eoi_token", "gen_placeholder_token"),
                ),
            )
        elif dataset_format in ("wds_image_with_meta_json", "parquet_image_with_meta"):
            if dataset_format == "parquet_image_with_meta":
                raise ValueError(
                    "parquet_image_with_meta is deprecated due to the compatibility edit data, use parquet direct"
                )
            return Glm4TransfusionImageTransform(
                gen_boi_token=image_start_token_id,
                gen_eoi_token=image_end_token_id,
                gen_placeholder_token=image_token_id,
                vq_compression_ratio=vq_compression_ratio,
                vae_compression_ratio=vae_compression_ratio,
                **_get_compatible_kwargs(
                    Glm4TransfusionImageTransform,
                    kwargs,
                    ("seed", "gen_boi_token", "gen_eoi_token", "gen_placeholder_token"),
                ),
            )
        else:
            return Glm4TransfusionImageTransform(
                gen_boi_token=image_start_token_id,
                gen_eoi_token=image_end_token_id,
                gen_placeholder_token=image_token_id,
                vq_compression_ratio=vq_compression_ratio,
                vae_compression_ratio=vae_compression_ratio,
                **_get_compatible_kwargs(
                    Glm4TransfusionImageTransform,
                    kwargs,
                    ("seed", "gen_boi_token", "gen_eoi_token", "gen_placeholder_token"),
                ),
            )

    return factory


# ── 逐字抽自 tasks/transfusion/train.py:1508-1526 ──
class HYInputAdapter:

    def __init__(self, transform):
        self.transform = transform

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "transform"), name)

    def __call__(self, item):
        import json
        from .processing_hy import flatten_nested_dict

        row = dict(item)
        if "file_path_binary" in row:
            row.setdefault("jpg", row["file_path_binary"])
        if "caption_info" in row and "meta_info" not in row:
            info = row["caption_info"]
            row["meta_info"] = flatten_nested_dict(json.loads(info) if isinstance(info, str) else info)
        return self.transform(row)


# ── 逐字抽自 tasks/transfusion/train.py:1529-1559 ──
def _build_thinkgen_hy(create, options: dict, mode: str):
    """luyizhuo 的 SFT/CoT transform（packed layout），按他 ``train_sft_lyz.py:52-75`` 的配方装。

    做法是**借一个配好的基类实例再搬字段**，而不是重新解析一遍 options：先用 gen 分支
    (``wds_image_with_meta_json``) 造出 :class:`Glm4TransfusionImageTransform`，于是 geometry
    (``configure_transform_geometry``: vae_latent_scale / latent_scale / cond_merge_size)、三个
    角色 token id（user / assistant / gen_image,含 tokenizer 查表与 fallback)、
    ``template_mode`` / ``split_quoted_text`` 的 setdefault、按 dataclass 字段过滤 —— 全部复用。
    这一层**没有**重复实现，上游改了那套逻辑我们自动跟上。

    ⚠ 他的 transform 只有 packed 一种布局（没有 native 分支、没有开关），输出与本仓 HY 契约
    逐项相符（2026-11-13 用真样本实测：``token_type_ids == build_text_token_type_ids(L)``、
    ``diff_pos_anchor == [len(input_ids)]``、``vit_image_grid_thw == vae_latent_shape``、
    序列里没有行内 ``<|vae|>``）。所以它能直接喂给 HY 的 collator 和 DMD 的 trainer ——
    我们自己那份 ``wds_thinkgen_transform.py``（native-only）才是不行的那个。
    """
    from dataclasses import fields


    wrapped = create(dict(options), "wds_image_with_meta_json")
    base = getattr(wrapped, "transform", wrapped)
    values = {f.name: getattr(base, f.name) for f in fields(type(base))}
    # 子类独有的字段（mode / res_sched / quote_impl / caption_quote ...）从该源的
    # transform_kwargs 里取；基类没有的键不会被上面那层过滤掉，所以要在这里补。
    own = {f.name for f in fields(Glm4TransfusionThinkGenTransform)} - set(values)
    values.update({k: v for k, v in options.items() if k in own})
    values["mode"] = mode
    # 和 gen 分支一样套 HYInputAdapter：parquet 源的 file_path_binary / caption_info 字段名
    # 归一化，WDS 源走到这里是恒等变换。
    return HYInputAdapter(Glm4TransfusionThinkGenTransform(**values))


# ── 逐字抽自 tasks/transfusion/train.py:1562-1707 ──
def make_hy_transform_factory(native_vision_config=None, vae_token_stride=None, cond_merge_size=None, **kwargs):
    """Select shared raw-image templates and adapters for preformatted data.

    template_mode controls newly constructed image/text templates. glm_binary
    is already plain pretraining text; FFR already contains its text layout and
    target mask, which are preserved in either stage.
    """

    factory = make_transform_factory(**kwargs)
    processor = None
    if native_vision_config is not None:
        from transformers import Glm4vImageProcessor

        vc = native_vision_config.vision_config
        processor = Glm4vImageProcessor(
            patch_size=vc.patch_size, temporal_patch_size=vc.temporal_patch_size, merge_size=vc.spatial_merge_size
        )

    def create(options, dataset_format="webdataset_with_meta_json"):
        options = dict(options)
        if vae_token_stride is not None:
            from .geometry_hy import configure_transform_geometry

            options = configure_transform_geometry(options, vae_token_stride, cond_merge_size)
        template_mode = options.get("template_mode", "pretrain_mode")
        if template_mode not in ("sft_mode", "pretrain_mode"):
            raise ValueError("template_mode must be 'sft_mode' or 'pretrain_mode'")
        if dataset_format in ("glm_binary", "ffrecord"):
            # These sources already carry their text layout (and FFR's target
            # mask). Preserve it instead of reconstructing a raw-image template.
            options.pop("template_mode", None)
        options.setdefault("vae_token_id", kwargs.get("vae_token_id"))
        tokenizer = kwargs["tokenizer"]
        source_tokenizer = options.pop("pretokenized_tokenizer_path", None)
        mapper = None
        if source_tokenizer and dataset_format in ("ffrecord", "glm_binary"):
            from .base import _eval_only_placeholder as _p_lazy
            get_token_mapper = _p_lazy('get_token_mapper', 'veomni.data.transforms.tokenizer_conversion')

            mapper = get_token_mapper(source_tokenizer, tokenizer)

        def finish(transform):
            if mapper is None:
                return transform
            from .base import _eval_only_placeholder as _p_lazy
            PretokenizedTransform = _p_lazy('PretokenizedTransform', 'veomni.data.transforms.tokenizer_conversion')

            return PretokenizedTransform(transform, mapper)

        native = native_vision_config is not None
        if native:
            # Keep the data source's resize alignment when adapting ViT inputs.
            options.update(
                cond_dual_encoder=True,
                cond_latent_scale=vae_token_stride or 16,
                cond_merge_size=2,
                latent_scale=32,
                vit_patch_size=16,
                vq_pre_pool_factor=2,
                eot_token=tokenizer.eos_token_id,
                vision_processor=processor,
                image_input_mode=native_vision_config.image_input_mode,
                requested_dual=native_vision_config.cond_image_dual_encoder,
            )
        for name, token, fallback in (
            ("user_token_id", "<|user|>", 154827),
            ("assistant_token_id", "<|assistant|>", 154828),
            ("gen_image_request_token_id", "<|gen_image|>", 154857),
        ):
            value = tokenizer.convert_tokens_to_ids(token)
            options.setdefault(name, int(value) if value is not None else fallback)
        if dataset_format == "ffrecord":
            from dataclasses import fields
            HYFFRSampleTransform = _p("HYFFRSampleTransform", "ffrecord 数据源")
            from .base import _eval_only_placeholder as _p2  # noqa: F811

            cls = HYFFRSampleTransform
            if native:
                from .base import _eval_only_placeholder as _p_lazy
                NativeVisionFFRDataTransform = _p_lazy('NativeVisionFFRDataTransform', 'veomni.models.transfusion.hy.data_native')

                cls = NativeVisionFFRDataTransform
                options.update(boi_token=kwargs["image_start_token_id"], eoi_token=kwargs["image_end_token_id"])
            accepted = {item.name for item in fields(cls)}
            options = {name: value for name, value in options.items() if name in accepted}
            options.update(
                image_config=kwargs.get("ffr_image_config") or ImageProcessingConfig(),
                image_token_id=kwargs["image_token_id"],
                cond_dual_encoder=True,
            )
            return finish(cls(**options))
        if dataset_format in ("wds_und_image_with_meta_json", "parquet_und_image_with_meta", "parquet_edit_with_meta"):
            from dataclasses import fields
            from .base import _eval_only_placeholder as _p_lazy
            Glm4ARImageUndTransform = _p_lazy('Glm4ARImageUndTransform', 'veomni.data.transforms.wds_und_transform')
            from .base import _eval_only_placeholder as _p_lazy
            Glm4TransfusionEditTransform = _p_lazy('Glm4TransfusionEditTransform', 'veomni.data.transforms.wds_edit_transform')

            cls = (
                Glm4TransfusionEditTransform
                if dataset_format == "parquet_edit_with_meta"
                else Glm4ARImageUndTransform
            )
            if dataset_format != "parquet_edit_with_meta":
                options.setdefault("template_mode", "pretrain_mode")
                options.setdefault("latent_scale", 32)
                options.setdefault("split_quoted_text", False)
                options.setdefault("cond_dual_encoder", True)
            if native:
                from .base import _eval_only_placeholder as _p_lazy
                NativeVisionEditDataTransform = _p_lazy('NativeVisionEditDataTransform', 'veomni.models.transfusion.hy.data_native')
                NativeVisionUndDataTransform = _p_lazy('NativeVisionUndDataTransform', 'veomni.models.transfusion.hy.data_native')

                cls = (
                    NativeVisionEditDataTransform
                    if dataset_format == "parquet_edit_with_meta"
                    else NativeVisionUndDataTransform
                )
            accepted = {item.name for item in fields(cls)}
            options = {name: value for name, value in options.items() if name in accepted}
            options.update(
                boi_token=kwargs["image_start_token_id"],
                eoi_token=kwargs["image_end_token_id"],
                image_token_id=kwargs["image_token_id"],
            )
            return HYInputAdapter(cls(**options))
        # luyizhuo 的 SFT/CoT 三个 format（t2i / ttg / to）。放在 gen 分支**之前**：
        # _build_thinkgen_hy 会借 gen 分支造基类实例，所以它必须先被这一层拦下来。
        _lyz_mode = LYZ_WDS_FORMATS.get(dataset_format)
        if _lyz_mode is not None:
            return _build_thinkgen_hy(create, options, _lyz_mode)
        if dataset_format in ("wds_image_with_meta_json", "parquet_image_with_meta", "parquet"):
            from dataclasses import fields

            options.setdefault("template_mode", "pretrain_mode")
            options.setdefault("split_quoted_text", False)
            accepted = {item.name for item in fields(Glm4TransfusionImageTransform)}
            options = {name: value for name, value in options.items() if name in accepted}
            options.update(
                gen_boi_token=kwargs["image_start_token_id"],
                gen_eoi_token=kwargs["image_end_token_id"],
                gen_placeholder_token=kwargs["image_token_id"],
            )
            native = Glm4TransfusionImageTransform(**options)
            return HYInputAdapter(native)
        options.pop("vae_token_id", None)
        return finish(factory(options, dataset_format))

    return create


# ── 逐字抽自 tasks/transfusion/train.py:296-489 ──
@dataclass
class MyCollator(DataCollator):
    concat_features: list = field(
        default_factory=lambda: [
            "input_ids",
            "uncond_input_ids",
            "attention_mask",
            "loss_masks",
            "labels",
            "vit_pixel_values",
            "vit_image_grid_thw",
            "vae_grid_thw",
            "is_ref_vae",
        ]
    )
    concat_last_dim_features: list = field(default_factory=lambda: ["position_ids", "uncond_position_ids"])
    offset_then_concat_features: list = field(default_factory=lambda: ["block_range", "uncond_block_range"])
    list_features: list[str] = field(default_factory=lambda: [])
    list_flatten_features: list[str] = field(default_factory=lambda: ["vae_pixel_values"])
    unsqueeze_0_features: list[str] = field(default_factory=lambda: ["input_ids", "uncond_input_ids", "labels"])
    repeat_times: int = 1

    def _collate_native_features(self, features: List[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        if self.repeat_times > 1:
            features = [feature for feature in features for _ in range(self.repeat_times)]
        batch = {}
        lens = torch.tensor([len(feature["input_ids"]) for feature in features])
        batch["cu_seqlens"] = len2culen(lens)
        batch["max_seqlen"] = lens.max().item()
        batch["seq_lens"] = lens
        keys = features[0].keys()
        if "uncond_input_ids" in keys:
            uncond_lens = torch.tensor([len(feature["uncond_input_ids"]) for feature in features])
            batch["uncond_cu_seqlens"] = len2culen(uncond_lens)
            batch["uncond_max_seqlen"] = uncond_lens.max().item()
        for input_name in keys:
            if input_name in self.concat_features:
                batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=0)
            elif input_name in self.concat_last_dim_features:
                batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1)
            elif input_name in self.list_features:
                batch[input_name] = [feature.get(input_name, None) for feature in features]
            elif input_name in self.list_flatten_features:
                batch[input_name] = sum([feature.get(input_name, None) for feature in features], start=[])
            elif input_name in self.offset_then_concat_features:
                cu_seqlens = batch["uncond_cu_seqlens"] if input_name.startswith("uncond") else batch["cu_seqlens"]
                feature_key_list = [feature[input_name] for feature in features]
                assert len(feature_key_list) == cu_seqlens.shape[0] - 1
                feature_key_list = [feature_key + cu_seqlens[idx] for idx, feature_key in enumerate(feature_key_list)]
                batch[input_name] = torch.cat(feature_key_list, dim=0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])
        for key in self.unsqueeze_0_features:
            if key in batch:
                batch[key] = batch[key].unsqueeze(0)
        return batch

    def _collate_packed_features(self, features: List[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        all_input_ids, all_labels, all_loss_masks = ([], [], [])
        all_seq_lens, all_position_ids = ([], [])
        all_ds_idx, all_source_names = ([], [])
        all_vit_pixel_values, all_vit_image_grid_thw, all_vae_pixel_values, all_vae_latent_shapes = ([], [], [], [])
        all_visual_indices = []
        all_use_quantize, all_placeholder_ids = ([], [])
        all_pixel_placeholder_ids: List[int] = []
        all_index_placeholder_ids: List[int] = []
        all_diff_pos_anchors, all_diff_vit_grids = ([], [])
        per_item_has_diff: List[bool] = []
        for item in features:
            input_ids = item["input_ids"]
            token_type_ids = item["token_type_ids"]
            loss_masks = item["loss_masks"]
            labels = item["labels"]
            seq_len_val = item.get("seq_lens")
            if seq_len_val is None:
                seq_len_val = torch.tensor([input_ids.shape[0]])
            all_input_ids.append(input_ids.unsqueeze(0))
            all_labels.append(labels.unsqueeze(0))
            all_loss_masks.append(loss_masks.unsqueeze(0))
            all_seq_lens.append(seq_len_val)
            all_position_ids.append(token_type_ids.permute(1, 0).unsqueeze(1))
            if "ds_idx" in item:
                all_ds_idx.append(item["ds_idx"])
            if "source_name" in item:
                all_source_names.append(item["source_name"])
            vit_pixel_values = item.get("vit_pixel_values")
            visual_indices = item.get("visual_indices")
            if vit_pixel_values is not None or visual_indices is not None:
                pid = item.get("placeholder_token_id")
                pid_v = int(pid.item()) if pid is not None else 0
                all_placeholder_ids.append(pid_v)
                if vit_pixel_values is not None:
                    all_vit_pixel_values.extend(vit_pixel_values)
                    all_vit_image_grid_thw.extend(item["vit_image_grid_thw"])
                    all_pixel_placeholder_ids.append(pid_v)
                    all_use_quantize.extend(item["use_quantize"])
                if visual_indices is not None:
                    all_visual_indices.append(visual_indices)
                    all_index_placeholder_ids.append(pid_v if pid is not None else 154854)
            if item.get("has_diff", False):
                all_vae_pixel_values.extend(item["vae_pixel_values"])
                all_vae_latent_shapes.extend(item["vae_latent_shape"])
                all_diff_pos_anchors.extend(item["diff_pos_anchor"])
                all_diff_vit_grids.extend(item["vit_image_grid_thw"])
                per_item_has_diff.append(True)
            else:
                per_item_has_diff.append(False)
        batch_input_ids = torch.cat(all_input_ids, dim=1)
        batch_labels = torch.cat(all_labels, dim=1)
        batch_loss_masks = torch.cat(all_loss_masks, dim=1)
        batch_seq_lens = torch.cat(all_seq_lens, dim=0)
        batch_position_ids = torch.cat(all_position_ids, dim=2)
        batch_cu_seqlens = len2culen(batch_seq_lens)
        batch_max_seqlens = int(batch_seq_lens.max().item())
        batch = {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "loss_masks": batch_loss_masks,
            "seq_lens": batch_seq_lens,
            "cu_seqlens": batch_cu_seqlens,
            "position_ids": batch_position_ids,
            "max_seqlens": batch_max_seqlens,
            "attention_mask": torch.ones(batch_input_ids.shape, dtype=torch.long),
            "vit_pixel_values": torch.cat(all_vit_pixel_values, dim=0) if all_vit_pixel_values else None,
            "vit_image_grid_thw": torch.stack(all_vit_image_grid_thw, dim=0) if all_vit_image_grid_thw else None,
            "visual_indices": torch.cat(all_visual_indices, dim=0) if all_visual_indices else None,
            "use_quantize": torch.tensor(all_use_quantize, dtype=torch.bool) if all_use_quantize else None,
            "placeholder_token_ids": (
                torch.tensor(all_placeholder_ids, dtype=torch.long) if all_placeholder_ids else None
            ),
            "do_quantize_uniform": all(all_use_quantize) == any(all_use_quantize) if all_use_quantize else True,
            "do_quantize_value": bool(all(all_use_quantize)) if all_use_quantize else True,
            "vae_pixel_values": all_vae_pixel_values if all_vae_pixel_values else None,
            "vae_latent_shapes": torch.stack(all_vae_latent_shapes, dim=0) if all_vae_latent_shapes else None,
            "diff_pos_anchors": torch.stack(all_diff_pos_anchors, dim=0) if all_diff_pos_anchors else None,
            "diff_vit_grids": torch.stack(all_diff_vit_grids, dim=0) if all_diff_vit_grids else None,
            "per_item_has_diff": torch.tensor(per_item_has_diff, dtype=torch.bool),
            "placeholder_token_id_first": int(all_placeholder_ids[0]) if all_placeholder_ids else 0,
            "pixel_placeholder_token_id": int(all_pixel_placeholder_ids[0]) if all_pixel_placeholder_ids else None,
            "index_placeholder_token_id": int(all_index_placeholder_ids[0]) if all_index_placeholder_ids else None,
        }
        if all_ds_idx:
            batch["ds_idx"] = torch.tensor(all_ds_idx, dtype=torch.long)
        if all_source_names:
            batch["source_names"] = all_source_names
        return batch

    def __call__(self, features):
        if not features or "token_type_ids" not in features[0]:
            return self._collate_native_features(features)
        batch = self._collate_packed_features(features)
        mask = [
            bool(item.get("has_cond", False) and item.get("cond_pixel_values") is not None)
            for item in features
            if item.get("vit_pixel_values") is not None
            for _ in item["vit_image_grid_thw"]
        ]
        batch["reference_image_mask"] = torch.tensor(mask, dtype=torch.bool) if mask else None
        for item in features:
            if item.get("has_diff", False):
                grids = item.get("diff_vit_grid", item["vit_image_grid_thw"])
                shapes = item["vae_latent_shape"]
                if len(grids) != len(shapes) or any(
                    (not torch.equal(g.long(), v.long()) for g, v in zip(grids, shapes))
                ):
                    raise ValueError("HY diffusion RoPE requires the target latent grid")
        # Keep the original Transfusion ordering: references and targets of
        # each sample share one VAE batch. Only targets enter diffusion loss.
        images, grids, references, anchors, spatial = [], [], [], [], []
        for item in features:
            for image_key, grid_key, anchor_key, spatial_key, is_ref in (
                ("cond_pixel_values", "cond_latent_shape", "cond_pos_anchor", "cond_vit_grid", True),
                ("vae_pixel_values", "vae_latent_shape", "diff_pos_anchor", "diff_vit_grid", False),
            ):
                if not item.get("has_cond" if is_ref else "has_diff", False):
                    continue
                values = item.get(image_key) or []
                shapes = item.get(grid_key)
                if shapes is None:
                    shapes = []
                if len(values) != len(shapes):
                    raise ValueError(f"{image_key} and {grid_key} must describe the same images")
                images.extend(values)
                grids.extend(shapes)
                references.extend([is_ref] * len(values))
                anchors.extend(item.get(anchor_key) or [torch.tensor(0)] * len(values))
                coordinates = item.get(spatial_key)
                spatial.extend(coordinates if coordinates is not None and len(coordinates) else shapes)
        batch["vae_pixel_values"] = images or None
        batch["vae_grid_thw"] = torch.stack(grids) if grids else torch.empty((0, 3), dtype=torch.long)
        batch["is_ref_vae"] = torch.tensor(references, dtype=torch.bool)
        batch["diff_pos_anchors"] = torch.stack(anchors) if anchors else None
        batch["diff_vit_grids"] = torch.stack(spatial) if spatial else None
        return batch


# ── 逐字抽自 tasks/transfusion/train.py:929-1011 ──
def build_diff_batch(
    *,
    vae_pixel_values: List[torch.Tensor],
    vae_latent_shapes: torch.Tensor,
    diff_patch_size: int,
    per_item_has_diff: torch.Tensor,
    ar_cu_seqlens: torch.Tensor,
    vae_model,
    vae_dtype: torch.dtype,
    device: torch.device,
    diff_space: str = "vae",
    precomputed_latents: Optional[torch.Tensor] = None,
    shard_vae: bool = False,
    vae_latent_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Encode native VAE images (references and targets) and build fields for
    GlmImageARDiffForConditionalGeneration.forward.

    Returns dict with keys:
      vae_latents       : (L_diff_total, C_patch)   packed continuous latent tokens
      diff_grid_thw     : (N_images, 3)             per-image latent grid (T=1, H, W)
      diff_cu_seqlens   : (N_images + 1,)            packed VAE cu_seqlens
      ar_prefix_ranges  : (N_generation_items, 2)   full AR span of each target's item
    """
    with torch.no_grad():
        if precomputed_latents is not None:
            vae_latents = precomputed_latents
        elif diff_space == "pixel":
            flatten_list = [
                rearrange(
                    img.to(device, non_blocking=True),
                    "1 c (h ph) (w pw) -> (h w) (c ph pw)",
                    ph=diff_patch_size,
                    pw=diff_patch_size,
                )
                for img in vae_pixel_values
            ]
            vae_latents = torch.cat(flatten_list, dim=0)
        elif diff_space == "raev2":
            imgs = [img.to(device, non_blocking=True) for img in vae_pixel_values]
            vae_latents = vae_model(imgs).to(device, torch.bfloat16)
        else:
            if shard_vae and vae_pixel_values:
                def encode_one(image):
                    latent = vae_model.encode(image.to(device, vae_dtype, non_blocking=True))
                    return rearrange(
                        latent, "1 c (h ph) (w pw) -> (h w) (c ph pw)",
                        ph=diff_patch_size, pw=diff_patch_size,
                    )
                channels = (vae_latent_dim if vae_latent_dim is not None else vae_model.config.z_channels) * diff_patch_size**2
                parts = cp_vae_encode(vae_pixel_values, vae_latent_shapes.prod(-1).tolist(),
                                      encode_one, channels, vae_dtype, device)
            else:
                vae_latents_list = [
                    vae_model.encode(image.to(device, vae_dtype, non_blocking=True)) for image in vae_pixel_values
                ]
                parts = [
                    rearrange(latent, "1 c (h ph) (w pw) -> (h w) (c ph pw)",
                              ph=diff_patch_size, pw=diff_patch_size)
                    for latent in vae_latents_list
                ]
            vae_latents = torch.cat(parts)
    diff_grid_thw = vae_latent_shapes.clone()
    assert (
        diff_grid_thw.prod(dim=-1).sum() == vae_latents.shape[0]
    ), f"diff_grid_thw.prod().sum() = {diff_grid_thw.prod().sum()}, vae_latents.shape[0] = {vae_latents.shape[0]}"
    lens = diff_grid_thw.prod(dim=-1).to(torch.long)
    diff_cu_seqlens = torch.zeros(lens.shape[0] + 1, dtype=torch.long, device=lens.device)
    diff_cu_seqlens[1:] = lens.cumsum(0)
    has = per_item_has_diff.to(torch.bool)
    diff_item_ar_indices = torch.nonzero(has, as_tuple=False).squeeze(-1)
    ar_cu = ar_cu_seqlens.to(torch.long)
    ar_prefix_ranges = (
        torch.stack([torch.stack([ar_cu[i], ar_cu[i + 1]]) for i in diff_item_ar_indices.tolist()], dim=0)
        if diff_item_ar_indices.numel() > 0
        else torch.zeros(0, 2, dtype=torch.long, device=device)
    )
    return {
        "vae_latents": vae_latents.to(device),
        "diff_grid_thw": diff_grid_thw.to(device),
        "diff_cu_seqlens": diff_cu_seqlens.to(device),
        "ar_prefix_ranges": ar_prefix_ranges.to(device),
    }


# ── 逐字抽自 tasks/transfusion/train.py:1201-1238 ──
def prepare_inference_batch(batch, args, vae, device, precomputed_references=None):
    """Use native VAE inputs while leaving target noise to the denoiser."""
    grid = batch['vae_grid_thw'].to(device)
    flags = batch['is_ref_vae'].to(device=device, dtype=torch.bool)
    images = batch.get('vae_pixel_values') or []
    order = torch.cat((flags.nonzero().flatten(), (~flags).nonzero().flatten()))
    reference_grid = grid[flags]
    channels = args.diff_vae_latent_dim * args.diff_patch_size**2
    dtype = getattr(vae, 'dtype', torch.bfloat16)
    if not reference_grid.shape[0]:
        references = torch.empty((0, channels), dtype=dtype, device=device)
    elif precomputed_references is not None:
        references = precomputed_references
    else:
        references = build_diff_batch(
            vae_pixel_values=[image for image, ref in zip(images, flags.tolist()) if ref],
            vae_latent_shapes=reference_grid, diff_patch_size=args.diff_patch_size,
            per_item_has_diff=batch['per_item_has_diff'], ar_cu_seqlens=batch['cu_seqlens'],
            vae_model=vae, vae_dtype=dtype, device=device,
        )['vae_latents']
    targets = references.new_zeros((int(grid[~flags].prod(-1).sum()), channels))
    grid = grid.index_select(0, order)
    lengths = grid.prod(-1)
    cu = batch['cu_seqlens'].to(device, torch.long)
    fields = dict(
        input_ids=batch['input_ids'], position_ids=batch['position_ids'], labels=None, return_logits=False,
        pixel_values=batch.get('vit_pixel_values'), image_grid_thw=batch.get('vit_image_grid_thw'),
        placeholder_token_id=batch.get('pixel_placeholder_token_id') or batch.get('placeholder_token_id_first'),
        ar_cu_seqlens=batch['cu_seqlens'], ar_prefix_ranges=torch.stack((cu[:-1], cu[1:]), dim=1),
        has_diff_per_item=batch['per_item_has_diff'], reference_image_mask=batch.get('reference_image_mask'),
        vae_noised_latents=torch.cat((references, targets)), diff_grid_thw=grid,
        diff_cu_seqlens=torch.cat((lengths.new_zeros(1), lengths.cumsum(0))),
        is_ref_vae=flags.index_select(0, order), diff_timesteps=torch.zeros(grid.shape[0], device=device),
    )
    for name in ('diff_pos_anchors', 'diff_vit_grids'):
        values = batch.get(name)
        fields[name] = values.to(device).index_select(0, order) if values is not None else None
    return fields


# ── 逐字抽自 tasks/transfusion/infer.py:220-231 ──
def _batch_to_device(micro_batch: dict, device: torch.device) -> dict:
    """把 collate 出来的 batch 搬到卡上。``cu_seqlens`` 留在 host —— 与 train.py 的
    搬运规则逐字一致(blockmask 核的 meta 以 cu 张量**对象身份**做缓存,换设备等于每
    次重建)。"""
    return {
        k: _batch_to_device(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
        if hasattr(v, "to") and k != "cu_seqlens"
        else [ele.to(device, non_blocking=True) for ele in v]
        if isinstance(v, list) and len(v) > 0 and hasattr(v[0], "to")
        else v
        for k, v in micro_batch.items()
    }


# ── 逐字抽自 tasks/transfusion/infer.py:285-357 ──
@torch.no_grad()
def denoise_latents(
    fields, *, model, timesteps, cfg_scale, timeinput_scaler, prediction_type,
    unconditional_fields=None, x0_reference=None, sampler="ode",
):
    """Native all-image Euler state; reference rows stay fixed at every step.

    The optional x0_reference retains the existing native caller's x0 formula.
    HY already uses the current iterate; sharing this loop must not silently
    change either caller's numerical recipe.

    ``sampler="renoise"`` swaps only the state update (see SamplingArguments.sampler);
    the default "ode" keeps both callers bit-identical to before.
    """
    grid = fields["diff_grid_thw"]
    reference = fields["is_ref_vae"].to(dtype=torch.bool)
    target_rows = (~reference).repeat_interleave(grid.prod(-1))
    x_t = fields["vae_noised_latents"]
    arcflow = getattr(model, "_trd_arcflow", None) if sampler == "arcflow" else None
    if sampler == "arcflow":
        if arcflow is None:
            raise RuntimeError("sampler=arcflow but the policy heads were never attached to the model")
        if prediction_type != "v" or unconditional_fields is not None and cfg_scale != 1.0:
            raise ValueError("sampler=arcflow needs diff_pred_type v and cfg 1 (as trd.py trains)")
    for step in range(timesteps.shape[1] - 1):
        current, previous = timesteps[:, step], timesteps[:, step + 1]
        inputs = dict(fields, vae_noised_latents=x_t, diff_timesteps=current * timeinput_scaler)
        if arcflow is not None:
            arcflow.value = None
        prediction = model(**inputs).diff_pred[target_rows]
        if unconditional_fields is not None and cfg_scale != 1.0:
            negative = model(**dict(unconditional_fields, vae_noised_latents=x_t,
                                   diff_timesteps=current * timeinput_scaler)).diff_pred[target_rows]
            prediction = negative + cfg_scale * (prediction - negative)
        target = x_t[target_rows]
        if prediction_type == "x0":
            sigma = current.repeat_interleave(grid.prod(-1))[target_rows].unsqueeze(1)
            source = target if x0_reference is None else x0_reference[target_rows]
            prediction = (source - prediction) / sigma.clamp_min(1e-8)
        dt = (previous - current).repeat_interleave(grid.prod(-1))[target_rows].unsqueeze(1)
        if sampler == "arcflow":
            from .base import _eval_only_placeholder as _p_lazy
            arc_endpoint = _p_lazy('arc_endpoint', 'veomni.schedulers.arcflow')

            feat = arcflow.pop()
            if feat.shape[0] != prediction.shape[0]:
                raise RuntimeError(
                    f"arcflow: captured {feat.shape[0]} diffusion rows but {prediction.shape[0]} target "
                    "rows; the policy would score the wrong tokens"
                )
            policy = arcflow.head(feat, prediction)
            t_cur = current.repeat_interleave(grid.prod(-1))[target_rows].unsqueeze(1).float()
            t_prev = previous.repeat_interleave(grid.prod(-1))[target_rows].unsqueeze(1).float()
            # trd.py policy_endpoint: integrate t -> 0 at EVERY stage, then re-noise.
            x0 = arc_endpoint(policy, target.float(), t_cur, torch.zeros_like(t_cur),
                              log_gamma_clamp=arcflow.clamp).float()
            noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32)
            updated = ((1 - t_prev) * x0 + t_prev * noise).to(x_t.dtype)  # t_prev=0 on the last step -> x0
        elif sampler == "renoise":
            # The DMD2 student's own recipe, mirroring dmd2.py student_x0:
            #   x0 = x_t - t*v ; x_{t_prev} = (1 - t_prev)*x0 + t_prev*eps_fresh
            # `target` (the current iterate) is the x0 base on purpose -- x0_reference exists only
            # for the native ODE caller's formula and must not leak in here.
            # Reference rows need no special casing: only target_rows is ever written back.
            t_cur = current.repeat_interleave(grid.prod(-1))[target_rows].unsqueeze(1)
            t_prev = previous.repeat_interleave(grid.prod(-1))[target_rows].unsqueeze(1)
            x0 = target.float() - t_cur.float() * prediction.float()
            noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32)
            updated = ((1 - t_prev.float()) * x0 + t_prev.float() * noise).to(x_t.dtype)
        else:
            updated = (target + prediction * dt).to(x_t.dtype)
        x_t = x_t.clone()
        x_t[target_rows] = updated
    return x_t


# ── 逐字抽自 tasks/transfusion/infer.py:360-476 ──
@torch.no_grad()
def _denoise_and_decode(
    micro_batch: dict,
    *,
    model,
    vae_model,
    noise_scheduler,
    model_config,
    model_args: MyModelArguments,
    infer_args: MyInferArguments,
    model_runtime,
    device: torch.device,
) -> list["torch.Tensor"]:
    """一个 packed batch 的 flow-matching 采样 + VAE 解码。

    返回按 ``vae_grid_thw`` 行顺序排列的图像张量列表(每张 [c, h, w],值域 [-1, 1])。
    一个样本若带 n 张参考图,就占前 n 行(``is_ref_vae=True``,去噪时被 force_zero 冻住)
    加最后一行生成图,所以调用方要按每样本的图数切片、取最后一张。
    HY 与原生共用全部图像的去噪状态；HY 调用方仍只接收目标图。
    """
    if is_hy(model_config):

        # Keep HY's existing target-only draw, dtype and draw-before-encode
        # order. Drawing unused reference noise would change seeded samples.
        target_grid = micro_batch["vae_latent_shapes"].to(device)
        channels = model_config.diff_vae_latent_dim * model_config.diff_patch_size**2
        initial = torch.randn((int(target_grid.prod(-1).sum()), channels), device=device,
                              dtype=vae_model.dtype, generator=noise_scheduler.generator)
        fields = prepare_inference_batch(micro_batch, model_config, vae_model, device)
        reference = fields["is_ref_vae"]
        target_rows = (~reference).repeat_interleave(fields["diff_grid_thw"].prod(-1))
        fields["vae_noised_latents"][target_rows] = initial
        negative = None
        if infer_args.cfg_scale != 1.0:
            references = fields["vae_noised_latents"][~target_rows]
            negative = prepare_inference_batch(micro_batch["uncond_batch"], model_config, vae_model,
                                               device, precomputed_references=references)
        times = noise_scheduler.get_infernece_timesteps(fields["diff_grid_thw"], force_zero=reference)
        latent = denoise_latents(
            fields, model=model, timesteps=times[:, :infer_args.sampling_num_steps + 1], cfg_scale=infer_args.cfg_scale,
            timeinput_scaler=model_args.timeinput_scaler, prediction_type=model_config.diff_pred_type,
            unconditional_fields=negative, sampler=infer_args.sampler,
        )
        parts = latent[target_rows].split(target_grid.prod(-1).tolist())
        return [vae_model.decode(rearrange(
            part, "(h w) (c ph pw) -> 1 c (h ph) (w pw)", h=int(grid[1]), w=int(grid[2]),
            c=model_config.diff_vae_latent_dim, ph=model_config.diff_patch_size, pw=model_config.diff_patch_size,
        ).to(vae_model.dtype)).squeeze(0) for part, grid in zip(parts, target_grid)]
    is_ref_vae = micro_batch["is_ref_vae"]  # type: torch.Tensor
    vae_pixel_values = micro_batch["vae_pixel_values"]

    # --- vae ---
    vae_latents_list = [
        vae_model.encode(vae_pixel_value.to(device, vae_model.dtype, non_blocking=True))
        for vae_pixel_value in vae_pixel_values
    ]
    flatten_vae_latents_list = [
        rearrange(
            vae_latent,
            "1 c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=model_config.diff_patch_size,
            pw=model_config.diff_patch_size,
        )
        for vae_latent in vae_latents_list
    ]
    vae_clean = torch.cat(flatten_vae_latents_list, dim=0)
    vae_grid_thw = micro_batch["vae_grid_thw"]  # type: torch.Tensor
    vae_seqlens = vae_grid_thw.prod(dim=-1)
    noise_scale = None if model_config.diff_space != "pixel" else (vae_seqlens / 256.0).sqrt()
    vae_noised_latents, _, _ = noise_scheduler.add_noise(
        original_samples=vae_clean,
        grid_thw=vae_grid_thw,
        force_zero=is_ref_vae,
        noise_scale=noise_scale,
        min_t=0.0,
        force_one=~is_ref_vae,
    )

    timesteps_2d = noise_scheduler.get_infernece_timesteps(vae_grid_thw, force_zero=is_ref_vae)  # [N_diff, T+1]
    image_seqlens = vae_grid_thw.prod(dim=-1)

    fields = dict(
        input_ids=micro_batch["input_ids"], position_ids=micro_batch["position_ids"],
        pixel_values=micro_batch["vit_pixel_values"], image_grid_thw=micro_batch["vit_image_grid_thw"],
        vae_noised_latents=vae_noised_latents, diff_grid_thw=vae_grid_thw, is_ref_vae=is_ref_vae,
        cu_seqlens=micro_batch["cu_seqlens"], block_range=micro_batch["block_range"],
        max_seqlen=micro_batch["max_seqlen"], **model_runtime.inference_kwargs,
    )
    negative = None
    if infer_args.cfg_scale != 1.0:
        negative = dict(fields, input_ids=micro_batch["uncond_input_ids"],
                        position_ids=micro_batch["uncond_position_ids"],
                        cu_seqlens=micro_batch["uncond_cu_seqlens"],
                        block_range=micro_batch["uncond_block_range"],
                        max_seqlen=micro_batch["uncond_max_seqlen"])
    x_t = denoise_latents(
        fields, model=model, timesteps=timesteps_2d[:, :infer_args.sampling_num_steps + 1], cfg_scale=infer_args.cfg_scale,
        timeinput_scaler=model_args.timeinput_scaler, prediction_type=model_config.diff_pred_type,
        unconditional_fields=negative, sampler=infer_args.sampler,
        x0_reference=vae_noised_latents,
    )

    x_t_list = x_t.split(image_seqlens.tolist())
    x_t_list = [
        rearrange(
            x_t_,
            "(h w) (c ph pw) -> 1 c (h ph) (w pw)",
            h=grid_thw_[1],
            w=grid_thw_[2],
            ph=model_config.diff_patch_size,
            pw=model_config.diff_patch_size,
            c=model_config.diff_vae_latent_dim,
        )
        for x_t_, grid_thw_ in zip(x_t_list, vae_grid_thw)
    ]
    return [vae_model.decode(x_t_.to(vae_model.dtype)).squeeze(0) for x_t_ in x_t_list]


# ── 逐字抽自 veomni/models/transfusion/hy/inference.py:18-24 ──
def is_hy(config):
    return getattr(config, "model_type", "") in {
        "glm_image_moe_transfusion_hy", "glm_image_moe_transfusion_hy_selfflow",
        "glm4v_moe_hy_selfflow",
        "glm_image_moe_transfusion_hy_mot",
        "qwen3_transfusion_hy_mot",
    }


# ── 逐字抽自 veomni/models/transfusion/hy/inference.py:27-44 ──
def prepare_inference_parameters(model):
    """Store the same dtypes FSDP2 uses in forward, without optimizer masters.

    Preserve all FP32-exempt subtrees and buffers. Converting only other
    parameters moves FSDP's existing BF16 cast to load time; it adds no cast.
    """
    if torch.is_grad_enabled():
        raise ValueError("Inference parameter preparation requires gradients to be disabled")
    ignored = model.get_ignore_modules_in_mixed_precision()
    fp32_ids = {
        id(parameter)
        for module in model.modules() if isinstance(module, ignored)
        for parameter in module.parameters()
    }
    for parameter in model.parameters():
        if parameter.is_floating_point():
            dtype = torch.float32 if id(parameter) in fp32_ids else torch.bfloat16
            parameter.data = parameter.data.to(dtype)


# ── 逐字抽自 veomni/models/transfusion/hy/inference.py:180-192 ──
def build_scheduler(path, config, steps, seed):
    if not is_hy(config):
        scheduler = FlowMatchNativeResolutionScheduler.from_pretrained(path)
        scheduler.num_inference_steps = steps
        return scheduler
    values = {}
    file = Path(path) / "scheduler_config.json" if path else None
    if file is not None and file.is_file():
        values = json.loads(file.read_text())
    accepted = inspect.signature(FlowMatchNativeResolutionScheduler).parameters
    values = {k: v for k, v in values.items() if k in accepted}
    values.update(num_inference_steps=steps, seed=seed)
    return FlowMatchNativeResolutionScheduler(**values)


