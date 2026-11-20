"""每条可推理的权重线 = 一个 ModelSpec。

**所有默认值都取"已经被实跑验证过"的那套**，不是我现推出来的：

* 扩散配方（4 步 / renoise / cfg 1.0）来自 `configs/transfusion/zzx/dmd2_hy_infer.yaml`，
  DMD 学生的 kaodian 评测就是用它出的图。
* timeshift 四锚点 4096/1.15/16384/1.8 来自蒸 luyizhuo SFT 时训练侧传的 `SHIFT=1.8`
  （`scripts/zzx/pt_dmd.sh`）。**和训练那次不一致是静默劣化**：2048² 上 α 会差 4 倍，
  图只糊不报错。加载时会按 run 的 `training_config.yaml` 对账。
* AR 采样那组（temp 0.6 / top_p 0.9 / top_k 50 / rep **1.0** / max_new 4072）是
  SFT 作者 2026-11-14 给的定稿组合。rep 不能沿用旧的 1.3 —— 16 臂消融里 1.3 的
  自然收尾率是 0%。
* `transform_kwargs` 逐键抄 `configs/transfusion/zzx/dmd2_lyz_ttg_infer.yaml`。

`config_dir` 里的 `config.json` 是 `hygen/tools/dump_effective_config.py` 从那条验证过的
启动命令里 dump 出来的 effective config，**不是** `model_assets/config.json`
（为什么必须 dump 见那个工具的 docstring）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


HYGEN_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HYGEN_ROOT))
#: 资产路径一律走环境变量，**源码里不留任何绝对路径**（对外发布的前提）。
#: 解析顺序：环境变量 → `hygen/local_paths.json`（本机默认值，**不随开源导出**）
#: → 报错并打印该设哪个变量。
#: `HYGEN_LOCAL_PATHS` 可以指到任意一份 json（省得每次 export 一堆变量）；
#: 不给就找包目录旁边的 local_paths.json（开发机上的默认值，**不随发布导出**）。
_LOCAL_PATHS_FILE = os.environ.get("HYGEN_LOCAL_PATHS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "local_paths.json")
try:
    _LOCAL_PATHS = {k: v for k, v in json.load(open(_LOCAL_PATHS_FILE)).items()
                    if not k.startswith("_")}
except OSError:
    if os.environ.get("HYGEN_LOCAL_PATHS"):
        # 显式指了却读不到 ⇒ 报错，别静默退回"什么都没有"
        raise RuntimeError(f"HYGEN_LOCAL_PATHS={_LOCAL_PATHS_FILE} 读不到")
    _LOCAL_PATHS = {}
except ValueError as _error:
    raise RuntimeError(f"{_LOCAL_PATHS_FILE} 不是合法 json：{_error}") from None


def asset(name: str, what: str, *, required: bool = True) -> str:
    """取一个资产路径。缺了就**响亮报错**并说清该设什么、那是什么东西。"""
    value = os.environ.get(name) or _LOCAL_PATHS.get(name, "")
    if not value and required:
        raise RuntimeError(
            f"没有 {name} —— {what}。\n"
            f"  export {name}=<路径>；或者写进一份 json 再 "
            f"export HYGEN_LOCAL_PATHS=<那个 json>\n"
            f"  （现在找的是 {_LOCAL_PATHS_FILE}）\n"
            f"  （源码里不留绝对路径：这个仓库要对外发布，内部挂载点对别人没意义。）")
    return value


def _configs_root() -> str:
    """effective config（`<spec>/config.json`）在哪。

    两种布局都要能跑：
      * veomni 侧的开发目录：`hygen/specs/<spec>/`
      * 对外发布的 src-layout：`<repo>/configs/<spec>/`（它是**数据不是代码**，
        所以不在包里；pip 装上之后包里没有它 ⇒ 必须能用 HYGEN_CONFIGS_ROOT 指）
    顺序：环境变量 → 包内 specs/ → 仓库根 configs/ → 报错并说清该设什么。
    """
    override = os.environ.get("HYGEN_CONFIGS_ROOT")
    if override:
        return override
    for candidate in (os.path.join(HYGEN_ROOT, "specs"),
                      os.path.join(os.path.dirname(os.path.dirname(HYGEN_ROOT)), "configs")):
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(
        "找不到 configs 目录（每条 spec 的 effective config）。\n"
        "  export HYGEN_CONFIGS_ROOT=<仓库>/configs\n"
        "  —— 它是数据不是代码，不在 pip 包里；HY 的 config 是从三份训练 yaml 的 ~70 个\n"
        "  model_args 键现场装配的，**不能自己拼**（见 tools/dump_effective_config.py）。")


CONFIGS_ROOT = _configs_root()

#: 权重父目录（可选）。spec 的默认权重 = `$HYGEN_WEIGHTS_ROOT/<spec 名>`；
#: 直接用 `--ckpt <目录>` / `Inferencer.from_spec(..., ckpt=...)` 指定时不需要它。
WEIGHTS_ROOT = asset("HYGEN_WEIGHTS_ROOT",
                     "merge_hp_ckpt.py 合出来的 safetensors 放哪（每份约 57-113 GiB）",
                     required=False)


def _default_weights(name: str) -> str:
    return os.path.join(WEIGHTS_ROOT, name) if WEIGHTS_ROOT else ""


@dataclass
class ModelSpec:
    name: str
    #: 合并好的权重目录（`hygen/tools/merge_hp_ckpt.py` 的产物）
    weights: str
    #: dump 出来的 effective config 目录
    config_dir: str
    tokenizer_path: str
    vae_path: str
    #: `scheduler_config.json` 所在目录（原链路是 `model_args.scheduler_path or model_path`）
    #: ⚠ 它只提供**初始值**，四锚点最终由 `timeshift` 覆盖（见 loading.build_scheduler）。
    #: 所以这个目录里的锚点和 spec 不一致**不是错误** —— dmd2 那条指的就是一份无关的旧文件。
    scheduler_path: str
    #: 来源 ckpt，只为记账与 timeshift 对账
    source_checkpoint: str = ""
    #: ★ 训练侧 timeshift 真值的来源目录（含 `scheduler_config.json`，四锚点齐全）。
    #: 与 `scheduler_path` **刻意解耦**：scheduler_path 只是读初始值的地方，可能是无关文件；
    #: 这里指的是"训练时实际用的那组锚点"，是对账的权威。查不到就退回 training_config 的 diff_shift。
    timeshift_reference: str = ""

    # ── 扩散配方 ──────────────────────────────────────────────────────────
    sampler: str = "renoise"
    num_steps: int = 4
    cfg_scale: float = 1.0
    #: (base_seqlen, base_shift, shift_seqlen, shift)
    timeshift: tuple[int, float, int, float] = (4096, 1.15, 16384, 1.8)
    #: `diff_timesteps = t * 这个数`（原链路 model_args.timeinput_scaler，yaml 里是 1000.0）
    timeinput_scaler: float = 1000.0

    # ── 建模 ──────────────────────────────────────────────────────────────
    #: **None = 按设备选**（npu→flash_attention_2 / cuda→fa3_block，见
    #: loading.resolve_attn_implementation）。与 moe_implementation 一起，
    #: 构成 NPU / GPU 之间仅有的两处建模差异（口径见 start_ref/{npu,gpu}_infer.sh）。
    attn_implementation: str | None = None
    #: **None = 按设备选**（npu→fused / cuda→sonic，见 loading.resolve_moe_implementation）。
    #: 这是 NPU 与 GPU 之间唯一与建模有关的差异，写死成 "fused" 会在 GPU 上静默坏掉。
    moe_implementation: str | None = None
    #: 与 `model_args.additional_special_tokens` 同值（dump_recipe.json 里的实测值）
    additional_special_tokens: list[str] = field(default_factory=lambda: ["<|vae|>", "<|gen_request|>"])

    # ── AR（思考链）──────────────────────────────────────────────────────
    ar: dict = field(default_factory=lambda: dict(
        temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.0,
        frequency_penalty=0.0, presence_penalty=0.0, max_new_tokens=4072,
        stop_token_ids=[154830, 154827],  # <boi>, <|user|>
        attn_impl="eager", heartbeat=128,
    ))

    # ── token 布局 ────────────────────────────────────────────────────────
    transform_kwargs: dict = field(default_factory=lambda: dict(
        scale_factor=32, quote_policy="cot_off", quote_impl="stack_tolerant",
        shape_info_trainable=True, template_mode="pretrain_mode",
        image_max_pixels=4194304, image_min_pixels=4194304,
        text_dropout_rate=0.0, resolution_prompt_prob=1.0,
        image_min_pixels_filter=None, max_ratio=None, low_res_bias=None, res_sched=None,
    ))
    #: user 段是否带 ` h:H w:W`（训练 0.9 概率带；恒带 = 落在训练的多数分支）
    with_resolution_suffix: bool = True

    def check(self) -> None:
        if not self.weights:
            raise FileNotFoundError(
                f"[{self.name}] 没指定权重目录：传 --ckpt <权重目录>"
                f"（Python：Inferencer.from_spec(..., ckpt=...)），"
                f"或 export HYGEN_WEIGHTS_ROOT=<父目录>（取其下 {self.name}/）")
        for label, path in (("weights", self.weights), ("config_dir", self.config_dir),
                            ("tokenizer_path", self.tokenizer_path), ("vae_path", self.vae_path),
                            ("scheduler_path", self.scheduler_path)):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"[{self.name}] {label} 不存在: {path}")
        index = os.path.join(self.weights, "model.safetensors.index.json")
        if not os.path.isfile(index):
            raise FileNotFoundError(
                f"[{self.name}] 权重目录缺 model.safetensors.index.json: {self.weights}\n"
                f"    先跑: python3 hygen/tools/merge_hp_ckpt.py --src {self.source_checkpoint or '<ckpt>'}"
                f" --out {self.weights}")

    def assert_timeshift_matches_training(self) -> str:
        """timeshift 四锚点必须与训练一致，不一致**直接 raise**。

        不一致是**静默劣化**：2048² 上 α 会差 4 倍（6.05 vs 25.28），图只糊不报错。
        两条对账来源，都查：

        ① `scheduler_path/scheduler_config.json` —— 作者把四个值统一放这里，
           他那份的注释写明"推理侧必须与训练侧同值，否则 timeshift 不一致、出图偏移"。
           这是最直接的来源（四个值全有）。
        ② run 自己的 `training_config.yaml` 的 `diff_shift` —— 只有 shift 一个值，
           而且不是每个 run 都写（teacher 那份就没有），所以只能当补充。

        两条都查不到就返回说明并放过 —— 但会在日志里说清楚"没对上账"，
        不能让"查不到"看起来像"查过了没问题"。
        """
        notes = []
        want = tuple(self.timeshift)

        config_path = os.path.join(self.timeshift_reference or "", "scheduler_config.json")
        if self.timeshift_reference and os.path.isfile(config_path):
            values = json.load(open(config_path))
            keys = ("base_seqlen", "base_shift", "shift_seqlen", "shift")
            if all(k in values for k in keys):
                got = (values["base_seqlen"], values["base_shift"],
                       values["shift_seqlen"], values["shift"])
                if tuple(float(x) for x in got) != tuple(float(x) for x in want):
                    raise ValueError(
                        f"[{self.name}] scheduler_config.json 的四锚点 {got} 与 spec 的 {want} 不一致。"
                        " 这是静默劣化（2048² 上 α 差 4 倍，图只糊不报错）——改 spec 或换 scheduler_path。")
                notes.append(f"scheduler_config.json 四锚点 {want} 一致")
            else:
                notes.append(f"scheduler_config.json 只有 {sorted(set(keys) & set(values))}，"
                             "四锚点不全（spec 的值会覆盖它）")
        elif self.timeshift_reference:
            notes.append(f"⚠ timeshift_reference 下没有 scheduler_config.json: {self.timeshift_reference}")

        if self.source_checkpoint:
            path = os.path.join(os.path.dirname(os.path.dirname(self.source_checkpoint)),
                                "training_config.yaml")
            if os.path.isfile(path):
                import yaml

                trained = yaml.safe_load(open(path)).get("train_args", {}).get("diff_shift")
                if trained is None:
                    notes.append("training_config.yaml 没有 diff_shift 键")
                elif abs(float(trained) - float(self.timeshift[3])) > 1e-9:
                    raise ValueError(
                        f"[{self.name}] 训练时 diff_shift={trained}，而 spec 的 shift={self.timeshift[3]}。"
                        " 静默劣化，改 spec 或确认后手工覆盖。")
                else:
                    notes.append(f"training_config diff_shift={trained} 一致")
            else:
                notes.append("ckpt 旁边没有 training_config.yaml")
        if not notes:
            notes.append("⚠ 没有任何对账来源（既无 timeshift_reference 也无 training_config）")
        return "；".join(notes)

    def merge_recipe(self) -> dict:
        path = os.path.join(self.weights, "merge_recipe.json")
        return json.load(open(path)) if os.path.isfile(path) else {}


_SHARED_TOKENIZER = asset("HYGEN_TOKENIZER_PATH",
                          "tokenizer 目录（同时也是 scheduler_config.json 的来源）")
_VAE = asset("HYGEN_VAE_PATH", "Flux2 VAE 的 HF 目录")
#: 重锚过的 scheduler（四锚点齐全；推理侧必须与训练侧同值，否则是静默劣化）
_LYZ_SCHEDULER = asset("HYGEN_LYZ_SCHEDULER_PATH",
                       "teacher 那条线重锚过的 scheduler 目录", required=False)
_LYZ_TEACHER_CKPT = asset("HYGEN_LYZ_TEACHER_CKPT",
                          "teacher 的训练产物（只用于对账 training_config.yaml）",
                          required=False)

#: teacher（lyz SFT v7）的采样配方，**逐键照抄作者主 run** 的 run_config.json
#: （作者主 run 的 `out/sft_v7_s7000/run_config.json`）。
#: 他那边 7 个 run 的扩散段全是 40 步 / cfg 4.0 / **无 EMA**，一致；
#: AR 段早期 sweep 用 t0.8/rep1.3，后期（主 run / cotcache / force_close_ab）定稿成
#: t0.6/rep1.0 —— 与 DMD 学生那条同一组，印证了 rep1.3 是被 16 臂消融否掉的旧值。
_TEACHER_AR = dict(
    temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.0,
    frequency_penalty=0.0, presence_penalty=0.0,
    max_new_tokens=3072,          # ⚠ 学生那条是 4072；这里照作者主 run 的 3072
    stop_token_ids=[154830, 154827],
    attn_impl="eager", heartbeat=128,
)


SPECS: dict[str, ModelSpec] = {
    # DMD2 学生，teacher = luyizhuo SFT v7 的 10000 步。现在 ttg 批跑链路用的就是它。
    "dmd2_lyzsft10000_s2000": ModelSpec(
        name="dmd2_lyzsft10000_s2000",
        weights=_default_weights("dmd2_lyzsft10000_s2000"),
        config_dir=os.path.join(CONFIGS_ROOT, "dmd2_lyz"),
        tokenizer_path=_SHARED_TOKENIZER,
        vae_path=_VAE,
        scheduler_path=_SHARED_TOKENIZER,
        source_checkpoint=asset("HYGEN_DMD2_SOURCE_CKPT",
                                "学生的训练产物（只用于对账 training_config.yaml）",
                                required=False),
    ),

    # ── teacher（luyizhuo SFT v7 第 10000 步）──────────────────────────────
    #  扩散段 40 步 / cfg 4.0（不是学生那条 4 步 renoise cfg1）。
    #  ⚠ cfg≠1 会走 teacher 的 uncond 分支；HY 有个已知静默 bug：一个 batch 混多个 prompt
    #    时约一半 prompt 出纹理噪声。hygen 天然 batch=1，且 `_denoise` 里有断言挡住。
    "lyz_sft_v7_s10000": ModelSpec(
        name="lyz_sft_v7_s10000",
        weights=_default_weights("lyz_sft_v7_s10000"),
        config_dir=os.path.join(CONFIGS_ROOT, "dmd2_lyz"),   # 架构零 delta，见下
        tokenizer_path=_SHARED_TOKENIZER,
        vae_path=_VAE,
        scheduler_path=_LYZ_SCHEDULER,
        timeshift_reference=_LYZ_SCHEDULER,
        source_checkpoint=_LYZ_TEACHER_CKPT,
        sampler="ode",
        num_steps=40,
        cfg_scale=4.0,
        ar=dict(_TEACHER_AR),
    ),

    #  同一个 ckpt 的 EMA 权重（`ema.*` 771 键 + 缺的 527 键用同步 `model.*` 补）。
    #  ⚠ **作者自己从不开 EMA**（他 7 个 run 的 `use_ema_weights` 全是 false），
    #    所以这条**没有参照出图**，质量未知。默认请用上面那条非 EMA 的。
    "lyz_sft_v7_s10000_ema": ModelSpec(
        name="lyz_sft_v7_s10000_ema",
        weights=_default_weights("lyz_sft_v7_s10000_ema"),
        config_dir=os.path.join(CONFIGS_ROOT, "dmd2_lyz"),
        tokenizer_path=_SHARED_TOKENIZER,
        vae_path=_VAE,
        scheduler_path=_LYZ_SCHEDULER,
        timeshift_reference=_LYZ_SCHEDULER,
        source_checkpoint=_LYZ_TEACHER_CKPT,
        sampler="ode",
        num_steps=40,
        cfg_scale=4.0,
        ar=dict(_TEACHER_AR),
    ),
}

# teacher 为什么能复用 `specs/dmd2_lyz/config.json`（2026-11-17 验过，不是照搬记忆）：
# 他 run 的 `training_config.yaml` 的 `model_args` 与这份 effective config 共有 34 键，
# 只差 3 个，且三个都不影响架构 ——
#   ar_loss_weight (0.2888 vs 0.05)、loss_norm_tokens (10922 vs 21845)：训练侧，采样不读；
#   diff_time_emb_dim (None vs 2048)：`modular_glm_image_moe_transfusion.py:128` 写明
#     `None` 取 `text_config.hidden_size`，而它正是 2048 ⇒ 等价。
# 最硬的判据是 `hygen/tools/check_weight_keys.py`：拿这份 config 建模型，与合并后的权重
# key 集逐键比，零缺失零多余。


def get_spec(name: str) -> ModelSpec:
    if name not in SPECS:
        raise KeyError(f"没有这条 spec: {name!r}；已有：{sorted(SPECS)}")
    return SPECS[name]
