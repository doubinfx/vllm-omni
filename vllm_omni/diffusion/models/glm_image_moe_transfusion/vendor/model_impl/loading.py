"""单进程把 HY-30B 加载起来，并按层铺到多张 NPU 上。

与现在跑通的 8 卡链路的差别只有"参数放在哪"
--------------------------------------------
`tasks/transfusion/infer.py:prepare_inference_runtime` 走 hp_fsdp2：每张卡存 1/8 的参数，
每层 forward 前 all-gather。本文件改成：**参数按层切给几张卡、各自常驻**
（accelerate 的 `dispatch_model`，等价于 BAGEL 那种 `device_map`），没有任何集合通信。
数值配方（config / dtype / scheduler / 采样）一律沿用那条路，见各处注释。

dtype：为什么要调 `prepare_inference_parameters`
-----------------------------------------------
原链路 `enable_mixed_precision: true` ⇒ 模型建成 **fp32**，由 hp_fsdp2 在 forward 里转 bf16，
而某些子树（norm / timestep embedder 等，由 `model.get_ignore_modules_in_mixed_precision()`
给出）**豁免**、始终 fp32。`infer.py:1009-1013` 的 `keep_parameter_dtype` 只在
`data_parallel_mode == "fsdp2"` 时才调 `prepare_inference_parameters`，hp_fsdp2 下不调——
因为 HP 自己会转。单进程没有 HP 替我们转，所以这里**必须显式调一次**：
`hy/inference.py:27-45` 的那个函数就是"FSDP2 在 forward 里用的同一套 dtype"。
全量压 bf16 会把豁免子树一起降精度；全留 fp32 则是 113 GiB 且与 forward 口径不符。

rope 的非持久 buffer 闸门（不装就静默出垃圾）
---------------------------------------------
`init_empty_weights()` / `to_empty()` 会把非持久 buffer `rotary_emb.inv_freq` 变成清零页，
它**不在 state_dict 里**（ckpt 不会恢复），`init_weights()` 的 rotary 分支实测也覆盖不到
⇒ 所有位置 cos≡1 / sin≡0 ⇒ rope 退化成恒等、位置信息全丢、**且不报错**。
`tasks/transfusion/infer_thinkgen_dmd.py:_check_rope_buffers` 是同源实现（改一处要改两处）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch

from .base import get_device_type, install_torch_npu_guard


def _log(message: str) -> None:
    print(f"[vllm-omni-glm-image] {message}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  环境
# ════════════════════════════════════════════════════════════════════════════
def _memory_allocated(index: int) -> int:
    """跨设备的已分配显存读数（torch.npu / torch.cuda 两套 API 名字一样但入口不同）。"""
    module = getattr(torch, get_device_type(), None)
    return int(module.memory_allocated(index)) if module is not None else 0


def configure_runtime(devices: list[int], *, memory_fraction: float | None = None) -> dict:
    """必须在第一次分配**之前**调。

    `memory_fraction` 默认 **None = 不设**。原链路（`infer_thinkgen_dmd.py:_configure_npu_runtime`）
    设它是因为 HCCL 的 buffer 在 PyTorch 分配器之外、两者叠加会把 HBM 吃光；
    单进程没有 HCCL，这个理由不成立，所以默认不动它 —— 没验过的旋钮不该默认开。
    真要限额时再传。

    NPU / GPU 两条路（口径取自仓库根的 `start_ref/{npu,gpu}.sh`）
    ----------------------------------------------------------
    两边都用 `attn_implementation=fa3_block`，但 blockmask kernel 的来源不同：
    NPU 靠 **进程级**的 patched CANN 算子包（`ASCEND_CUSTOM_OPP_PATH`，没有它会崩在
    161001）；GPU 靠一个外部 python 包（gpu_infer.sh 里靠
    `PYTHONPATH=<blockmask FA3 包>` 提供，本仓用 `VLLM_OMNI_GLM_IMAGE_FA3_LIB` 指），
    对应 `model_impl/ops/block_mask_fa3.py`。两者都得在起 python 之前准备好，所以这里
    只能**查**、不能补；查不到就响亮告警。

    ⚠ GPU 这条路在本机（纯 NPU）**没法实跑验证**。结构上按 start_ref/gpu.sh 对齐，
       但第一次上 GPU 集群时要把 `[vllm-omni-glm-image] 设备` 那行打出来对一遍。
    """
    device_type = get_device_type()
    if device_type == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError(
                "VEOMNI_DEVICE_TYPE=npu 但没有可用的 NPU。blockmask attention 与 rope 都没有"
                "纯 torch 实现，必须经 scripts/with_hy_zwd_runtime.sh 这类启动器拉起。")
        if memory_fraction is not None:
            if npu.is_initialized():
                raise RuntimeError("NPU 已初始化，memory_fraction 必须在第一次分配之前设")
            for index in devices:
                npu.set_per_process_memory_fraction(float(memory_fraction), int(index))
        extra = {"ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")}
        if not extra["ASCEND_CUSTOM_OPP_PATH"]:
            _log("⚠ 环境里没有 ASCEND_CUSTOM_OPP_PATH —— 自定义 FA 算子包多半没 source，"
                 "blockmask attention 会崩在 161001。用 scripts/with_hy_zwd_runtime.sh 起。")
        # 入口卡不是 0 号时，把**进程的当前设备**也挪过去（理由见下面 cuda 分支）。
        # devices[0]==0 时**一行都不执行** ⇒ 已验证的 NPU 口径（0-3）一个字节不变。
        if int(devices[0]) != 0:
            npu.set_device(int(devices[0]))
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("VEOMNI_DEVICE_TYPE=cuda 但 torch.cuda 不可用")
        if memory_fraction is not None:
            for index in devices:
                torch.cuda.set_per_process_memory_fraction(float(memory_fraction), int(index))
        # ★ 把进程的**当前设备**设成入口卡。不设的后果（2026-11-18 实测）：
        #   `--devices 4,5` / `4,5,6,7` 崩 `illegal memory access` / `CUBLAS_STATUS_ALLOC_FAILED`，
        #   而 `0,1` / `0,1,2,3` 正常 ⇒ 不是卡数问题，是**起始卡号不为 0**：
        #   当前设备还停在 cuda:0，而权重和输入都在 cuda:4.. ⇒ 某些 kernel / cublas handle
        #   在 0 号卡上建、拿到别卡的指针。`device_map_patches._on_device()` 只在**逐层**
        #   forward 里切，模型外层（ViT 入口、VAE 解码、采样循环）没人切。
        torch.cuda.set_device(int(devices[0]))
        # GPU 上**不需要**昇腾那种进程级算子包，但 **blockmask FA3 的 python 包是必需的**：
        # attn 解析成 fa3_block ⇒ `ops/block_mask_fa3.py` 要
        # `flash_attn_interface._flash_attn_forward(..., blk_hi, blk_lo)`，stock FA3/FA2 没有
        # 这两个参数。那个 import 在模块级被 `except ModuleNotFoundError` 吞掉 ⇒ 不补这句
        # 告警的话，第一次 attention 才报 `NameError: _flash_attn_forward`，离病因很远。
        # 用 VLLM_OMNI_GLM_IMAGE_FA3_LIB 指过去（scripts/run.sh 会把它加进 PYTHONPATH）。
        import importlib.util

        found = bool(importlib.util.find_spec("blockmask_fa3")
                     or importlib.util.find_spec("flash_attn_3")
                     or importlib.util.find_spec("flash_attn_interface"))
        if not found:
            _log("⚠ import 不到 flash_attn_interface / flash_attn_3 —— GPU 上 attn 走 fa3_block，"
                 "第一次 attention 会 NameError。export VLLM_OMNI_GLM_IMAGE_FA3_LIB=<blockmask FA3 包目录>")
        extra = {"blockmask_fa3_importable": found}
    else:
        raise RuntimeError(
            f"device_type={device_type!r}：model_impl 只支持 npu / cuda。blockmask attention 的 "
            f"eager 分支是空函数（modeling_hy.py:531 就是 pass），CPU 上会**静默**出错。")
    _log(f"设备 = {device_type} {devices}"
         + (f"  memory_fraction={memory_fraction}" if memory_fraction is not None else ""))
    return {"device_type": device_type, "devices": devices,
            "memory_fraction": memory_fraction, **extra}


# ════════════════════════════════════════════════════════════════════════════
#  rope 闸门
# ════════════════════════════════════════════════════════════════════════════
def check_rope_buffers(model) -> dict:
    """按解析式复算 `inv_freq`，不一致就就地修回。见模块 docstring。"""
    persistent = set(model.state_dict().keys())
    suspect = {name: buffer for name, buffer in model.named_buffers() if name not in persistent}
    rope = getattr(getattr(model.config, "text_config", None), "rope_parameters", None) or {}
    theta = float(rope.get("rope_theta", 1000000.0))
    repaired, bad, ok, meta = [], [], [], []
    for name, buffer in suspect.items():
        if buffer.is_meta:
            meta.append(name)
            continue
        value = buffer.detach().float().cpu()
        if "inv_freq" not in name:
            if not torch.isfinite(value).all():
                bad.append(f"{name}: 含 inf/nan 且不是 inv_freq，没法按解析式修")
            continue
        dim = value.numel() * 2
        want = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if torch.allclose(value, want, rtol=1e-4, atol=1e-8):
            ok.append(name)
            continue
        with torch.no_grad():
            buffer.copy_(want.to(buffer.device, buffer.dtype))
        repaired.append(f"{name}: 原值前 3={value[:3].tolist()}")
    if meta:
        bad.append(f"{len(meta)} 个非持久 buffer 仍在 meta 上（没有被物化），前几个：{meta[:3]}")
    if bad:
        raise RuntimeError("[vllm-omni-glm-image] rope 非持久 buffer 异常：\n  " + "\n  ".join(bad))
    if repaired:
        _log(f"★ rope 非持久 buffer 与解析式不符，已就地修复（不修 = rope 退化成恒等、"
             f"位置信息全丢且不报错）：{len(repaired)} 个")
    else:
        _log(f"rope 闸门通过：{len(ok)} 个 inv_freq 与 theta={theta:g} 一致")
    return {"repaired": repaired, "ok": ok, "n_suspect": len(suspect)}


# ════════════════════════════════════════════════════════════════════════════
#  建模 + 加载 + 分发
# ════════════════════════════════════════════════════════════════════════════
#: `moe_implementation` 是 NPU / GPU 之间**唯一**与建模有关的差异
#: （口径取自仓库根的 `start_ref/{npu,gpu}.sh`：npu=fused，gpu=sonic；
#:  两边的 `attn_implementation` 都是 fa3_block，只是 blockmask kernel 来源不同）。
#:
#: 为什么不能两边都写 fused：它改**参数布局**，而且换布局的那段是 NPU 专属的 ——
#:   `modeling_hy.py:1840`  if isinstance(module, Glm4MoeLiteExperts) and is_torch_npu_available():
#:       gate_up_proj = cat([gate_proj, up_proj]); del gate_proj; del up_proj
#:       module.__class__ = HYExperts; module.forward = fused_forward
#:   ⇒ 在 GPU 上这段整体跳过，参数留在 gate_proj/up_proj 而 forward 仍是 fused_forward
#:     （要 npu_grouped_matmul）⇒ 坏掉。sonic 则**直接**建出合并的 gate_up_proj
#:     并走 sonic_forward（gpu.sh 里 `uv sync --extra sonic-moe` 那个库）。
#: 好消息：两条路最终的 state_dict 键名都是 `experts.gate_up_proj`，
#: 所以 merge_hp_ckpt.py 的产物（138 个 gate_up_proj）两边通用，不用分别合权重。
_MOE_BY_DEVICE = {"npu": "fused", "cuda": "sonic"}

#: `attn_implementation` 同理。口径取自 `start_ref/{npu,gpu}_infer.sh`：
#:   npu_infer.sh 两个都**不给**，吃 yaml 默认
#:     （configs/transfusion/tjy/infer_selfflow_hy_...sd50.yaml:187-188
#:      attn_implementation: flash_attention_2 / moe_implementation: fused）
#:   gpu_infer.sh 显式覆盖成 `--model_args.attn_implementation fa3_block`
#:                          `--model_args.moe_implementation sonic`
#: 注意 flash_attention_2 走的是 transformers 的 `_flash_attention_forward`，
#: 它本身是设备无关的 ⇒ 在 GPU 上**大概也能跑**。但作者的 gpu_infer.sh 用的是
#: fa3_block，那是被实跑验证过的那套，这里就跟着它，不自己推断。
_ATTN_BY_DEVICE = {"npu": "flash_attention_2", "cuda": "fa3_block"}


def resolve_attn_implementation(requested: str | None) -> str:
    """`None` = 按设备选。给了就按给的来（不报错 —— 这个旋钮两边都有可用实现，
    不像 moe 那样会静默坏掉）。"""
    if requested is not None:
        return requested
    device_type = get_device_type()
    default = _ATTN_BY_DEVICE.get(device_type)
    if default is None:
        raise RuntimeError(f"device_type={device_type!r} 没有对应的 attn_implementation")
    return default


def resolve_moe_implementation(requested: str | None) -> str:
    """`None` = 按设备选（推荐）；给了字符串就按给的来，但与设备不匹配时**响亮报错**。"""
    device_type = get_device_type()
    default = _MOE_BY_DEVICE.get(device_type)
    if requested is None:
        if default is None:
            raise RuntimeError(f"device_type={device_type!r} 没有对应的 moe_implementation")
        return default
    if default is not None and requested != default and requested in ("fused", "sonic"):
        raise RuntimeError(
            f"moe_implementation={requested!r} 与设备 {device_type} 不匹配（该用 {default!r}）。\n"
            f"  fused 的参数布局是分开的 gate_proj/up_proj，靠 modeling_hy.py:1840 那段"
            f"**NPU 专属**的 patch 才并成 gate_up_proj；在 GPU 上那段不执行 ⇒ 静默坏掉。\n"
            f"  真要越过这个检查，传 spec.moe_implementation='eager'（慢，但两边都能跑）。")
    return requested


def _register_model_classes():
    """把 HY / Flux2VAE 的 config+model 注册进 Auto* 表。

    原链路靠 import 包的 `__init__` 做这件事（`veomni/models/cogview/__init__.py:66-67`
    注册 flux2_vae；HY 由 `modular_..._selfflow.py:2472-2478` 自己注册）。
    model_impl 自带的副本把大部分包 `__init__` 置空了（原文件第一行就把整棵模型树拉进来），
    所以 VAE 那两行要在这里补上。
    """
    from transformers import AutoConfig, AutoModel

    # 非昇腾设备上先装 torch_npu 占位模块：生成文件里有**模块级** `import torch_npu`
    # （`modeling_hy_text.py:361` 等），它们落在 always-on 的 import 链上 ⇒ GPU 上
    # 下面那行 `import vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.modeling_hy` 会直接 ModuleNotFoundError。
    # 占位模块**访问任何属性即 raise**，所以不会让 NPU 专属算子在 GPU 上悄悄跑。
    # NPU 上这个调用什么都不做（真的 torch_npu 在），已验证的 NPU 口径不变。
    if install_torch_npu_guard():
        _log("torch_npu 占位模块已装（非昇腾设备；访问即 raise，只为让模块级 import 通过）")

    # import 它就完成了 HY 的注册（模块底部自己 register）
    import vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.modeling_hy  # noqa: F401  （import 它就完成 HY 的 Auto* 注册）
    from .configuration_hy import Flux2VAEConfig
    from .modeling_hy_vae import Flux2VAE

    try:
        AutoConfig.register("flux2_vae", Flux2VAEConfig)
        AutoModel.register(Flux2VAEConfig, Flux2VAE)
    except ValueError:
        pass  # 已注册（重复调用）


def _from_config_on_meta(config, spec, *, torch_dtype: str = "float32",
                         attn_implementation: str | None = None):
    """在 meta 上按 config 建一个空模型。

    逐步复刻原链路 `build_foundation_model` → `HuggingfaceLoader.load_model`
    （`veomni/models/auto.py:108-127` + `loader.py:56-100`）的那条 meta 分支：
    设 `config._moe_implementation` → 注册 `flash_attention_2` 的实现 →
    `with init_empty_weights(): AutoModel.from_config(...)`。
    **不走 `build_foundation_model` 的原因**：它 import `veomni.models`，而那个包的
    `__init__` 第一行 `from . import custom, seed_omni, transformers` 会拉进 270 个模块 /
    10 万行 —— vendoring 就没法做了。
    """
    from transformers import AutoModel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from .loading_weights_hy import init_empty_weights
    from .ops.attention import flash_attention_forward

    _register_model_classes()
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)
    if spec is not None:
        config._moe_implementation = resolve_moe_implementation(spec.moe_implementation)
    with init_empty_weights():
        return AutoModel.from_config(
            config=config,
            torch_dtype=getattr(torch, torch_dtype),
            attn_implementation=attn_implementation or (
                resolve_attn_implementation(spec.attn_implementation) if spec else "eager"),
            trust_remote_code=True,
        )


def _module_bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters()) + \
           sum(b.numel() * b.element_size() for b in module.buffers())


def build_device_map(model, devices: list[int]) -> dict:
    """**只把 decoder 层铺开，其余全部钉在主卡。**

    为什么不用 `infer_auto_device_map` 自动切（2026-11-16 实测两次崩在这上面）：
    模型前向的主干是按单卡写的 —— `_interleave` 要把扩散入口的输出和文本 embedding
    拼起来、`_encode_image_entry` 要把 `cu_seqlens` 和扩散 latent 一起用。自动切法会把
    `embed_tokens` / `diff_input_proj` / ViT 分到不同卡上，这些地方就逐个崩
    （`Expected all tensors to be on the same device`），补起来没有尽头。

    （另一处实测：自动切法还把 `model.vq_visual.blocks.2` 这个 ViT 块拦腰切开 ——
    q_proj/k_proj 在一张卡、v_proj/out_proj 在另一张，注意力直接崩。模型自带的
    `_no_split_modules` 只列了语言骨干那几个块，没有 ViT。）

    钉死"非层模块全在主卡"之后，主干上的张量永远同卡；层是通过 `_run_layer` 调的，
    那里由 `device_map_patches` 保证"进去搬到本层的卡、出来搬回主卡"。
    非层模块合计很小（embedding + ViT + 扩散头 ≈ 2 GB），主卡装得下。
    """
    main = int(devices[0])
    layers = model.model.layers
    layer_bytes = [_module_bytes(layer) for layer in layers]
    other_bytes = _module_bytes(model) - sum(layer_bytes)
    share = (sum(layer_bytes) + other_bytes) / len(devices)

    # ⚠ **不能放 `""` 兜底键**。实测（2026-11-16 第八轮冒烟）：device_map 里有 `""` 时
    #   accelerate 把整个模型当成**一个块**、execution_device 统一成主卡，于是它的
    #   AlignDevicesHook 会把输入搬回主卡，而权重还在各自的卡上 ⇒ 每一层都崩
    #   `gamma is on npu:1, different from other tensors on npu:0`。
    #   逐个模块显式列出，accelerate 才会按模块设 execution_device。
    device_map: dict = {}
    for name, _child in model.named_children():
        if _child is model.model:
            continue
        device_map[name] = main
    for name, _child in model.model.named_children():
        if name != "layers":
            device_map[f"model.{name}"] = main

    budget = max(share - other_bytes, 0.0)  # 主卡先背了非层模块
    index, used = 0, 0.0
    for position, nbytes in enumerate(layer_bytes):
        if used + nbytes > budget and index < len(devices) - 1:
            index += 1
            used, budget = 0.0, share
        device_map[f"model.layers.{position}"] = int(devices[index])
        used += nbytes
    return device_map


def build_model(spec, devices: list[int]):
    """返回已铺到 `devices` 上、dtype 与 forward 口径一致的模型。"""
    from accelerate.big_modeling import dispatch_model
    from accelerate.utils import load_checkpoint_in_model
    from transformers import AutoConfig

    from .pipeline import prepare_inference_parameters

    spec.check()
    _log(f"spec      = {spec.name}")
    _log(f"实现      = model_impl 自带（{os.path.dirname(os.path.abspath(__file__))}），不依赖外部 veomni")
    _log(f"timeshift = {spec.assert_timeshift_matches_training()}")

    t0 = time.time()
    # 注册必须在 from_pretrained 之前：config.json 里的 model_type 是 HY 自己的，
    # 不注册 transformers 不认（"does not recognize this architecture"）。
    _register_model_classes()
    config = AutoConfig.from_pretrained(spec.config_dir, trust_remote_code=True)
    # 与原链路同参：enable_mixed_precision=true ⇒ float32 建模（见模块 docstring）
    model = _from_config_on_meta(config, spec).eval()
    _log(f"建模完成（meta）{time.time() - t0:.1f}s  model_type={config.model_type}")

    # ① 先整模型物化到 CPU 并灌权重。
    #    不直接 load_checkpoint_and_dispatch 到 NPU 的原因：非持久 buffer（rope inv_freq）
    #    不在 ckpt 里，分发之后还留在 meta 上；在 CPU 上物化完、修好 buffer、转好 dtype，
    #    再一次性搬到卡上，每一步都能单独检查。CPU 侧峰值 fp32 约 113 GiB。
    t1 = time.time()
    model.to_empty(device="cpu")
    load_checkpoint_in_model(model, spec.weights, dtype=torch.float32)
    _log(f"权重灌入 CPU 完成 {time.time() - t1:.1f}s")

    # ② 非持久 buffer：ckpt 里没有，必须自己复算
    check_rope_buffers(model)

    # ③ dtype：复刻 FSDP2 在 forward 里用的那套（bf16 + 豁免子树 fp32）
    t2 = time.time()
    with torch.no_grad():
        prepare_inference_parameters(model)
    dtypes: dict = {}
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
    _log("dtype 分布（按参数量）: " + "  ".join(
        f"{k}={v / 1e9:.2f}B" for k, v in sorted(dtypes.items())) + f"  {time.time() - t2:.1f}s")

    # ④ 铺到卡上：只铺 decoder 层，其余钉主卡（见 build_device_map 的 docstring）
    t3 = time.time()
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    device_map = build_device_map(model, devices)
    placed: dict = {}
    for name, value in device_map.items():
        if name:
            placed[str(value)] = placed.get(str(value), 0) + 1
    model = dispatch_model(model, device_map=device_map, main_device=torch.device(f"{get_device_type()}:{devices[0]}"))
    if len(devices) > 1:
        # 模型前向主干是按单卡写的，层的进出要归位。见该模块 docstring。
        from .device_map_patches import install as install_device_patches

        install_device_patches(model, torch.device(f"{get_device_type()}:{devices[0]}"),
                               log=lambda message: _log(message.replace("[vllm-omni-glm-image] ", "")))
    _log(f"分发完成 {time.time() - t3:.1f}s；权重合计 {total / 2**30:.1f} GiB；"
         f"decoder 层分布 {placed}（非层模块全在 {get_device_type()}:{devices[0]}）")
    if os.environ.get("VLLM_OMNI_GLM_IMAGE_DEBUG_LAYERS"):
        _install_layer_probe(model)
    _log(f"各卡已用 HBM: " + "  ".join(
        f"{get_device_type()}{d}={_memory_allocated(d) / 2**30:.1f}GiB" for d in devices))
    return model


def _install_layer_probe(model) -> None:
    """逐层打印 hidden 的 absmax / 是否有限。查"数值在哪一层炸"用。

    现象对不上就先加打点再猜 —— 跨卡边界（层从一张卡换到另一张）与数值发散
    是两种完全不同的毛病，光看最终图分不出来。
    """
    def make(index, layer):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            value = tensor.detach().float()
            print(f"[model_impl-layer] {index:02d} dev={tensor.device} absmax={value.abs().max().item():.4g} "
                  f"finite={bool(torch.isfinite(value).all())}", flush=True)
        return hook

    for index, layer in enumerate(model.model.layers):
        layer.register_forward_hook(make(index, layer))


def build_vae(spec, device):
    """与原链路同一条路：meta 上建空壳 + `load_model_weights`（`infer.py:985-997`，
    torch_dtype 取 `build_foundation_model` 的默认值 bfloat16、attn eager）。"""
    from transformers import AutoConfig

    from .loading_weights_hy import load_model_weights

    _register_model_classes()
    config = AutoConfig.from_pretrained(spec.vae_path, trust_remote_code=True)
    vae = _from_config_on_meta(config, None, torch_dtype="bfloat16", attn_implementation="eager")
    load_model_weights(vae, spec.vae_path, device.type)
    vae = vae.eval().to(device)
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae


def build_tokenizer(spec):
    import inspect

    from .tokenizer_hy import load_tokenizer  # 与 veomni infer.py:29 同一个实现

    tokenizer = load_tokenizer(spec.tokenizer_path, trust_remote_code=True)
    special_tokens = {"additional_special_tokens": list(spec.additional_special_tokens)}
    if "replace_additional_special_tokens" in inspect.signature(tokenizer.add_special_tokens).parameters:
        special_tokens["replace_additional_special_tokens"] = False
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def build_scheduler(spec, config, *, seed: int = 0):
    """读 `scheduler_config.json` 再按 spec 覆盖 timeshift 四锚点，并打 α 对账表。"""
    import math

    from .pipeline import build_scheduler as _build

    scheduler = _build(spec.scheduler_path, config, spec.num_steps, seed)
    base_seqlen, base_shift, shift_seqlen, shift = spec.timeshift
    scheduler.base_seqlen, scheduler.base_shift = base_seqlen, base_shift
    scheduler.shift_seqlen, scheduler.shift = shift_seqlen, shift

    def alpha(height: int, width: int) -> float:
        tokens = (height // int(config.vae_compression_ratio)) * (width // int(config.vae_compression_ratio))
        if scheduler.use_fix_shift:
            return float(scheduler.shift)
        mu = base_shift + (shift - base_shift) * ((tokens - base_seqlen) / (shift_seqlen - base_seqlen))
        return math.exp(mu)

    rows = {f"{s}x{s}": round(alpha(s, s), 4) for s in (512, 1024, 1536, 2048)}
    _log(f"timeshift 锚点 ({base_seqlen},{base_shift}) -> ({shift_seqlen},{shift})  α: {rows}")
    # 挂在 scheduler 上，让 Inferencer 能逐图把 α 写进 meta —— 8 卡那条链路的 manifest
    # 每条都存 `alpha`（`infer_thinkgen_dmd.py:_log_alpha_curve` 同一个算法），
    # 对账 timeshift 时要逐图比。α 错是**静默劣化**：2048² 上差 4 倍，图只糊不报错。
    scheduler._glm_image_alpha = alpha
    return scheduler


@dataclass
class Runtime:
    """一次加载的全部产物。"""

    spec: object
    model: object
    vae: object
    tokenizer: object
    scheduler: object
    config: object
    devices: list[int]
    #: 输入张量落在哪张卡（= 模型第一个模块所在的卡）
    entry_device: torch.device

    def summary(self) -> dict:
        return {
            "spec": self.spec.name,
            "devices": self.devices,
            "entry_device": str(self.entry_device),
            "merge_recipe": self.spec.merge_recipe(),
            "sampler": self.spec.sampler,
            "num_steps": self.spec.num_steps,
            "cfg_scale": self.spec.cfg_scale,
            "timeshift": list(self.spec.timeshift),
        }


def load_runtime(spec, devices: list[int] | None = None, *, memory_fraction: float | None = None,
                 seed: int = 0) -> Runtime:
    devices = [int(d) for d in (devices or [0, 1, 2, 3])]
    configure_runtime(devices, memory_fraction=memory_fraction)
    torch.set_grad_enabled(False)
    model = build_model(spec, devices)
    entry = torch.device(f"{get_device_type()}:{devices[0]}")
    vae = build_vae(spec, entry)
    tokenizer = build_tokenizer(spec)
    scheduler = build_scheduler(spec, model.config, seed=seed)
    runtime = Runtime(spec=spec, model=model, vae=vae, tokenizer=tokenizer, scheduler=scheduler,
                      config=model.config, devices=devices, entry_device=entry)
    _log("加载完成：" + json.dumps(runtime.summary(), ensure_ascii=False))
    return runtime
