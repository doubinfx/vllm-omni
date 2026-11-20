"""跨卡（device_map）下的张量对齐补丁。

为什么需要
----------
HY 的模型前向是**按"只有一张卡"写的**：`_run_layer` 把 `item_cu / blk_start / blk_end /
position_embeddings` 这些 meta 张量以**闭包**形式交给注意力核，而 accelerate 的
`AlignDevicesHook` 只搬"模块 forward 的声明参数"。于是把 47 层铺到多张卡上之后：
`hidden_states` 在本层的卡上，meta 张量还在入口卡上 ⇒

    RuntimeError: Expected all tensors to be on the same device,
    but got repeats is on npu:0, different from other tensors on npu:3

（2026-11-16 第一次跑单进程多卡就撞上，分别崩在
`modular_glm_image_moe_transfusion_selfflow.py:1448 _encode_image_entry`
和 `:1308 _run_layer`。）

补的是什么
----------
**不改数值、不复制逻辑**：只在几个干净的方法边界上包一层，把该方法的张量入参搬到
"这一段计算实际所在的那张卡"，然后原样调原方法。目标卡取自模块自己的参数
（`decoder_layer` 的参数在哪，这一层就在哪），所以与 device_map 怎么切无关。

⚠ 只在**跨卡**时才装（`len(devices) > 1`）。单卡下这些搬运全是恒等，装了也只是白花时间，
而且会把"模型本来就是单卡语义"这件事藏起来。

⚠ 阶段二把模型代码复制进 hygen 时，应当把这些搬运**内联**到被复制的代码里，
然后删掉本文件 —— 猴补丁依赖上游方法签名不变，而这件事没人替我们保证。
"""

from __future__ import annotations

import contextlib
import os

import torch


def _to(value, device: torch.device):
    """把张量（含 list/tuple 嵌套）搬到 device。

    ★ **不做任何缓存**。这里曾经按张量对象身份缓存过搬运结果（省掉 47 层重复拷贝
      cos/sin），但那是我自己加的、没验过的优化：张量可能被原地改写、对象可能被释放后
      地址重用，缓存会让某一层悄悄吃到上一层的数据。先要对，再谈快。

    ★ host 张量**一律不动**。`cu_seqlens` 这类 meta 在原链路里是**故意留在 host** 的
      （`infer.py:220 _batch_to_device` 的注释），而 blockmask 核把 meta 缓存在 cu 张量
      的**对象身份**上。
    """
    if isinstance(value, torch.Tensor):
        if value.device == device or value.device.type == "cpu":
            return value
        return value.to(device, non_blocking=True)
    if isinstance(value, (list, tuple)):
        return type(value)(_to(item, device) for item in value)
    return value


@contextlib.contextmanager
def _on_device(device: torch.device):
    """把**当前设备**也切过去，不只是搬张量。

    昇腾上很多算子（含 patched FlashAttentionScore）按"当前设备"分配 workspace / 下发流，
    而 accelerate 的 AlignDevicesHook 只搬张量、不动当前设备。不切的后果**不是报错**：
    第一个跨卡层会算出"有限但错"的结果，再往下两层就全 nan
    （2026-11-16 逐层打点：层 11 absmax 与层 10 一模一样、层 12 起全 nan）。

    **GPU 上同样需要**（2026-11-18 实测）：sonic-moe 的 gemm 走 quack / CuTe DSL，
    kernel 是用裸指针经 tvm_ffi 下发的，按**当前 CUDA 设备**取 context ⇒ 张量在
    cuda:1 而当前设备还是 cuda:0 时崩 `CUDA Error: cudaErrorIllegalAddress`
    （栈在 `sonicmoe/functional:_down_projection_forward` → `quack/gemm.py:gemm`）。
    所以这里按设备类型取 `torch.npu` / `torch.cuda` 的同名 API，两边同一套语义。
    """
    if device.type not in ("npu", "cuda") or device.index is None:
        yield                                           # cpu / meta：与改动前完全一致
        return
    module = getattr(torch, device.type, None)          # torch.npu / torch.cuda
    if module is None or not hasattr(module, "device"):
        yield
        return
    with module.device(device.index):
        yield


def _device_of(module) -> torch.device | None:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def install(model, main_device: torch.device, *, log=print) -> list[str]:
    """给 `model` 装上对齐补丁，返回装了哪几处（幂等）。

    `main_device` = 主卡：模型前向主干上的张量一律归位到它。

    ⚠ 判据用 `hasattr(type(module), ...)`（走 MRO），不是 `type(module).__dict__`：
    这两个方法定义在**基类**上，只查 `__dict__` 会一处都装不上 —— 第六轮冒烟就这么
    打出了 "已装: []" 还继续往下跑。

    ⚠ **必须遍历所有子模块找拥有者**，不能只看顶层对象。实测（2026-11-16 第五轮冒烟）：
    顶层 `...ForMMGeneration` 和 backbone 都定义了 `_run_layer`，而真正跑的是
    **backbone 那一份**（`selfflow.py:2451` 顶层 forward → `:2017` backbone forward →
    `:1386 _run_layer`）。只给顶层打补丁 ⇒ 补丁一次都没生效，而且日志还说"已装"。
    """
    installed: list[str] = []
    for name, module in [("", model)] + [(n, m) for n, m in model.named_modules() if n]:
        installed.extend(_install_on(module, main_device, label=name or "<root>"))
    installed.extend(_wrap_layers(model, main_device))
    log(f"[hygen] 跨卡对齐补丁已装: {installed}")
    return installed


def _wrap_layers(model, main_device: torch.device) -> list[str]:
    """把设备上下文下沉到**每个 decoder 层的 forward**。

    为什么 `_run_layer` 那一处不够：**AR 解码走的是另一条路**。
    `hy/ar_decode.py` 的 `HYTextDecoder._chunk_logits` 自己逐层调 `self.layers[i](...)`
    （它要 KV-cache，不能用扩散那条 blockmask 路），完全不经过 `_run_layer` ⇒
    当前设备没切 ⇒ 思考链出来是**乱码**（2026-11-16 实测：2500 token 全是随机词、
    没有 <think> 开合、stop_reason=max_new_tokens；同一个 ckpt 在 8 卡链路上写的是通顺中文）。
    ⚠ 又一次"不报错只出垃圾"：产物齐全、耗时正常、tok/s 还更快。

    包在层的 forward 上，扩散路（`_layer_fn` 里 `decoder_layer(...)`）和 AR 路
    （`self.layers[i](...)`）就都覆盖到了。
    """
    wrapped: list[str] = []
    for index, layer in enumerate(model.model.layers):
        device = _device_of(layer)
        if device is None or device == main_device or getattr(layer, "_hygen_layer_wrapped", False):
            continue
        original = layer.forward

        def make(original=original, device=device):
            def forward(*args, **kwargs):
                with _on_device(device):
                    return original(*(_to(a, device) for a in args),
                                    **{k: _to(v, device) for k, v in kwargs.items()})
            return forward

        layer.forward = make()
        layer._hygen_layer_wrapped = True
        wrapped.append(index)
    return [f"layers[{len(wrapped)} 个跨卡层].forward"] if wrapped else []


def _install_on(module, main_device: torch.device, *, label: str) -> list[str]:
    if getattr(module, "_hygen_device_patched", False):
        return []
    installed: list[str] = []

    # ── ① 每个 decoder 层：进去的搬到本层的卡，出来的搬回主卡 ────────────
    #    "层内在自己的卡上算，进出都归位到主卡" —— 有了这条不变量，模型前向的**主干**
    #    （`_interleave` 把扩散入口输出和文本 embedding 拼起来那类）永远只看见主卡张量，
    #    不必逐处去补。代价是每层一次 hidden_states 的卡间拷贝（1024² 下约 17 MB，
    #    47 层往返合计 ~1.6 GB D2D，毫秒量级）。
    if hasattr(type(module), "_run_layer"):
        original_run_layer = module._run_layer

        counter = {"n": 0}

        def run_layer(decoder_layer, hidden_states, *args, **kwargs):
            device = _device_of(decoder_layer)
            if os.environ.get("HYGEN_DEBUG_DEVICES") and device != main_device and counter["n"] < 4:
                counter["n"] += 1
                devices = {str(p.device) for p in decoder_layer.parameters()}
                print(f"[hygen-dbg] layer#{counter['n']} params={sorted(devices)} "
                      f"_device_of={device} hidden={hidden_states.device} "
                      f"hook={hasattr(decoder_layer, '_hf_hook')} main={main_device}", flush=True)
            if device is None or device == main_device:
                return original_run_layer(decoder_layer, hidden_states, *args, **kwargs)
            with _on_device(device):
                out = original_run_layer(
                    decoder_layer, _to(hidden_states, device),
                    *(_to(a, device) for a in args),
                    **{k: _to(v, device) for k, v in kwargs.items()},
                )
            return _to(out, main_device)

        module._run_layer = run_layer
        installed.append(f"{label}._run_layer")

    # ── ② 扩散入口：meta 张量跟着 diff_input_proj 的卡走 ──────────────────
    #    device_map 把它钉在主卡时这是恒等；留着是为了换切法时也不会崩。
    if hasattr(type(module), "_encode_image_entry"):
        original_encode = module._encode_image_entry
        owner = module

        def encode_image_entry(*args, **kwargs):
            projection = getattr(owner, "diff_input_proj", None)
            device = projection.weight.device if projection is not None else main_device
            return original_encode(*(_to(a, device) for a in args),
                                   **{k: _to(v, device) for k, v in kwargs.items()})

        module._encode_image_entry = encode_image_entry
        installed.append(f"{label}._encode_image_entry")

    if installed:
        module._hygen_device_patched = True
    return installed
