"""model_impl 的基础层：日志、设备、以及单进程下的并行状态**桩**。

为什么有"桩"
------------
算子文件（`model_impl/ops/attention.py` 等）是**模块级** import `parallel_state` 和
`sequence_parallel` 的，所以不能简单删掉。但 model_impl 是**单进程**推理：

* `ulysses/sp/ep` 一律关闭 ⇒ 那些 `if get_parallel_state().xxx_enabled:` 分支永远不走；
* 真 veomni 的 `parallel_state.py` 有 769 行（建 device mesh、HP 分片组、各种 rank 换算），
  单进程一行都用不到，而且它会把 `torch.distributed` 拖进来。

所以这里给一个**只回答"全部关闭"**的桩。它的字段名与上游逐个对齐（算子只读这 6 个
属性，见下），改名就会 AttributeError —— 响亮，不会静默。

⚠ 这个桩的语义是"**单进程**"。要在 model_impl 里做多进程/序列并行，必须换回真实现，
不能在这里加逻辑 —— 否则就是在重写一套没人验证过的并行状态。
"""

from __future__ import annotations

import logging as _logging
import os
import sys


# ════════════════════════════════════════════════════════════════════════════
#  日志（算子只用 get_logger）
# ════════════════════════════════════════════════════════════════════════════
_FORMAT = "[vllm-omni-glm-image][%(levelname)s][%(name)s:%(lineno)d] %(asctime)s >> %(message)s"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = _logging.StreamHandler(sys.stdout)
    handler.setFormatter(_logging.Formatter(_FORMAT, datefmt="%m/%d/%Y %H:%M:%S"))
    root = _logging.getLogger("vllm_omni.glm_image_moe")
    root.addHandler(handler)
    root.setLevel(os.environ.get("VLLM_OMNI_GLM_IMAGE_LOG_LEVEL", "INFO").upper())
    root.propagate = False
    _configured = True


class _Logger(_logging.Logger):
    """补上 veomni 那套 `*_rank0` 方法 —— 被复制进来的代码到处在用。

    单进程下 rank0 就是自己，所以这些方法等于普通日志；`warning_once` 按消息去重。
    """

    _seen: set = set()

    def info_rank0(self, message, *args, **kwargs):
        self.info(message, *args, **kwargs)

    def warning_rank0(self, message, *args, **kwargs):
        self.warning(message, *args, **kwargs)

    def debug_rank0(self, message, *args, **kwargs):
        self.debug(message, *args, **kwargs)

    def warning_once(self, message, *args, **kwargs):
        if message not in _Logger._seen:
            _Logger._seen.add(message)
            self.warning(message, *args, **kwargs)


def get_logger(name: str | None = None) -> _Logger:
    _configure()
    _logging.setLoggerClass(_Logger)
    try:
        logger = _logging.getLogger(f"vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.{name}" if name else "vllm_omni.glm_image_moe")
    finally:
        _logging.setLoggerClass(_logging.Logger)
    if not isinstance(logger, _Logger):          # 已经被别处创建过
        for method in ("info_rank0", "warning_rank0", "debug_rank0", "warning_once"):
            if not hasattr(logger, method):
                setattr(logger, method, getattr(_Logger, method).__get__(logger))
    return logger  # type: ignore[return-value]


# ════════════════════════════════════════════════════════════════════════════
#  设备
# ════════════════════════════════════════════════════════════════════════════
def get_device_type() -> str:
    """"npu" / "cuda" / "cpu"。与上游 `veomni/utils/device.py` 同口径。"""
    override = os.environ.get("VEOMNI_DEVICE_TYPE") or os.environ.get("VLLM_OMNI_GLM_IMAGE_DEVICE_TYPE")
    if override:
        return override
    import torch

    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def install_torch_npu_guard() -> bool:
    """非昇腾设备上，让生成文件里**模块级**的 `import torch_npu` 不至于 ImportError。

    为什么需要
    ----------
    `modeling_hy_text.py:361`（从 veomni 的 `glm4_moe_lite/npu_patch.py` 逐字内联进来的
    那一段）、`ops/group_gemm/kernel/npu_group_gemm.py:16`、
    `ops/npu_patch/npu_fused_operator.py:18` 都是**模块级** `import torch_npu`。
    在 veomni 里这几个模块只在 NPU 上被 import；内联之后它们落在**always-on** 的
    import 链上（loading → modeling_hy → modeling_hy_vision → modeling_hy_text），
    于是 GPU 上 `import vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.modeling_hy` 直接 ModuleNotFoundError。

    而这三个文件是**生成产物**，不能手改（手改会被下次重新生成覆盖，而"逐字节复制"
    是数值不可能漂的全部依据）。所以这里在**手写**代码里处理：非 NPU 设备上塞一个
    占位模块进 `sys.modules`，让那三行 import 成功。

    为什么不是"静默打桩"
    --------------------
    占位模块**任何属性访问都会响亮 raise**（`torch_npu.npu_grouped_matmul` 之类），
    所以它只让 import 通过，不会让某条 NPU 专属算子路径在 GPU 上悄悄跑出错结果 ——
    GPU 上这三个文件里的函数本来就一行都不该执行（MoE 走 sonic、rope/attn 走 GPU 实现）。
    真走到了，报错会直接说"这是 NPU 专属实现，设备判成了 X"。

    NPU 上这个函数**什么都不做**（真的 torch_npu 在，`get_device_type()=="npu"`），
    所以已验证的 NPU 口径一个字节都不变。

    返回：是否装了占位模块。
    """
    if "torch_npu" in sys.modules:
        return False
    device_type = get_device_type()
    if device_type == "npu":
        return False
    import importlib.util

    try:
        if importlib.util.find_spec("torch_npu") is not None:
            return False                      # 环境里真有 torch_npu，用真的
    except (ImportError, ValueError):
        pass

    import types

    def _explode(name: str):
        raise RuntimeError(
            f"torch_npu.{name} 被访问了，但这台机器上没有 torch_npu"
            f"（设备判成 {device_type!r}）。\n"
            "  这是**昇腾专属**实现（npu_patch / npu_group_gemm / npu_fused_operator 里的"
            "那几个函数）。GPU 上它们一行都不该执行：MoE 应当走 sonic、rope/attention 走"
            "GPU 实现。\n"
            "  所以这说明某条 NPU 分支真的被选上了 —— 先查 moe_implementation / "
            "attn_implementation 解析成了什么，以及 get_device_type() 为什么是这个值。"
            "不要在这里加回退实现。")

    class _NpuGuard(types.ModuleType):
        __glm_image_guard__ = True

        def __getattr__(self, name):           # 只在正常查找失败时才走到
            # dunder 探测要按"没有这个属性"回答，**不能** raise RuntimeError：
            # `inspect.getmodule()` 会遍历 sys.modules 并对每个模块做
            # `hasattr(module, "__file__")`（torch 的 custom_op 注册路径上就有，
            # flash_attn 一 import 就触发）。RuntimeError 不会被 hasattr 吞掉 ⇒
            # 一个与 NPU 毫不相干的 import 会炸在这里。AttributeError 才是正确语义。
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            _explode(name)

    module = _NpuGuard("torch_npu")
    # 故意**不设** `__file__`：上面的 __getattr__ 对 dunder 抛 AttributeError
    # ⇒ `hasattr(module, "__file__")` 直接 False，inspect 会跳过这个模块。
    # （设成 None 反而更糟：`inspect.getabsfile()` 会拿它去 os.path.abspath(None)。）
    #
    # `__spec__` 则**必须**设：`importlib.util.find_spec("torch_npu")` 看到
    # sys.modules 里有这个名字就直接读 `module.__spec__`，是 None 就
    # `raise ValueError("torch_npu.__spec__ is None")`。accelerate 的
    # `is_npu_available()`（`utils/imports.py`）和 transformers 的
    # `is_torch_npu_available()` 都走这条路，而 accelerate 的那个在
    # `set_module_tensor_to_device()` 里 ⇒ 不设就崩在 dispatch_model。
    # 设上之后它们继续判 `hasattr(torch, "npu")` ⇒ GPU 上仍然是 False，口径不变。
    import importlib.machinery

    module.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    module.__doc__ = "model_impl 的占位：非昇腾设备上让模块级 `import torch_npu` 通过，访问即 raise。"
    sys.modules["torch_npu"] = module
    return True


def empty_cache() -> None:
    import torch

    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
#  并行状态的桩（单进程 = 全部关闭）
# ════════════════════════════════════════════════════════════════════════════
class _SingleProcessParallelState:
    """算子只读这 6 个属性（grep 过 `model_impl/ops/`）：

        ulysses_enabled / ulysses_size / ulysses_group
        sp_enabled / ep_enabled / ep_group

    其余 rank/world 相关的也一并给出，取单进程的值 —— 被复制进来的 modeling 代码
    里有防御式读取（`getattr(parallel_state, "dp_rank", 0)` 之类）。
    """

    # ── 序列并行（Ulysses / CP）────────────────────────────────────────────
    ulysses_enabled = False
    ulysses_size = 1
    ulysses_group = None
    sp_enabled = False
    sp_size = 1
    sp_group = None
    # ── 专家并行 ──────────────────────────────────────────────────────────
    ep_enabled = False
    ep_size = 1
    ep_group = None
    # ── 数据并行 / 分片（单进程全是 1）────────────────────────────────────
    dp_enabled = False
    dp_size = 1
    dp_rank = 0
    dp_group = None
    dp_shard_enabled = False
    dp_shard_size = 1
    dp_shard_rank = 0
    dp_replicate_size = 1
    dp_replicate_rank = 0
    param_shard_size = 1
    param_shard_rank = 0
    param_shard_group = None
    # ── 流水并行 ──────────────────────────────────────────────────────────
    pp_enabled = False
    pp_size = 1
    pp_rank = 0
    # ── 全局 ──────────────────────────────────────────────────────────────
    world_size = 1
    global_rank = 0
    local_rank = 0

    def __repr__(self) -> str:
        return "<model_impl 单进程并行状态：ulysses/sp/ep/dp 全部关闭>"


_STATE = _SingleProcessParallelState()


def get_parallel_state() -> _SingleProcessParallelState:
    return _STATE


# ════════════════════════════════════════════════════════════════════════════
#  序列并行的两个函数（单进程走不到，留下来是因为算子**模块级** import 它们）
# ════════════════════════════════════════════════════════════════════════════
def gather_seq_scatter_heads(tensor, *args, **kwargs):
    raise RuntimeError(
        "gather_seq_scatter_heads 被调用了 —— 说明有序列并行分支真的执行了，"
        "而 model_impl 是单进程。要用 Ulysses 请换回 veomni 的实现，别在桩里加逻辑。")


def gather_heads_scatter_seq(tensor, *args, **kwargs):
    raise RuntimeError(
        "gather_heads_scatter_seq 被调用了 —— 同上，model_impl 是单进程。")


def reduce_sequence_parallel_loss(loss, *args, **kwargs):
    # loss 只在训练用；单进程下原样返回就是对的
    return loss


# ════════════════════════════════════════════════════════════════════════════
#  可选依赖探测（上游 `veomni/utils/import_utils.py` 的最小子集）
# ════════════════════════════════════════════════════════════════════════════
def _available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def is_torch_npu_available() -> bool:
    import torch

    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


def is_triton_available() -> bool:
    return _available("triton")


def is_liger_kernel_available() -> bool:
    return _available("liger_kernel")


def is_seed_kernels_available() -> bool:
    return _available("seed_kernels")


def is_fused_moe_available() -> bool:
    """逐字照上游 `veomni/utils/import_utils.py:85-88`：**CUDA 且有 triton**，
    再加一条 model_impl 自己的前提：**那个 triton 实现有没有被 vendoring 进来**。

    ⚠ 注意它要 `torch.cuda.is_available()` —— 所以在 NPU 上恒为 False，
    fused_moe 会走 npu 那条分支。别"顺手改成也认 NPU"：那会把 MoE 切到
    从没在这条路上验证过的 triton 实现上。

    为什么要加 `_available("vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.fused_moe.group_gemm")`
    -------------------------------------------------------
    `ops/fused_moe/__init__.py:apply_veomni_fused_moe_patch()` 在**模块级**被调用
    （`modeling_hy_text.py:1574`），非 NPU 分支里 `from .group_gemm import
    group_gemm_fused_moe_forward` —— 而 `group_gemm.py` **没有被导出进 model_impl**
    （`ops/fused_moe/` 里只有 npu_group_gemm / npu_mixed_unpermute / sonic_moe）。
    于是 GPU 上 `import vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.modeling_hy` 直接 ModuleNotFoundError。

    GPU 上的推理口径是 `moe_implementation=sonic` ⇒ 专家走
    `Glm4MoeLiteExperts.sonic_forward` → `sonic_moe_forward`，**根本不碰**
    `fused_moe_forward` / `_fused_moe_forward`（那是 `fused` 那条路用的）。
    所以这里让它判 False、`_fused_moe_forward = None` 是**这条路上正确的结果**，
    不是绕过检查：真有谁走到 fused 那条路，`fused_moe_forward()` 第一行就
    `raise NotImplementedError` —— 响亮，不会静默出错图。

    NPU 上这个条件**不改变任何行为**（`torch.cuda.is_available()` 已经是 False）。
    根治要回生成器侧：要么把 `group_gemm.py` 一起导出，要么让那个 import 变成惰性。
    """
    import torch

    return (
        torch.cuda.is_available()
        and is_triton_available()
        and _available("vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.ops.fused_moe.group_gemm")
    )


def is_flash_attn_available() -> bool:
    return _available("flash_attn_interface") or _available("flash_attn")


#: `veomni/data/constants.py` 里 ops/loss.py 唯一用到的常量
IGNORE_INDEX = -100

__all__ = [
    "get_logger", "get_device_type", "empty_cache",
    "get_parallel_state", "gather_seq_scatter_heads", "gather_heads_scatter_seq",
    "reduce_sequence_parallel_loss",
    "is_torch_npu_available", "is_triton_available", "is_liger_kernel_available",
    "is_seed_kernels_available", "is_flash_attn_available", "is_fused_moe_available",
    "transfusion_cp_manual", "wait_parameter", "all_to_all", "sort_chunks_by_idxs",
    "attention_checkpoint_kwargs", "attention_checkpoint_contexts",
    "_eval_only_placeholder", "get_cache_dir", "get_dtype_size",
    "IGNORE_INDEX",
]


def transfusion_cp_manual() -> bool:
    """上游 `sequence_parallel/transfusion_cp.py` 的同名函数：训练脚本有没有显式
    `enable_transfusion_cp()`。单进程推理从不开，恒 False ⇒ 调用方的 CP 分支全部短路。
    """
    return False


# ════════════════════════════════════════════════════════════════════════════
#  MoE / FSDP 相关的桩（`ops/fused_moe/npu_group_gemm.py` 要，单进程走不到）
# ════════════════════════════════════════════════════════════════════════════
def wait_parameter(parameter) -> None:
    """逐字等价于 `veomni/distributed/fsdp_direct_gather.py:21-27`。

    它等的是"专家权重的延迟 all-gather 事件"，而那个事件由 FSDP 的 direct-gather 挂在
    参数上（`_veomni_gather_ready`）。model_impl 不用 FSDP ⇒ 属性不存在 ⇒ 原实现本来就是
    空操作。这里保留同样的语义（而不是直接 `pass`），万一将来参数真带了事件也能等上。
    """
    if parameter is None:
        return
    event = getattr(parameter, "_veomni_gather_ready", None)
    if event is not None:
        import torch

        torch.npu.current_stream().wait_event(event)


def all_to_all(group, tensor, output_splits=None, input_splits=None):
    raise RuntimeError(
        "all_to_all 被调用了 —— 说明专家并行（ep）分支真的执行了，而 model_impl 是单进程"
        "（base 的并行桩里 ep_enabled=False）。要用 EP 请换回 veomni 的实现。")


def sort_chunks_by_idxs(*args, **kwargs):
    raise RuntimeError("sort_chunks_by_idxs 被调用了 —— 同上，只在 ep 路径用到。")


# ════════════════════════════════════════════════════════════════════════════
#  梯度检查点策略（逐字抄自 `veomni/optimizations/hy/attention_checkpoint.py`，30 行）
# ════════════════════════════════════════════════════════════════════════════
# 它只在**训练**的 gradient checkpointing 里起作用（让 fused attention 的结果活到
# backward）。推理 `torch.no_grad()` 下 `attention_checkpoint_kwargs()` 返回 {}，
# 调用方 `torch.utils.checkpoint.checkpoint(..., **kwargs)` 那条分支也不执行。
# 逐字搬进来是因为 `modeling_hy.py` 是**模块级** import 它。
def _attention_policy(context, operation, *args, **kwargs):
    from torch.utils.checkpoint import CheckpointPolicy

    if operation._schema.name in {
        "npu::npu_fusion_attention",
        "npu::npu_fusion_attention_v2",
    }:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def attention_checkpoint_contexts():
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(_attention_policy)


def attention_checkpoint_kwargs():
    if os.environ.get("VEOMNI_HY_SAVE_ATTN", "0") != "1":
        return {}
    return {"context_fn": attention_checkpoint_contexts}


# ════════════════════════════════════════════════════════════════════════════
#  "只做 eval"的占位符
# ════════════════════════════════════════════════════════════════════════════
def _eval_only_placeholder(name: str, what: str, dataclass_like: bool = False):
    """造一个占位类：被实例化或调用时**响亮地** raise，说清为什么没有它。

    用在哪：`pipeline.py` 的 `make_transform_factory` 要在签名和分支里提到一堆
    训练数据源的 transform（ffrecord / lmdb / parquet / understanding / glm_binary），
    而 eval 只走 `wds_image_with_meta_json` 一个分支（用户 2026-11-17 的约束：
    只做 eval 的适配）。

    为什么用占位而不是搬进来：搬了就等于把训练数据管线也带进 model_impl（2015+808+395… 行），
    而且会让"这个仓库只做 eval"这件事在代码上看不出来。占位则是：
    **走到就炸，而且错误信息直接告诉你走到了哪条训练路径。**
    """
    message = (f"model_impl 不包含 {name}（{what}）—— 这个推理仓库只做 eval，"
               f"训练数据管线没有搬进来。如果真需要它，去 veomni 跑训练那条路。")

    class _Placeholder:
        __name__ = name

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(message)

        def __call__(self, *args, **kwargs):
            raise NotImplementedError(message)

        def __repr__(self):
            return f"<model_impl 占位: {name}（eval 不需要）>"

    _Placeholder.__name__ = name
    _Placeholder.__qualname__ = name
    if dataclass_like:
        # 有些地方只是把它当类型注解/默认 None 用，不实例化 —— 那就别在构造时炸
        class _TypeOnly:
            __name__ = name

            def __repr__(self):
                return f"<model_impl 占位类型: {name}>"

        _TypeOnly.__name__ = name
        _TypeOnly.__qualname__ = name
        return _TypeOnly
    return _Placeholder


# ════════════════════════════════════════════════════════════════════════════
#  权重加载要的两个小工具（逐字抄自 `veomni/utils/helper.py:580-620`）
# ════════════════════════════════════════════════════════════════════════════
# 为什么抄而不是搬整个 helper.py：那文件 830 行，绝大部分是训练期工具
# （FlopsCounter、step2token、hdfs 上传…），eval 只要这两个。
from functools import lru_cache  # noqa: E402
from typing import Optional  # noqa: E402  （抄来的 get_cache_dir 签名里用到）

# ⚠ 这里要模块级 `torch`：`get_dtype_size` 的函数体（逐字抄来的）直接用 `torch.int64` 等。
#   base.py 其余部分刻意把 torch 放函数内 import（让 `import model_impl` 很轻），
#   但为了不改抄来的函数体，这里破例。
import torch  # noqa: E402

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "veomni")))

def get_cache_dir(path: Optional[str] = None) -> str:
    """
    Returns the cache directory for the given path.
    """
    if path is None:
        return CACHE_DIR

    path = os.path.normpath(path)
    if not os.path.splitext(path)[-1]:  # is a dir
        path = os.path.join(path, "")

    path = os.path.split(os.path.dirname(path))[-1]
    return os.path.join(CACHE_DIR, path, "")  # must endswith os.path.sep


@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L350
    """
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]
