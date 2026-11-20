"""【自动生成，勿手改】权重加载 —— 只有 `init_empty_weights` / `load_model_weights` 的闭包。

从 `veomni/models/module_utils.py`（830 行）按依赖闭包抽出 12 个定义 / 351 行。
裁掉的是 `save_model_weights` / `save_model_assets` / `rank0_load_and_broadcast_weights`
等**保存和训练期**的函数 —— eval 只读权重。

⚠ `_ExpertWeightMerger` 必须在：它把 ckpt 里的 `experts.gate_proj`/`up_proj` 拼成模型要的
`experts.gate_up_proj`（HY 原生吃 fused）。漏了会是"键缺失"而不是静默，但仍别动它。
"""

from __future__ import annotations

# ── import 区自动筛出 ──
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union
import torch
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
# ⚠ safetensors 的 import 在上游是**条件块**（`if transformers.__version__ == "5.0.0": ... else:`），
#   AST 搬运只扫顶层 Import 节点会漏掉它 ⇒ 运行到 `StateDictIterator.__iter__` 才 NameError
#   （2026-11-17 踩过）。这里逐字保留上游那段条件逻辑。
import transformers

if transformers.__version__ == "5.0.0":
    from safetensors import safe_open
    from safetensors.torch import save_file  # noqa: F401
else:
    try:
        from transformers.utils.import_utils import is_safetensors_available
    except ImportError:
        # transformers >=5 removed this helper; safetensors is a hard
        # dependency of the vLLM-Omni runtime, so the capability is present.
        def is_safetensors_available():
            return True

    if is_safetensors_available():
        from safetensors import safe_open
        from safetensors.torch import save_file  # noqa: F401

from . import base as logging
from .base import empty_cache, get_cache_dir, get_dtype_size, get_parallel_state

logger = logging.get_logger(__name__)

# ── veomni/models/module_utils.py:72-103 ──
@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.

    Borrowed from: https://github.com/huggingface/accelerate/blob/v1.0.0rc1/src/accelerate/big_modeling.py#L57
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module: "nn.Module", name: str, param: "nn.Parameter"):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            # When we have a case of tensor2 = tensor1, it would call the set_attr
            # of param, which in turn would call the register_parameter API.
            # In this case, the new param is already on meta-device, since it was moved
            # previously when it was initialized. Hence, when resetting, you can
            # directly assign that tensor instead of re-init. If you re-init you would
            # lose the relationship.
            module._parameters[name] = (
                param
                if param.device == torch.device("meta")
                else param_cls(module._parameters[name].to("meta"), **kwargs)
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


# ── veomni/models/module_utils.py:106-119 ──
@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


# ── veomni/models/module_utils.py:130-162 ──
def _load_state_dict(weights_path: str, **kwargs) -> List["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


# ── veomni/models/module_utils.py:165-176 ──
def _find_submodule(module: "nn.Module", name: str) -> Tuple["nn.Module", str]:
    """
    Finds the leaf module according to the name.
    """
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        module = getattr(module, piece)

    return module, pieces[-1]


# ── veomni/models/module_utils.py:179-217 ──
def _dispatch_parameter(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    parallel_plan: Optional["ParallelPlan"] = None,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    full_param_name = name
    module, local_name = _find_submodule(module, name)
    target = module._parameters[local_name]
    orig_tensor = target.data

    # Handle parameter slicing according to parallel_plan, now only EP-aware
    if parallel_plan is not None:
        # HP Parameter.shape is global, while Parameter.data.shape is local.
        # Passing the latter would mistake FSDP shards for EP shards and slice twice.
        tensor = parallel_plan.shard_tensor(tensor, full_param_name, target.shape)

    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._parameters[local_name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        # HyperParallel exposes a plain local tensor through Parameter.data;
        # its mesh/placements live on the Parameter itself (as in zwd's loader).
        if hasattr(target, "device_mesh"):
            if dtensor_factory is None:
                raise ValueError("Loading a sharded parameter requires a dtensor_factory.")
            tensor = dtensor_factory(tensor, target.device_mesh, target.placements).to_local()
        module._parameters[local_name].data.copy_(tensor)


# ── veomni/models/module_utils.py:220-240 ──
def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name]

    if hasattr(orig_tensor, "device_mesh"):  # dtensor buffer
        if dtensor_factory is None:
            raise ValueError("dtensor buffer requires a dtensor_factory.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._buffers[name] = dtensor_factory(buffer.to(dtype=orig_tensor.dtype), device_mesh, placements)
    else:
        module._buffers[name].copy_(buffer.to(device=orig_tensor.device, dtype=orig_tensor.dtype))


# ── veomni/models/module_utils.py:243-275 ──
def _init_parameter(
    module: "nn.Module",
    name: str,
) -> None:
    """
    Initializes parameter in model.
    """
    pieces = name.split(".")
    init_func = None
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        if hasattr(module, "_init_weights"):
            init_func = getattr(module, "_init_weights")

        module = getattr(module, piece)

    if init_func is None:
        raise ValueError(f"Cannot retrieve `_init_weights` function in the parents of {module}.")

    hp_parameters = any(
        hasattr(param, "to_local") and not hasattr(param.data, "device_mesh") for param in module.parameters()
    )
    if hp_parameters:
        # Missing HF keys still need initialization. HP's distributed dispatch
        # lacks some init operators; initialize the persistent local shards.
        from hyper_parallel import SkipDTensorDispatch

        with SkipDTensorDispatch():
            module.apply(init_func)
    else:
        module.apply(init_func)


# ── veomni/models/module_utils.py:278-298 ──
def _convert_weight_key(key: str, model: "PreTrainedModel") -> str:
    """
    Convert a single state dict key using the model's checkpoint conversion mapping.

    For example, in the InternVL, we have _checkpoint_conversion_mapping = {"^model": "language_model"}

    This is to adapt to the big breaking change introduced in HF transformers 4.52:
    https://github.com/huggingface/transformers/pull/38385
    """
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return key

    for pattern, replacement in model._checkpoint_conversion_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        converted_key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return converted_key

    return key


# ── veomni/models/module_utils.py:302-325 ──
class _ExpertWeightMerger:
    """Join legacy HF gate/up halves only when the model has a fused target."""

    def __init__(self, parameter_names):
        self.targets = {name for name in parameter_names if name.endswith(".experts.gate_up_proj")}
        self.pending = {}

    def merge(self, name, tensor):
        for suffix, half in ((".experts.gate_proj", "gate"), (".experts.up_proj", "up")):
            if name.endswith(suffix):
                target = name[:-len(suffix)] + ".experts.gate_up_proj"
                if target in self.targets:
                    parts = self.pending.setdefault(target, {})
                    parts[half] = tensor.detach().clone()
                    if len(parts) < 2:
                        return None, None
                    value = torch.cat([parts["gate"], parts["up"]], dim=1)
                    del self.pending[target]
                    return target, value
        return name, tensor

    def finish(self):
        if self.pending:
            raise ValueError(f"Incomplete HF expert gate/up pairs: {sorted(self.pending)}")


# ── veomni/models/module_utils.py:327-395 ──
@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    expert_merger = _ExpertWeightMerger(parameter_names_to_load)
    model.to_empty(device=init_device)

    # Get parallel plan if available
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    unexpected_keys = []
    state_dict_iterators = _load_state_dict(weights_path)
    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for name, tensor in state_dict_iterator:
            # IMPORTANT: Call this function to adapt to transformers 4.52 breaking change
            # on model structure. See the comment for details.
            name = _convert_weight_key(name, model)
            name, tensor = expert_merger.merge(name, tensor)
            if name is None:
                continue

            if name in buffer_dict.keys():  # persistent buffers
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names_to_load:
                parameter_names_to_load.remove(name)
                _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan)
            else:
                unexpected_keys.append(name)

        del state_dict_iterator
        empty_cache()

    expert_merger.finish()

    # ── Explicit summary of weight loading status ──
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        if unexpected_keys:
            logger.warning_rank0(
                f"\n⚠ WEIGHT LOADING: {len(unexpected_keys)} UNEXPECTED key(s) in checkpoint (ignored):"
            )
            for k in sorted(unexpected_keys):
                logger.info_rank0(f"    {k}")
        if parameter_names_to_load:
            logger.warning_rank0(
                f"\n⚠ WEIGHT LOADING: {len(parameter_names_to_load)} MISSING key(s) (randomly initialized):"
            )
            for k in sorted(parameter_names_to_load):
                logger.info_rank0(f"    {k}")
        if not unexpected_keys and not parameter_names_to_load:
            logger.info_rank0("✓ WEIGHT LOADING: All model parameters loaded successfully (0 missing, 0 unexpected).")
        else:
            logger.warning_rank0(
                f"  WEIGHT LOADING SUMMARY: {len(parameter_names_to_load)} missing, {len(unexpected_keys)} unexpected"
            )

    parameter_names_to_load = sorted(parameter_names_to_load)  # 保证多次运行的 init 顺序一致和曲线的可复现性
    post_process_after_weight_loading(model, buffer_dict, parameter_names_to_load, dtensor_factory)


# ── veomni/models/module_utils.py:549-572 ──
@torch.no_grad()
def _broadcast_params_dp_replicate(model: "nn.Module", names) -> None:
    """把指定参数(通常是 checkpoint 缺失、由 ``_init_parameter`` 各 rank 独立随机初始化的 key)
    沿 ``dp_replicate`` process group 广播 local shard,使 HSDP 各数据并行副本一致。

    ``_init_parameter`` 走 ``module.apply(_init_weights)``,各 rank 用各自 RNG 初始化 local shard →
    HSDP(``dp_replicate_size>1``)下同一逻辑参数的不同副本不一致;FSDP2 只 all-reduce 梯度、假设
    各副本权重一致,于是训练发散。这里在**初始化源头**把一致性显式构造出来(复用模型自身的 init
    分布,含 zero-init)。纯 FSDP(``dp_replicate_size==1``)逻辑全参本就跨 rank 一致,直接返回。
    """
    ps = get_parallel_state()
    if not ps.dp_replicate_enabled:
        return
    group = ps.dp_replicate_group
    src = dist.get_global_rank(group, 0)
    for name in sorted(names):  # 各 rank 迭代顺序须一致以对齐集合通信
        try:
            p = model.get_parameter(name)
        except AttributeError:
            continue
        local = p.to_local() if hasattr(p, "to_local") else p.data
        if local.numel() == 0 or local.is_meta:
            continue
        dist.broadcast(local, src=src, group=group)


# ── veomni/models/module_utils.py:575-607 ──
def post_process_after_weight_loading(
    model: Union["nn.Module", "PreTrainedModel"],
    buffer_dict: dict[str, "torch.Tensor"],
    parameter_names_left: Optional[list[str]] = None,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
):
    """
    shared logic after weight loading that handles buffer, missing weight keys and tied embedding weights.
    """
    parameter_names_left = parameter_names_left or []

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer, dtensor_factory)

    if parameter_names_left:
        logger.info_rank0(f"Find missing key(s) in state dict: {parameter_names_left}, initialize them.")
        for name in parameter_names_left:
            _init_parameter(model, name)
        # missing key 上面各 rank 独立随机初始化 → HSDP dp_replicate 副本间不一致;在源头沿
        # dp_replicate 广播 local shard,把跨副本一致性显式构造出来(与 train 侧 load 后的
        # broadcast_ckpt_state_dp_replicate 形成双保险)。
        _broadcast_params_dp_replicate(model, parameter_names_left)

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
    if getattr(model.config, "tie_word_embeddings", True):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
        except Exception as e:
            logger.info_rank0(f"Failed to tie embeddings: {e}")
            raise RuntimeError("Failed to tie input/output embeddings") from e

