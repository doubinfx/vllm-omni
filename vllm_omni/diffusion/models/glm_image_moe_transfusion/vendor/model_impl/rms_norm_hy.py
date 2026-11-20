# 逐字节搬自 `veomni/models/transfusion/hy/rms_norm.py`。

# 为什么搬：HY 的 affine RMSNorm 取整口径，**数值相关**且建模时必然执行（adapt_backbone 调 configure_hy_rmsnorm）—— 之前 335 个 veomni 模块里头一个就是它触发的
#
# ⚠ 除 import 行（见 gen_extras.py 的 VENDOR_REWRITES），**一个字都没改** ——
# 逐字节复制是"数值不可能漂"的全部依据。重新生成（必须在 gen_* 之后）：
#     bash model_impl/tools/regen.sh

"""Restore the original HY recipe's affine RMSNorm rounding mode."""

from types import MethodType

import torch

from .base import get_logger


logger = get_logger(__name__)


def _eager_round_once(self, hidden_states):
    value = hidden_states.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
    return (self.weight * value).to(hidden_states.dtype)


def configure_hy_rmsnorm(model, *, npu):
    """Use single-rounding affine RMSNorm for HY, preserving the native NPU kernel.

    The source recipe defaults to FP32 normalization and affine multiplication,
    followed by one output cast. Only affine GLM norms need this adjustment;
    no-affine diffusion norms and their AdaLN paths retain their own arithmetic.
    This is fixed model behavior, shared by training and inference.
    """
    from .modeling_hy_text import Glm4MoeLiteRMSNorm

    if npu:
        logger.info_rank0("HY RMSNorm: native NPU kernel preserved")
        return 0

    count = 0
    for module in model.modules():
        if not isinstance(module, Glm4MoeLiteRMSNorm) or module.weight is None:
            continue
        if hasattr(module, "casting_mode"):
            module.casting_mode = "gemma"
        else:
            module.forward = MethodType(_eager_round_once, module)
        count += 1
    logger.info_rank0(f"HY affine RMSNorm: single-rounding FP32 computation, modules={count}")
    return count
