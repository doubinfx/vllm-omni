"""类名别名表 —— 逐字抄自 `veomni/utils/transfusion_compat.py:11-32`。

为什么必须原样保留
------------------
`modular_..._selfflow.py` 底部调 `install_model_aliases(globals())`，它把一批**老名字**
指到新类上。其中有两条是**重定向**，不是单纯起别名：

    GlmImageMoeTransfusionConfig                     → _TransfusionImageConfig
    GlmImageMoeTransfusionForConditionalGeneration   → _TransfusionImageEntryModel

也就是说：selfflow 模块里的 `GlmImageMoeTransfusionConfig` 这个名字，**不是**
`modular_glm_image_moe_transfusion.py` 里那个同名的类，而是 selfflow 自己的
`_TransfusionImageConfig`。`specs/dmd2_lyz/config.json` 的 `architectures` 写的是
`GlmImageMoeTransfusionForConditionalGeneration` —— 它解析到的是 `_TransfusionImageEntryModel`。

把这张表抄错或"顺手简化"，加载时会拿到错的类，而形状/键集可能仍然对得上 ⇒ 静默出错。
"""

from __future__ import annotations

import importlib
import sys


#: 搬到 model_impl 之后，实现模块就是 model_impl.modeling_hy（原来是 veomni 那个 selfflow 模块）
_MODEL_MODULE = "vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.modeling_hy"

#: ⚠ 逐字照抄上游，顺序和内容都别动
_MODEL_ALIASES = {
    "HYConfig": "GlmImageMoeTransfusionSelfFlowBaseConfig",
    "HYSelfFlowConfig": "GlmImageMoeTransfusionSelfFlowConfig",
    "HYModel": "GlmImageMoeTransfusionSelfFlowModel",
    "HYSelfFlowModel": "GlmImageMoeTransfusionSelfFlowForMMGeneration",
    "GlmImageMoeTransfusionConfig": "_TransfusionImageConfig",
    "GlmImageMoeTransfusionForConditionalGeneration": "_TransfusionImageEntryModel",
    "GlmImageMoeTransfusionHYConfig": "_TransfusionConditioningConfig",
    "GlmImageMoeTransfusionHYForConditionalGeneration": "_TransfusionConditionedModel",
    "SelfFlowHYForwardMixin": "TransfusionSelfFlowForwardMixin",
}

_PUBLIC_MODEL_NAMES = ("HYConfig", "HYModel", "HYSelfFlowConfig", "HYSelfFlowModel")


def install_model_aliases(namespace) -> None:
    """Old and new class names refer to the same class, without wrappers."""
    for old, new in _MODEL_ALIASES.items():
        namespace[old] = namespace[new]


def install_legacy_model_module(module_name: str) -> None:
    """Preserve module identity so patches through either import path agree."""
    sys.modules[module_name] = importlib.import_module(_MODEL_MODULE)


def export_legacy_model_api(namespace) -> None:
    implementation = importlib.import_module(_MODEL_MODULE)
    namespace.update({name: getattr(implementation, name) for name in _PUBLIC_MODEL_NAMES})


__all__ = ["install_model_aliases", "install_legacy_model_module", "export_legacy_model_api",
           "_MODEL_ALIASES", "_PUBLIC_MODEL_NAMES"]
