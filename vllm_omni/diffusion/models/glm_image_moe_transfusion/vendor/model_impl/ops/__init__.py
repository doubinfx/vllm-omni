"""model_impl 的算子层。

这些是**自定义 CANN / triton 算子的封装**，不是模型结构，所以保持独立模块
（transformers 也把第三方 kernel 放 `integrations/` 而不是塞进 modeling 文件）。

⚠ blockmask attention 与 rope **没有纯 torch 实现**（`_eager_forward` 是空函数、
`apply_rotary_emb_fused` 默认 raise），必须有 patched CANN 算子包才能跑 ——
见 model_impl/README.md「为什么必须经 scripts/run.sh 起」。
"""

from .attention import flash_attention_forward
from .block_mask_fa3 import fused_block_mask_attn
from .fused_moe import fused_moe_forward
from .loss import causallm_loss_function


__all__ = [
    "flash_attention_forward",
    "fused_block_mask_attn",
    "fused_moe_forward",
    "causallm_loss_function",
]
