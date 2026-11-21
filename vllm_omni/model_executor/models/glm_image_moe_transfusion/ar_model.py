"""AR stage model for GLM-Image HY/MoE.

The transformer layers are the standard vLLM GLM4-MoE-lite implementation.
The GLM-Image checkpoint additionally contains diffusion and visual-VQ
parameters; this adapter filters those tensors so the AR stage can consume the
shared checkpoint without duplicating weights on disk.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MethodType
from typing import Any

import torch
from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteForCausalLM


def _is_ascend_a2() -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return False
    device_name = npu.get_device_name(npu.current_device()).upper()
    return "910B" in device_name


def _disable_mla_prolog_v3(model: torch.nn.Module) -> None:
    """Use the legacy MLA preprocessing that A2 can execute.

    vLLM-Ascend's PROLOG_V3 path converts ``weight_uq_qr`` to FRACTAL_NZ
    during weight loading.  On A2/CANN 9.2 the required TransData kernel is
    unavailable for this GLM-Image MLA geometry.  Apply the fallback only to
    this adapter's own attention modules so other GLM-Image variants and
    models retain their platform default.
    """
    if not _is_ascend_a2():
        return

    for module in model.modules():
        impl: Any = getattr(module, "impl", None)
        if not callable(getattr(impl, "supports_prolog_v3_writer", None)):
            continue
        impl.supports_prolog_v3_writer = MethodType(lambda _self: False, impl)


class GlmImageMoeTransfusionForCausalLM(Glm4MoeLiteForCausalLM):
    """GLM4-MoE-lite causal LM over the shared GLM-Image checkpoint."""

    supports_mrope = True

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        """Return text-only M-RoPE positions for the AR stage.

        GLM-Image uses three M-RoPE channels.  Like the Qwen-family contract,
        text-only inputs repeat the ordinary token positions on all channels.
        The native AR stage does not consume visual features; those belong to
        the diffusion/conditioning stage.
        """
        if mm_features:
            raise ValueError(
                "GlmImageMoeTransfusionForCausalLM AR stage only accepts text; "
                "visual features must be handled by the diffusion stage."
            )
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    @staticmethod
    def _is_ar_weight(name: str) -> bool:
        if name.endswith(".mlp.gate.local_tokens_per_expert"):
            return False
        if name.endswith(".mlp.gate.local_vision_tokens_per_expert"):
            return False
        if name in {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}:
            return True
        return name.startswith("model.layers.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def expand_ar_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if not self._is_ar_weight(name):
                    continue
                if name.endswith(".mlp.experts.gate_up_proj"):
                    base = name[: -len(".gate_up_proj")]
                    gate, up = tensor.chunk(2, dim=1)
                    for expert_id in range(tensor.shape[0]):
                        yield f"{base}.{expert_id}.gate_proj.weight", gate[expert_id].contiguous()
                        yield f"{base}.{expert_id}.up_proj.weight", up[expert_id].contiguous()
                    continue
                if name.endswith(".mlp.experts.down_proj"):
                    base = name[: -len(".down_proj")]
                    for expert_id in range(tensor.shape[0]):
                        yield f"{base}.{expert_id}.down_proj.weight", tensor[expert_id].contiguous()
                    continue
                yield name, tensor

        loaded = super().load_weights(expand_ar_weights())
        _disable_mla_prolog_v3(self)
        return loaded
