"""AR stage model for GLM-Image HY/MoE.

The transformer layers are the standard vLLM GLM4-MoE-lite implementation.
The GLM-Image checkpoint additionally contains diffusion and visual-VQ
parameters; this adapter filters those tensors so the AR stage can consume the
shared checkpoint without duplicating weights on disk.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteForCausalLM


class GlmImageMoeTransfusionForCausalLM(Glm4MoeLiteForCausalLM):
    """GLM4-MoE-lite causal LM over the shared GLM-Image checkpoint."""

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

        return super().load_weights(expand_ar_weights())
