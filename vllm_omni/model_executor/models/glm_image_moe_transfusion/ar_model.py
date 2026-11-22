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
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)

from vllm_omni.model_executor.models.glm_image_moe_transfusion.prompt import (
    build_ar_prefix_ids,
)


class GlmImageMoeProcessingInfo(BaseProcessingInfo):
    """Text-only AR stage exposed through the multimodal serving interface."""

    def get_supported_mm_limits(self) -> dict[str, int]:
        # The serving renderer only constructs an mm_processor for models with
        # at least one declared modality. Requests still carry zero media
        # items; the processor uses mm_processor_kwargs for target dimensions.
        return {"image": 1}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        del seq_len, mm_counts
        return {"image": 0}


class GlmImageMoeDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        del mm_counts
        return ""

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        del seq_len, mm_counts, mm_options
        return {}


class GlmImageMoeMultiModalProcessor(BaseMultiModalProcessor):
    """Tokenize text and let the stage processor hand tokens to diffusion."""

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        del hf_inputs, hf_processor_mm_kwargs
        return {}

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        del mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        return []

    def apply(self, inputs, timing_ctx):
        del timing_ctx
        if any(inputs.mm_data_items.get_all_counts().values()):
            raise ValueError(
                "Native GLM-Image AR currently supports text-to-image only; "
                "reference or source images are not supported."
            )
        tokenizer = self.info.get_tokenizer()
        prompt = inputs.prompt
        if not isinstance(prompt, str):
            prompt = tokenizer.decode(prompt, skip_special_tokens=False)
        processor_kwargs = inputs.hf_processor_mm_kwargs
        height = int(processor_kwargs.get("target_h") or processor_kwargs.get("height") or 1024)
        width = int(processor_kwargs.get("target_w") or processor_kwargs.get("width") or 1024)
        token_ids, _ = build_ar_prefix_ids(tokenizer, prompt, height, width)
        return {
            "type": "multimodal",
            "prompt_token_ids": token_ids,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


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


def _bootstrap_a2_mla_rope_cache(model: torch.nn.Module) -> None:
    """Expose MRoPE tables to the legacy A2 MLA metadata builder."""
    if not _is_ascend_a2():
        return

    for module in model.modules():
        cos_sin_cache = getattr(module, "cos_sin_cache", None)
        mrope_section = getattr(module, "mrope_section", None)
        if cos_sin_cache is None or not mrope_section:
            continue
        cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
        if not module.is_neox_style:
            # GLM-Image uses adjacent rotary pairs.  npu_interleave_rope
            # pair-transposes D before RotateHalf, so its 64-wide tables repeat
            # each half instead of repeating each coefficient twice.
            cos_cache = torch.cat((cos_cache, cos_cache), dim=-1)
            sin_cache = torch.cat((sin_cache, sin_cache), dim=-1)
        from vllm_ascend.ops.rotary_embedding import _record_cos_and_sin_cache

        _record_cos_and_sin_cache(cos_cache, sin_cache)
        return


@MULTIMODAL_REGISTRY.register_processor(
    GlmImageMoeMultiModalProcessor,
    info=GlmImageMoeProcessingInfo,
    dummy_inputs=GlmImageMoeDummyInputsBuilder,
)
class GlmImageMoeTransfusionForCausalLM(
    Glm4MoeLiteForCausalLM,
    SupportsMultiModal,
    SupportsMRoPE,
):
    """GLM4-MoE-lite causal LM over the shared GLM-Image checkpoint."""

    supports_mrope = True

    def embed_multimodal(self, **kwargs):
        """Accept the multimodal interface while carrying no media tensors."""
        del kwargs
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use vLLM's standard multimodal embedding merge interface."""
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model

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
        _bootstrap_a2_mla_rope_cache(self)
        return loaded
