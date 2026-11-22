import inspect

import pytest

from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.glm_image_moe_transfusion.ar_model import (
    GlmImageMoeTransfusionForCausalLM,
)
from vllm_omni.model_executor.models.glm_image_moe_transfusion.checkpoint import (
    resolve_ar_model_root,
)
from vllm_omni.model_executor.models.glm_image_moe_transfusion.pipeline import (
    GLM_IMAGE_MOE_TRANSFUSION_PIPELINE,
    GLM_IMAGE_MOE_TRANSFUSION_THINK_PIPELINE,
)
from vllm_omni.model_extras import build_text_to_image_tokenization


def test_direct_pipeline_is_single_diffusion_stage() -> None:
    assert len(GLM_IMAGE_MOE_TRANSFUSION_PIPELINE.stages) == 1
    stage = GLM_IMAGE_MOE_TRANSFUSION_PIPELINE.stages[0]
    assert stage.execution_type is StageExecutionType.DIFFUSION
    assert stage.final_output_type == "image"


def test_thinking_pipeline_has_ar_then_diffusion() -> None:
    ar, diffusion = GLM_IMAGE_MOE_TRANSFUSION_THINK_PIPELINE.stages
    assert ar.execution_type is StageExecutionType.LLM_AR
    assert ar.engine_output_type == "token_ids"
    assert diffusion.execution_type is StageExecutionType.DIFFUSION
    assert diffusion.input_sources == (0,)


def test_ar_model_root_resolves_sibling_view(tmp_path) -> None:
    diffusion = tmp_path / "diffusion"
    ar_view = tmp_path / "ar_model"
    diffusion.mkdir()
    ar_view.mkdir()
    (ar_view / "config.json").write_text("{}")

    assert resolve_ar_model_root(str(diffusion), None, None) == str(ar_view)


def test_ar_forward_accepts_omni_model_intermediate_buffer() -> None:
    parameters = inspect.signature(GlmImageMoeTransfusionForCausalLM.forward).parameters

    assert "model_intermediate_buffer" in parameters
    assert "runtime_additional_information" in parameters
    assert "seq_token_counts" in parameters
    assert "sampling_metadata" in parameters
    assert "logits_index" in parameters
    assert "sampler" in parameters


def test_text_to_image_tokenization_uses_native_ar_prefix() -> None:
    expected_tokens = {
        154824: "<sop>",
        154825: "<eop>",
        154827: "<|user|>",
        154828: "<|assistant|>",
        154830: "<|begin_of_image|>",
    }

    class Tokenizer:
        vocab_size = 154880

        def __call__(self, text: str, add_special_tokens: bool = False):
            assert text.startswith("Generate an image")
            assert add_special_tokens is False
            return {"input_ids": [10, 11, 12]}

        def convert_ids_to_tokens(self, token_id: int):
            return expected_tokens.get(token_id)

    tokenizer = Tokenizer()

    for model_class_name in (
        "GlmImageMoeTransfusionForCausalLM",
        "GlmImageMoeTransfusionPipeline",
    ):
        token_ids = build_text_to_image_tokenization(
            model_class_name,
            "a shiba inu",
            tokenizer,
            height=128,
            width=256,
        )["prompt_token_ids"]
        assert token_ids == [154827, 154857, 10, 11, 12, 154828]


def test_text_to_image_tokenization_is_opt_in() -> None:
    assert (
        build_text_to_image_tokenization(
            "UnrelatedImagePipeline",
            "a shiba inu",
            object(),
            height=128,
            width=128,
        )
        == {}
    )


def test_text_to_image_tokenization_rejects_tokenizer_id_drift() -> None:
    class Tokenizer:
        vocab_size = 154880

        def convert_ids_to_tokens(self, token_id: int):
            return "wrong-token"

    with pytest.raises(ValueError, match="special-token mismatch"):
        build_text_to_image_tokenization(
            "GlmImageMoeTransfusionPipeline",
            "a shiba inu",
            Tokenizer(),
            height=128,
            width=128,
        )
