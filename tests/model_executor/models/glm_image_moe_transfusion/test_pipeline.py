from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.glm_image_moe_transfusion.pipeline import (
    GLM_IMAGE_MOE_TRANSFUSION_PIPELINE,
    GLM_IMAGE_MOE_TRANSFUSION_THINK_PIPELINE,
)


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
