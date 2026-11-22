# SPDX-License-Identifier: Apache-2.0
"""GLM-Image HY/MoE native pipeline topologies."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_DIFFUSION_ARCH = "GlmImageMoeTransfusionPipeline"
_AR_ARCH = "GlmImageMoeTransfusionForCausalLM"
_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.glm_image_moe_transfusion"
_CHECKPOINT = "vllm_omni.model_executor.models.glm_image_moe_transfusion.checkpoint"


def _diffusion_stage(stage_id: int, input_sources: tuple[int, ...]) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=stage_id,
        model_stage="dit",
        execution_type=StageExecutionType.DIFFUSION,
        input_sources=input_sources,
        final_output=True,
        final_output_type="image",
        model_arch=_DIFFUSION_ARCH,
        custom_process_input_func=(
            f"{_PROCESSOR}.ar2diffusion" if input_sources else None
        ),
    )


GLM_IMAGE_MOE_TRANSFUSION_PIPELINE = PipelineConfig(
    model_type="glm_image_moe_transfusion_hy",
    model_arch=_DIFFUSION_ARCH,
    hf_architectures=(_DIFFUSION_ARCH,),
    diffusers_class_name=_DIFFUSION_ARCH,
    stages=(_diffusion_stage(0, ()),),
)


GLM_IMAGE_MOE_TRANSFUSION_THINK_PIPELINE = PipelineConfig(
    model_type="glm_image_moe_transfusion_think",
    model_arch=_AR_ARCH,
    hf_architectures=(_AR_ARCH,),
    diffusers_class_name=_DIFFUSION_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            model_arch=_AR_ARCH,
            engine_output_type="token_ids",
            sampling_constraints={"detokenize": False},
            model_path_resolver=f"{_CHECKPOINT}.resolve_ar_model_root",
        ),
        _diffusion_stage(1, (0,)),
    ),
)

__all__ = [
    "GLM_IMAGE_MOE_TRANSFUSION_PIPELINE",
    "GLM_IMAGE_MOE_TRANSFUSION_THINK_PIPELINE",
]
