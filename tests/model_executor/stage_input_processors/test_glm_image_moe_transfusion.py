from types import SimpleNamespace

from vllm_omni.model_executor.stage_input_processors.glm_image_moe_transfusion import (
    ar2diffusion,
)


def test_ar2diffusion_passes_thinking_tokens_and_dimensions() -> None:
    source = SimpleNamespace(
        outputs=[SimpleNamespace(cumulative_token_ids=[10, 11, 12])]
    )
    result = ar2diffusion(
        [source],
        {
            "prompt": "a red cube",
            "mm_processor_kwargs": {"target_h": 512, "target_w": 768},
        },
    )

    assert result == {
        "prompt": "a red cube",
        "height": 512,
        "width": 768,
        "extra": {"generated_token_ids": [10, 11, 12]},
    }
