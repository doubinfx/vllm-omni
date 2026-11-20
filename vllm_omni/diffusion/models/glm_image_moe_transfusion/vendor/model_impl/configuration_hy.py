"""【自动生成，勿手改】HY 的 config 类 —— eval 路径用到的全部 6 个。

重新生成 / 校验与上游是否仍一致：
    python3 model_impl/tools/gen_configuration.py
    python3 model_impl/tools/gen_configuration.py --check    # 上游改了没
    python3 model_impl/tools/gen_configuration.py --verify   # 新旧建出的 config 是否逐键相同

逐字抽自 veomni（见文件末尾的来源表）。**没有手写的字段**：config 的默认值直接决定
模型结构（`diff_time_emb_dim=None` 会取 `hidden_size`、`diff_moe_num_experts` 决定专家数…），
手抄一个错的默认值就是静默出错。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union  # noqa: F401

from transformers.configuration_utils import PretrainedConfig

from .modeling_rope_utils import RopeParameters


# ── 逐字抽自 veomni/models/cogview/configuration_glm_image.py:75-174 ──
class GlmImageVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageVisionModel`]. It is used to instantiate a
    GlmImage vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the GLM-Image
    [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 2048):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        vq_codebook_size (`int`, *optional*, defaults to 16384):
            The size of the VQ codebook.
        vq_codebook_dim (`int`, *optional*, defaults to 2048):
            The dimension of the VQ codebook embeddings.
        vq_num_conv_layers (`int`, *optional*, defaults to 2):
            The number of convolutional layers in the VQ projector.
         spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
    Example:

    ```python
    >>> from transformers import GlmImageVisionConfig, GlmImageVisionModel

    >>> # Initializing a GlmImageVisionConfig with google/glm_image-base-patch16-224 style configuration
    >>> configuration = GlmImageVisionConfig()

    >>> # Initializing a GlmImageVisionModel (with random weights) from the google/glm_image-base-patch16-224 style configuration
    >>> model = GlmImageVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=40,
        num_attention_heads=24,
        num_channels=3,
        image_size=2048,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        vq_codebook_size=16384,
        vq_codebook_dim=2048,
        vq_num_conv_layers=2,
        vq_pre_pool_factor=1,
        vq_pre_pool_type="none",
        spatial_merge_size=2,
        vq_use_post_conv: bool = True,
        vq_use_post_quant_conv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.vq_codebook_size = vq_codebook_size
        self.vq_codebook_dim = vq_codebook_dim
        self.vq_num_conv_layers = vq_num_conv_layers
        self.vq_pre_pool_factor = vq_pre_pool_factor
        self.vq_pre_pool_type = vq_pre_pool_type
        self.spatial_merge_size = spatial_merge_size
        self.vq_use_post_quant_conv = vq_use_post_quant_conv
        self.vq_use_post_conv = vq_use_post_conv


# ── 逐字抽自 veomni/models/cogview/configuration_glm_image.py:177-268 ──
class Glm4vVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vVisionModel`]. It is used to instantiate an Glm4vVisionModel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of
    GLM-4.1V-9B-Thinking [THUDM/GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking).

    Args:
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        depth (`int`, *optional*, defaults to 24):
            Number of layers (depth) in the model.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_size (`int` or `list[int]`, *optional*, defaults to `[336, 336]`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `14`):
            The size (resolution) of each patch.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_hidden_size (`int`, *optional*, defaults to 4096):
            The output hidden size of the vision model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The size used for patches along the temporal dimension.

    Example:

    ```python
    >>> from transformers import Glm4vVisionConfig, Glm4vVisionModel

    >>> # Initializing a Glm4vVisionConfig GLM-4.1V-9B style configuration
    >>> configuration = Glm4vVisionConfig()

    >>> # Initializing a model (with random weights) from the GLM-4.1V-9B configuration
    >>> model = Glm4vVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        hidden_size=1536,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        num_heads=12,
        in_channels=3,
        image_size=336,
        patch_size=14,
        rms_norm_eps=1e-05,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=4096,
        intermediate_size=13696,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout


# ── 逐字抽自 veomni/models/cogview/configuration_glm_image.py:271-415 ──
class GlmImageTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageModel`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 171392):
            Vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GlmImageModel`]. Under the cleaned-up vh171392-init-hf layout this is
            lang_vocab_size (154880) + vision_vocab_size (16512).
        vision_vocab_size (`int`, *optional*, defaults to 16512):
            Vision vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`GlmImageVisionModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.

    ```python
    >>> from transformers import GlmImageTextModel, GlmImageConfig

    >>> # Initializing a GlmImageConfig style configuration
    >>> configuration = GlmImageConfig()

    >>> # Initializing a model from the GlmImageConfig style configuration
    >>> model = GlmImageTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `GlmImage`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 171392,
        vision_vocab_size: Optional[int] = 16512,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 13696,
        num_hidden_layers: Optional[int] = 40,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 2,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 32768,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-05,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.vision_vocab_size = vision_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.rope_parameters = (
            rope_parameters
            if rope_parameters is not None
            else {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [8, 12, 12],
            }
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# ── 逐字抽自 veomni/models/cogview/configuration_glm_image.py:418-496 ──
class GlmImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GLM-Image`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `GlmImageTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `GlmImageVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 154856):
            The image (pixel-branch) placeholder token index. Reserved-but-unmapped
            slot inside the unified GLM-base tokenizer's vocab_size=154880 — the
            tokenizer never emits it, so it's safe as a sentinel.
        image_start_token_id (`int`, *optional*, defaults to 154830):
            Image-span start sentinel = tokenizer's native ``<|begin_of_image|>``.
        image_end_token_id (`int`, *optional*, defaults to 154831):
            Image-span end sentinel = tokenizer's native ``<|end_of_image|>``.

    ```python
    >>> from transformers import Glm4vForConditionalGeneration, Glm4vConfig

    >>> # Initializing a GLM-Image style configuration
    >>> configuration = Glm4vConfig()

    >>> # Initializing a model from the GLM-Image style configuration
    >>> model = Glm4vForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image"
    sub_configs = {"vision_config": GlmImageVisionConfig, "text_config": GlmImageTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vq_vision_config=None,
        image_token_id=154856,
        image_start_token_id=154830,
        image_end_token_id=154831,
        # NOTE: init_visual / share_vit_for_vq kept for ckpt-compat; ignored by single-VIT model.
        init_visual=True,
        init_vq=False,
        share_vit_for_vq=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        # Separate config for VQ's CLIP ViT (uses GlmImageVisionConfig)
        if isinstance(vq_vision_config, dict):
            self.vq_vision_config = GlmImageVisionConfig(**vq_vision_config)
        else:
            self.vq_vision_config = vq_vision_config  # None = no VQ

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.init_visual = init_visual
        self.init_vq = init_vq
        self.share_vit_for_vq = share_vit_for_vq

        super().__init__(**kwargs)


# ── 逐字抽自 veomni/models/glm4_moe_lite/configuration_glm4_moe_lite.py:27-247 ──
class Glm4MoeLiteConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4MoeLiteModel`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V3.
    e.g. [bzantium/tiny-deepseek-v3](https://huggingface.co/bzantium/tiny-deepseek-v3)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4MoeLiteModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10240):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 47):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 20):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.8):
            Scaling factor or routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 768):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimension of the query/key heads that don't use rotary position embeddings.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts, None means dense model.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        mlp_layer_types (`list`, *optional*):
            MLP (Moe vs Dense) pattern for each layer.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Glm4MoeLiteModel, Glm4MoeLiteConfig

    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = Glm4MoeLiteConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4_moe_lite"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_proj": "local_rowwise",
        "layers.*.mlp.experts.up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        n_shared_experts: int = 1,
        n_routed_experts: int = 64,
        routed_scaling_factor: float = 1.8,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 768,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 192,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 4,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] = None,
        rope_interleave: bool = True,
        mlp_layer_types=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # ── MoE balancing knobs ────────────────────────────────────────────
        # Plan A: classic Switch-Transformer aux load-balancing loss.
        #   coef <= 0 disables it. Typical value: 1e-3 (DeepSeek/Mixtral).
        router_aux_loss_coef: float = 0.0,
        # Plan B: DeepSeek-V3 aux-loss-free e_score_correction_bias update.
        #   speed <= 0 disables it. Typical value: 1e-3 ~ 1e-2.
        router_bias_update_speed: float = 0.0,
        _moe_implementation: str = "eager",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        # Default to MoE from the second layer and on
        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)

        # layer_type_validation is changed in in 5.0.0
        # current transformers version is 4.56.2
        # layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)
        # object of this func: self.mlp_layer_types should all in ["dense", "sparse"] and num_hidden_layers == len(layer_types)

        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.router_aux_loss_coef = router_aux_loss_coef if router_aux_loss_coef is not None else 0.0
        self.router_bias_update_speed = router_bias_update_speed if router_bias_update_speed is not None else 0.0
        self._moe_implementation = _moe_implementation

        super().__init__(**kwargs)
        self.tie_word_embeddings = tie_word_embeddings


# ── 逐字抽自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion.py:47-145 ──
class GlmImageMoeTransfusionConfig(PretrainedConfig):
    """Same as GlmImageMoeARDiffConfig minus the parallel-diff-stack dims.

    No ``diff_hidden_size`` / ``diff_intermediate_size``: image tokens run
    through the AR backbone, so they inherit ``text_config.hidden_size`` /
    ``intermediate_size`` by construction.
    """

    model_type = "glm_image_moe_transfusion"
    sub_configs = {
        "text_config": Glm4MoeLiteConfig,
        "vq_vision_config": GlmImageVisionConfig,
    }

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vq_vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 154854,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        vae_token_id: int = 154856,
        # === diffusion (image-latent) knobs — same names/semantics as ARDiff ===
        diff_vae_latent_dim: int = 16,
        diff_patch_size: int = 2,
        diff_space: Literal["vae", "pixel", "raev2"] = "vae",
        raev2_encoder_type: Literal["siglip", "dinov3"] = "siglip",
        diff_pred_type: Literal["x0", "v"] = "v",
        # x0-pred v-loss 的 σ 下限:σ=flow_t.clamp_min(diff_t_min),决定低 t
        # 样本隐含 1/σ² 权重的上限(仅 diff_pred_type="x0" 生效)。默认 0.05 = 历史行为
        # (HiDream pipeline 的 T_EPS)。
        diff_t_min: float = 0.05,
        ar_loss_weight: float = 1.0,
        diff_loss_weight: float = 1.0,
        diff_timestep_embed_dim: int = 256,
        diff_time_emb_dim: int | None = None,
        diff_attention_dropout: float = 0.0,
        diff_time_embed_mode: Literal["add", "adaln"] = "add",
        diff_rope_thw_layout: Literal["block", "interleave"] = "block",
        cast_rope_emb_to_input_dtype: bool = True,
        adaln_fp32: bool = False,
        init_vq: bool = False,
        vq_compression_ratio: int = 16,
        vae_compression_ratio: int = 16,
        **kwargs,
    ):
        # Older checkpoints may carry this retired option. All models now use
        # one spatially aligned VAE/ViT coordinate rule.
        kwargs.pop("rope_offset_vae_vit", None)
        kwargs.setdefault("tie_word_embeddings", False)

        if isinstance(vq_vision_config, dict):
            self.vq_vision_config = self.sub_configs["vq_vision_config"](**vq_vision_config)
        else:
            self.vq_vision_config = vq_vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config
        self.text_config: Glm4MoeLiteConfig
        self.vq_vision_config: GlmImageVisionConfig

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.vae_token_id = vae_token_id

        self.diff_vae_latent_dim = diff_vae_latent_dim
        self.diff_patch_size = diff_patch_size
        self.diff_space = diff_space
        self.raev2_encoder_type = raev2_encoder_type
        self.diff_pred_type = diff_pred_type
        self.diff_t_min = diff_t_min
        self.ar_loss_weight = ar_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.diff_timestep_embed_dim = diff_timestep_embed_dim
        # Fourier input width and timestep MLP width are independent. Legacy
        # configs omit the latter and keep the backbone's hidden dimension.
        self.diff_time_emb_dim = self.text_config.hidden_size if diff_time_emb_dim is None else int(diff_time_emb_dim)
        if diff_time_embed_mode != "adaln" and self.diff_time_emb_dim != self.text_config.hidden_size:
            raise ValueError("A different diff_time_emb_dim requires diff_time_embed_mode='adaln'")
        self.diff_attention_dropout = diff_attention_dropout
        self.diff_time_embed_mode = diff_time_embed_mode
        self.diff_rope_thw_layout = diff_rope_thw_layout
        self.cast_rope_emb_to_input_dtype = cast_rope_emb_to_input_dtype
        self.adaln_fp32 = adaln_fp32
        self.init_vq = init_vq
        self.vq_compression_ratio = vq_compression_ratio
        self.vae_compression_ratio = vae_compression_ratio

        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        if key == "_moe_implementation":
            self.text_config._moe_implementation = value
        return super().__setattr__(key, value)

# 上游在 import 处起的别名（见 ALIASES 的说明）
NativeConfig = GlmImageMoeTransfusionConfig

# ── 逐字抽自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:67-123 ──
class _TransfusionImageConfig(NativeConfig):
    """Add source-recipe fields while retaining the common 30B config."""

    sub_configs = {**NativeConfig.sub_configs, "vision_config": Glm4vVisionConfig}

    def __init__(
        self,
        vision_config=None,
        diff_enabled=True,
        loss_norm_tokens=5642.0,
        adaln_fp32=None,
        diff_head_num_layers=0,
        diff_head_attention_dropout=0.0,
        diff_input_process_num_layers=0,
        diff_input_process_attention_dropout=0.0,
        rope_table_dtype="float64",
        diff_rope_thw_layout: str = "qwen3",
        **kwargs,
    ):
        kwargs.setdefault("text_config", {})
        kwargs["diff_rope_thw_layout"] = diff_rope_thw_layout
        kwargs.setdefault("image_token_id", 154854)
        kwargs.setdefault("vae_token_id", 154856)
        if kwargs["image_token_id"] == kwargs["vae_token_id"] == 154856:
            # Legacy HY checkpoints used one placeholder for both branches.
            # Both IDs already exist in the original vocabulary.
            kwargs["image_token_id"] = 154854
        kwargs.setdefault("video_start_token_id", None)
        kwargs.setdefault("video_end_token_id", None)
        adaln = kwargs.get("diff_time_embed_mode", "add") == "adaln"
        # Source HF configs carry these names; the local recipe has one flag.
        kwargs.pop("diff_adaln_fp32", None)
        kwargs.pop("diff_adaln_fp32_params", None)
        super().__init__(adaln_fp32=adaln if adaln_fp32 is None else bool(adaln_fp32), **kwargs)
        self.vision_config = (
            Glm4vVisionConfig(**(vision_config or {}))
            if isinstance(vision_config, dict) or vision_config is None
            else vision_config
        )
        self.text_config.tie_word_embeddings = self.tie_word_embeddings
        self.diff_enabled = diff_enabled
        self.loss_norm_tokens = float(loss_norm_tokens)
        self.diff_head_num_layers = int(diff_head_num_layers)
        self.diff_head_attention_dropout = float(diff_head_attention_dropout)
        self.diff_input_process_num_layers = int(diff_input_process_num_layers)
        self.diff_input_process_attention_dropout = float(diff_input_process_attention_dropout)
        self.rope_table_dtype = rope_table_dtype
        if self.loss_norm_tokens <= 0:
            raise ValueError("loss_norm_tokens must be positive")
        if min(self.diff_head_num_layers, self.diff_input_process_num_layers) < 0:
            raise ValueError("Diffusion block counts must be nonnegative")
        if (self.diff_head_num_layers or self.diff_input_process_num_layers) and not adaln:
            raise ValueError("Diffusion blocks require adaln mode")
        if self.diff_rope_thw_layout not in ("qwen3", "hw_thw", "interleave"):
            raise ValueError("HY diffusion RoPE must be qwen3, hw_thw or interleave")
        if rope_table_dtype not in ("float32", "float64"):
            raise ValueError("rope_table_dtype must be float32 or float64")


# ── 逐字抽自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:944-1019 ──
class _TransfusionConditioningConfig(_TransfusionImageConfig):
    model_type = "glm_image_moe_transfusion_hy"

    def __init__(
        self,
        cond_vit_aligner: str = "glmv",
        cond_vit_aligner_depth: int = 2,
        cond_vit_aligner_context_dim: int = 0,
        cond_merge_size: int = 2,
        cond_image_dual_encoder: bool = True,
        attn_block_mode: str = "blockmask",
        cond_dual_block_causal: bool = True,
        diff_time_embed_split: bool = True,
        diff_time_embed_out_init: str = "clone",
        diff_block_gate: bool = True,
        hy_fullseq_diff_blocks: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("diff_time_embed_mode", "adaln")
        super().__init__(**kwargs)
        if cond_vit_aligner not in ("glmv", "mlp_gelu", "linear"):
            raise ValueError(f"cond_vit_aligner must be 'glmv', 'mlp_gelu' or 'linear', got {cond_vit_aligner!r}")
        if attn_block_mode not in ("blockmask", "tail"):
            raise ValueError(f"attn_block_mode must be 'blockmask' or 'tail', got {attn_block_mode!r}")
        if int(cond_merge_size) < 1:
            raise ValueError(f"cond_merge_size must be >= 1, got {cond_merge_size}")
        self.cond_vit_aligner = cond_vit_aligner
        self.cond_vit_aligner_depth = int(cond_vit_aligner_depth)
        self.cond_vit_aligner_context_dim = int(cond_vit_aligner_context_dim)
        self.diff_block_gate = bool(diff_block_gate)
        self.diff_time_embed_split = bool(diff_time_embed_split)
        if diff_time_embed_out_init not in ("clone", "random"):
            raise ValueError(f"diff_time_embed_out_init must be 'clone' or 'random', got {diff_time_embed_out_init!r}")
        self.diff_time_embed_out_init = diff_time_embed_out_init
        self.cond_merge_size = int(cond_merge_size)
        self.cond_image_dual_encoder = bool(cond_image_dual_encoder)
        self.attn_block_mode = attn_block_mode
        self.cond_dual_block_causal = bool(cond_dual_block_causal)
        self.hy_fullseq_diff_blocks = bool(hy_fullseq_diff_blocks)
        if self.hy_fullseq_diff_blocks:
            if attn_block_mode != "blockmask":
                raise ValueError(
                    f"hy_fullseq_diff_blocks=True requires attn_block_mode='blockmask' — the full-sequence stacks reuse the backbone's generalized-causal block mask verbatim, and 'tail' does not define one over the unified axis (got {attn_block_mode!r})."
                )
            if self.diff_time_embed_mode != "adaln":
                raise ValueError(
                    f"hy_fullseq_diff_blocks=True requires diff_time_embed_mode='adaln': the per-token t that drives the gated adaLN inside the stacks is built on the unified axis and has no counterpart in the {self.diff_time_embed_mode!r} parameterisation."
                )
            if not self.diff_block_gate:
                raise ValueError(
                    "hy_fullseq_diff_blocks=True requires diff_block_gate=True: the full-sequence body only exists on GatedDiffDiTBlock (forward_fullseq); the ungated DiffHeadBlock has no full-sequence form, so this combination would die at the first forward instead of at startup."
                )
            if (
                int(getattr(self, "diff_head_num_layers", 0) or 0) <= 0
                and int(getattr(self, "diff_input_process_num_layers", 0) or 0) <= 0
            ):
                raise ValueError(
                    "hy_fullseq_diff_blocks=True but both stacks are empty (diff_input_process_num_layers=diff_head_num_layers=0) — the switch would be a no-op advertising a geometry the run is not using."
                )
        if self.cond_dual_block_causal:
            if not self.cond_image_dual_encoder:
                raise ValueError(
                    "cond_dual_block_causal=True needs cond_image_dual_encoder=True — there is no conditional VAE run to separate from the ViT run otherwise. Single-encoder A/B: pass cond_dual_block_causal=False alongside cond_image_dual_encoder=False."
                )
            if attn_block_mode != "blockmask":
                raise ValueError(
                    f"cond_dual_block_causal=True requires attn_block_mode='blockmask' (the per-image bidirectional blocks live only there), got {attn_block_mode!r}. For the 'tail' A/B pass cond_dual_block_causal=False as well."
                )
        if float(getattr(self, "diff_input_process_repa_weight", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "diff_input_process_repa_weight > 0 is not supported by HY: the REPA target ran through the retired VQ chain. Set it to 0."
            )
        if self.cond_image_dual_encoder and self.diff_time_embed_mode != "adaln":
            raise ValueError(
                f"cond_image_dual_encoder=True requires diff_time_embed_mode='adaln' (the conditional VAE branch reuses the generation adaLN entry chain at t=0), got {self.diff_time_embed_mode!r}"
            )


# ── 逐字抽自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:2371-2390 ──
class GlmImageMoeTransfusionSelfFlowBaseConfig(_TransfusionConditioningConfig):
    model_type = "glm_image_moe_transfusion_hy"

    def __init__(
        self,
        selfflow_enabled=False,
        selfflow_student_layer_ratio=0.3,
        selfflow_teacher_layer_ratio=0.7,
        selfflow_align_weight=0.8,
        selfflow_mask_ratio=0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0 <= selfflow_student_layer_ratio <= selfflow_teacher_layer_ratio <= 1:
            raise ValueError("Invalid SelfFlow layer ratios")
        self.selfflow_enabled = selfflow_enabled
        self.selfflow_student_layer_ratio = selfflow_student_layer_ratio
        self.selfflow_teacher_layer_ratio = selfflow_teacher_layer_ratio
        self.selfflow_align_weight = selfflow_align_weight
        self.selfflow_mask_ratio = selfflow_mask_ratio


# ── 逐字抽自 veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:2393-2397 ──
class GlmImageMoeTransfusionSelfFlowConfig(GlmImageMoeTransfusionSelfFlowBaseConfig):
    model_type = "glm_image_moe_transfusion_hy_selfflow"

    def __init__(self, selfflow_enabled=True, **kwargs):
        super().__init__(selfflow_enabled=selfflow_enabled, **kwargs)


# ── 逐字抽自 veomni/models/cogview/vae/configuration_flux2_vae.py:6-58 ──
class Flux2VAEConfig(PretrainedConfig):
    """Config for the FLUX.2 autoencoder (BFL ``ae.safetensors``).

    Defaults are read off the released FLUX.2 checkpoint
    (huggingface.co/ai-toolkit/flux2_vae, identical to the VAE shipped inside
    FLUX.2-dev):  ch=128, ch_mult=(1,2,4,4) → 3 downsamples → 8× spatial,
    z_channels=32 (conv_out emits 2*32 = mean|logvar, only the mean is used).

    Latent normalization is *not* a per-channel affine like Qwen/Wan VAEs: FLUX.2
    2×2-patchifies the 32-ch latent into 128 ch and normalizes with a frozen
    ``BatchNorm2d(128, affine=False, eps=1e-4)``'s running statistics.
    """

    model_type = "flux2_vae"

    def __init__(
        self,
        z_channels: int = 32,
        resolution: int = 256,
        in_channels: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        # 2×2 patchify applied between the encoder and the BatchNorm. Together
        # with the 8× conv stack this gives the 16× "token grid" compression.
        patch_size: int = 2,
        bn_eps: float = 1e-4,
        # FLUX.2 uses the posterior mean (no reparameterization sampling).
        sample_posterior: bool = False,
        # encode() returns the latent laid out at the *pre-patchify* resolution
        # (32 ch @ H/8) so that the downstream ``rearrange(..., ph=pw=2)`` in
        # build_diff_batch reproduces FLUX.2's native 128-ch tokens verbatim.
        # Set False to get the native (128 ch @ H/16) layout instead.
        unpatchify_latents: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tie_word_embeddings = False

        self.z_channels = z_channels
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.patch_size = patch_size
        self.bn_eps = bn_eps
        self.sample_posterior = sample_posterior
        self.unpatchify_latents = unpatchify_latents




__all__ = ['GlmImageVisionConfig', 'Glm4vVisionConfig', 'GlmImageTextConfig', 'GlmImageConfig', 'Glm4MoeLiteConfig', 'GlmImageMoeTransfusionConfig', 'GlmImageMoeTransfusionSelfFlowBaseConfig', 'GlmImageMoeTransfusionSelfFlowConfig', 'Flux2VAEConfig', 'NativeConfig']

# ── 来源表（sha256 是**上游原文**的，供 --check 比对）──────────────────────
_SOURCES = {
    "GlmImageVisionConfig": {
        "source": "veomni/models/cogview/configuration_glm_image.py:75-174",
        "sha256_upstream": "5b63add700a54a85b3270dde3053f1cc652a2713410b331411dce7f605bc9e3d"
    },
    "Glm4vVisionConfig": {
        "source": "veomni/models/cogview/configuration_glm_image.py:177-268",
        "sha256_upstream": "8a7da25a915aabf9ef51fd54dbca8139087775c6b04e3c3797e965c264a52bd5"
    },
    "GlmImageTextConfig": {
        "source": "veomni/models/cogview/configuration_glm_image.py:271-415",
        "sha256_upstream": "a47511fb1359eea9c289d813fa70c2b417b9e91382c70411d0a629255a81bc5b"
    },
    "GlmImageConfig": {
        "source": "veomni/models/cogview/configuration_glm_image.py:418-496",
        "sha256_upstream": "c3e3010ae35c1d6f6f919b2e86b39b62ec9d6c212ef1a3d7bd4efe25742a42a4"
    },
    "Glm4MoeLiteConfig": {
        "source": "veomni/models/glm4_moe_lite/configuration_glm4_moe_lite.py:27-247",
        "sha256_upstream": "85bd1417538fde01a11fb4178b5b43f063443c5b0827d3f9645af3eb3c20dc7d"
    },
    "GlmImageMoeTransfusionConfig": {
        "source": "veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion.py:47-145",
        "sha256_upstream": "a1545356cc879b21df985cc4f51c998350bfc8f269bc461b46e31e4e954ed4f4"
    },
    "_TransfusionImageConfig": {
        "source": "veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:67-123",
        "sha256_upstream": "9095cd4dcef77c3d22766f16584c6b034d20252cb25555be0e2312c054cfdb27"
    },
    "_TransfusionConditioningConfig": {
        "source": "veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:944-1019",
        "sha256_upstream": "80285a58719b5ea2512ef07332858668fbb991b516f0593e42bb1741fa008a5b"
    },
    "GlmImageMoeTransfusionSelfFlowBaseConfig": {
        "source": "veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:2371-2390",
        "sha256_upstream": "48684be3273d475f1d17f487832d919ed5cde979ad78c2ae850e4b4ed07dc23d"
    },
    "GlmImageMoeTransfusionSelfFlowConfig": {
        "source": "veomni/models/glm4_moe_lite_glm_image/modular_glm_image_moe_transfusion_selfflow.py:2393-2397",
        "sha256_upstream": "376a63e7e562aaa40fbc0bd0caf3a8571e81e5cb9ea5671979f75e6207030365"
    },
    "Flux2VAEConfig": {
        "source": "veomni/models/cogview/vae/configuration_flux2_vae.py:6-58",
        "sha256_upstream": "f56d1512f69531a571f8f7f56e98865d10917c1cdb87a352c35bd04a27da10a6"
    }
}
