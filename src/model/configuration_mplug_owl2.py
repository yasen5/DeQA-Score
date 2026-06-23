import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MplugOwlVisionConfig(PretrainedConfig):
    r"""
    Configuration for the mPLUG-Owl vision encoder (ViT backbone).
    """

    model_type = "mplug_owl_vision_model"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        use_flash_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_flash_attn = use_flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["vision_config"]
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)


class ViTIQAConfig(PretrainedConfig):
    r"""
    Top-level configuration for a pure ViT image quality assessment model.
    """

    model_type = "vit_iqa"

    def __init__(
        self,
        vision_config=None,
        num_quality_levels=5,
        # Loss configuration (can be set at runtime via model.config.xxx = ...)
        softkl_loss=False,
        weight_rank=1.0,
        weight_softkl=1.0,
        weight_next_token=0.05,
        continuous_rating_loss=True,
        binary_rating_loss="fidelity",
        closeset_rating_loss=False,
        use_fix_std=True,
        detach_pred_std=False,
        image_aspect_ratio="pad",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store vision_config as a plain dict for JSON serialisation.
        if vision_config is None:
            self.vision_config = MplugOwlVisionConfig().to_dict()
        elif isinstance(vision_config, MplugOwlVisionConfig):
            self.vision_config = vision_config.to_dict()
        else:
            self.vision_config = vision_config  # already a dict

        self.num_quality_levels = num_quality_levels
        self.softkl_loss = softkl_loss
        self.weight_rank = weight_rank
        self.weight_softkl = weight_softkl
        self.weight_next_token = weight_next_token
        self.continuous_rating_loss = continuous_rating_loss
        self.binary_rating_loss = binary_rating_loss
        self.closeset_rating_loss = closeset_rating_loss
        self.use_fix_std = use_fix_std
        self.detach_pred_std = detach_pred_std
        self.image_aspect_ratio = image_aspect_ratio


if __name__ == "__main__":
    print(MplugOwlVisionConfig().to_dict())
    print(ViTIQAConfig().to_dict())
