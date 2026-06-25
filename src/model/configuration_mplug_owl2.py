from transformers.configuration_utils import PretrainedConfig

from src.constants import VISION_CONFIG_DEFAULTS


class ViTIQAConfig(PretrainedConfig):
    r"""
    Configuration for the pure ViT image quality assessment model.
    """

    model_type = "vit_iqa"

    def __init__(
        self,
        hidden_size=None,
        intermediate_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        image_size=None,
        patch_size=None,
        layer_norm_eps=None,
        attention_dropout=None,
        num_quality_levels=5,
        softkl_loss=False,
        weight_rank=1.0,
        weight_softkl=1.0,
        continuous_rating_loss=True,
        binary_rating_loss="fidelity",
        use_fix_std=True,
        detach_pred_std=False,
        **kwargs,
    ):
        vision_config = kwargs.get("vision_config")
        if not isinstance(vision_config, dict):
            vision_config = {}

        hidden_size = hidden_size if hidden_size is not None else vision_config.get(
            "hidden_size", VISION_CONFIG_DEFAULTS["hidden_size"]
        )
        intermediate_size = intermediate_size if intermediate_size is not None else vision_config.get(
            "intermediate_size", VISION_CONFIG_DEFAULTS["intermediate_size"]
        )
        num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else vision_config.get(
            "num_hidden_layers", VISION_CONFIG_DEFAULTS["num_hidden_layers"]
        )
        num_attention_heads = num_attention_heads if num_attention_heads is not None else vision_config.get(
            "num_attention_heads", VISION_CONFIG_DEFAULTS["num_attention_heads"]
        )
        image_size = image_size if image_size is not None else vision_config.get(
            "image_size", VISION_CONFIG_DEFAULTS["image_size"]
        )
        patch_size = patch_size if patch_size is not None else vision_config.get(
            "patch_size", VISION_CONFIG_DEFAULTS["patch_size"]
        )
        layer_norm_eps = layer_norm_eps if layer_norm_eps is not None else vision_config.get(
            "layer_norm_eps", VISION_CONFIG_DEFAULTS["layer_norm_eps"]
        )
        attention_dropout = attention_dropout if attention_dropout is not None else vision_config.get(
            "attention_dropout", VISION_CONFIG_DEFAULTS["attention_dropout"]
        )

        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_quality_levels = num_quality_levels
        self.softkl_loss = softkl_loss
        self.weight_rank = weight_rank
        self.weight_softkl = weight_softkl
        self.continuous_rating_loss = continuous_rating_loss
        self.binary_rating_loss = binary_rating_loss
        self.use_fix_std = use_fix_std
        self.detach_pred_std = detach_pred_std


if __name__ == "__main__":
    print(ViTIQAConfig().to_dict())
