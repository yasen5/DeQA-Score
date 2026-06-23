from transformers.configuration_utils import PretrainedConfig


class ViTIQAConfig(PretrainedConfig):
    r"""
    Configuration for the pure ViT image quality assessment model.
    """

    model_type = "vit_iqa"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=448,
        patch_size=14,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
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
