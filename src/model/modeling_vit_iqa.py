import os
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from src.constants import (
    CHECKPOINT_AUTO_DIR_PREFIX,
    CHECKPOINT_CONFIG_FILENAME,
    CHECKPOINT_CONFIG_METADATA_FIELDS,
    CHECKPOINT_HASH_PREFIX_LENGTH,
    CHECKPOINT_HEAD_FILENAME,
    CHECKPOINT_METADATA_FILENAME,
    CHECKPOINT_METADATA_VERSION,
    CHECKPOINT_MODEL_FILENAME,
    CHECKPOINTS_DIRNAME,
    IQA_HEAD_STATE_KEYS,
)
from .configuration_mplug_owl2 import ViTIQAConfig


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MplugOwlVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_size))

        self.pre_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.cls_token.expand(batch_size, 1, -1).to(image_embeds.dtype)
        embeddings = torch.cat([class_embeds, image_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1)].to(image_embeds.dtype)
        embeddings = self.pre_layernorm(embeddings)
        return embeddings


class MplugOwlVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, seq_len, embed_dim = hidden_states.size()

        mixed_qkv = self.query_key_value(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, seq_len, self.num_heads, 3, embed_dim // self.num_heads).permute(
            3, 0, 2, 1, 4
        )  # [3, b, np, sq, hn]
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # [B, n_heads, ]
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.dense(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MplugOwlMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MplugOwlVisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MplugOwlVisionAttention(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MplugOwlMLP(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MplugOwlVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MplugOwlVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MplugOwlVisionModel(PreTrainedModel):
    main_input_name = "pixel_values"
    _no_split_modules = ["MplugOwlVisionEncoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embeddings = MplugOwlVisionEmbeddings(config)
        self.encoder = MplugOwlVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


@dataclass
class ViTIQAOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    stds: Optional[torch.Tensor] = None


class ViTForIQA(nn.Module):
    """
    Pure Vision Transformer image quality assessment model.

    Pipeline: image → MplugOwlVisionModel (ViT) → CLS token → LayerNorm → Linear(hidden, 5)
    Score: softmax(logits) @ [5, 4, 3, 2, 1]
    """

    def __init__(self, config: ViTIQAConfig):
        super().__init__()
        self.config = config
        self.vision_model = MplugOwlVisionModel(config)
        hidden = config.hidden_size
        # BatchNorm over the patch-mean features before the head.
        # Raw encoder output is ~450K in magnitude and nearly identical across
        # images (cosine sim 0.998+). L2-normalisation would project those nearly-
        # parallel vectors onto the unit sphere with <0.06 L2 distance apart — too
        # small for the head to learn image-specific predictions.  More critically,
        # when all features are nearly collinear the head produces the same output
        # for every image, so every image's backward pass pushes the backbone in the
        # same direction, which makes features even MORE similar — a deadlock.
        # BatchNorm normalises per feature-dimension across the batch (÷ std_batch ≈ 28),
        # not per sample (÷ L2_norm ≈ 450K).  This amplifies the tiny inter-image
        # differences into unit-std signals the head can read from step 1, producing
        # image-specific head outputs → image-specific backbone gradients → features
        # diverge, breaking the deadlock.
        self.feat_bn = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, config.num_quality_levels),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _weights(self, device, dtype):
        return torch.tensor([5., 4., 3., 2., 1.], device=device, dtype=dtype)

    def _probs_scores_stds(self, logits):
        probs = torch.softmax(logits, dim=-1)
        w = self._weights(logits.device, logits.dtype)
        scores = probs @ w
        variances = (w.unsqueeze(0) - scores.unsqueeze(1)) ** 2
        stds = torch.sqrt((probs * variances).sum(dim=-1))
        return probs, scores, stds

    def _softkl_loss(self, logits, level_probs):
        log_probs = F.log_softmax(logits, dim=-1)
        target = level_probs.to(dtype=logits.dtype, device=logits.device).clamp(min=0)
        # gen_soft_label.py clips negatives to 0, which can leave the distribution summing
        # to slightly more than 1.  Renormalise so the KL target is a proper distribution.
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return F.kl_div(log_probs, target, reduction="batchmean")

    def _rating_loss(
        self,
        pred_scores_A, pred_stds_A, gt_scores_A, gt_stds_A,
        pred_scores_B, pred_stds_B, gt_scores_B, gt_stds_B,
    ):
        eps = 1e-8
        if self.config.use_fix_std:
            # Assume unit std (sqrt(2) denominator comes from sqrt(1^2 + 1^2))
            pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / 2))
        else:
            pred_var = pred_stds_A ** 2 + pred_stds_B ** 2 + eps
            if self.config.detach_pred_std:
                pred_var = pred_var.detach()
            pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / torch.sqrt(2 * pred_var)))
        gt_var = gt_stds_A ** 2 + gt_stds_B ** 2 + eps
        gt = 0.5 * (1 + torch.erf((gt_scores_A - gt_scores_B) / torch.sqrt(2 * gt_var))).to(pred.device)
        gt = gt.detach()
        loss = (1 - (pred * gt + eps).sqrt() - ((1 - pred) * (1 - gt) + eps).sqrt()).mean()
        return loss

    def _binary_rating_loss(self, pred_scores_A, gt_scores_A, pred_scores_B, gt_scores_B):
        pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / 2))
        gt = (gt_scores_A > gt_scores_B).to(pred.dtype).to(pred.device).detach()
        if self.config.binary_rating_loss == "bce":
            return F.binary_cross_entropy(pred, gt)
        elif self.config.binary_rating_loss == "fidelity":
            loss_1 = 1 - pred[gt == 1].sqrt()
            loss_2 = 1 - (1 - pred[gt == 0]).sqrt()
            return (loss_1.sum() + loss_2.sum()) / pred_scores_A.shape[0]
        raise NotImplementedError(f"Unknown binary_rating_loss: {self.config.binary_rating_loss}")

    def _cls_features(self, images):
        """Mean of patch tokens from raw encoder output (before post_layernorm).

        Patch mean rather than CLS token: the CLS token is trained for image
        captioning and suppresses quality variation in favour of semantic
        consistency.  Averaging the patch tokens (positions 1:) preserves
        spatial quality signals (sharpness, noise, artefacts) distributed across
        the image.  post_layernorm is intentionally skipped here; feat_bn in
        _encode_one handles normalisation in a way that preserves inter-image
        discriminability (see the comment on feat_bn in __init__).
        """
        vit = self.vision_model
        hidden = vit.embeddings(images)
        enc_out = vit.encoder(inputs_embeds=hidden, return_dict=True)
        return enc_out.last_hidden_state[:, 1:, :].mean(dim=1)  # (B, hidden)

    def _encode_one(self, images, level_probs=None):
        """Run the ViT on one batch and return (logits, probs, scores, stds, kl_loss)."""
        feat = self._cls_features(images)             # (B, hidden)
        feat = self.feat_bn(feat)                     # BatchNorm: zero batch-mean, unit batch-std per dim
        logits = self.head(feat)                      # (B, num_quality_levels)
        probs, scores, stds = self._probs_scores_stds(logits)
        kl_loss = None
        if level_probs is not None and self.config.softkl_loss:
            kl_loss = self._softkl_loss(logits, level_probs)
        return logits, probs, scores, stds, kl_loss

    # ------------------------------------------------------------------ #
    # Public forward                                                       #
    # ------------------------------------------------------------------ #

    def forward(self, input_type=None, **kwargs):
        if input_type == "pair":
            return self._forward_pair(**kwargs)
        return self._forward_single(**kwargs)

    def _forward_single(self, images, level_probs=None, **kwargs):
        logits, probs, scores, stds, kl_loss = self._encode_one(images, level_probs)
        return ViTIQAOutput(loss=kl_loss, logits=logits, scores=scores, stds=stds)

    def _forward_pair(self, item_A, item_B, **kwargs):
        logits_A, _, scores_A, stds_A, kl_A = self._encode_one(item_A.images, item_A.level_probs)
        logits_B, _, scores_B, stds_B, kl_B = self._encode_one(item_B.images, item_B.level_probs)

        gt_scores_A = item_A.gt_scores.to(scores_A)
        gt_scores_B = item_B.gt_scores.to(scores_B)

        if self.config.continuous_rating_loss:
            gt_stds_A = item_A.stds.to(stds_A)
            gt_stds_B = item_B.stds.to(stds_B)
            loss_rank = self._rating_loss(
                scores_A, stds_A, gt_scores_A, gt_stds_A,
                scores_B, stds_B, gt_scores_B, gt_stds_B,
            )
        else:
            loss_rank = self._binary_rating_loss(scores_A, gt_scores_A, scores_B, gt_scores_B)

        loss = self.config.weight_rank * loss_rank

        if self.config.softkl_loss and kl_A is not None:
            loss = loss + self.config.weight_softkl * (kl_A + kl_B)

        try:
            if dist.get_rank() == 0:
                print(f"[loss | ranking: {round(loss_rank.item(), 6)}]")
        except Exception:
            pass

        return ViTIQAOutput(loss=loss)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def score(self, images):
        """Return quality scores in [1, 5] for a batch of preprocessed images."""
        with torch.inference_mode():
            cls = self._cls_features(images)
            logits = self.head(self.feat_bn(cls))
            _, scores, _ = self._probs_scores_stds(logits)
            return scores

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def _config_metadata(cls, config):
        return {field: getattr(config, field) for field in CHECKPOINT_CONFIG_METADATA_FIELDS}

    @classmethod
    def _config_hash(cls, config):
        payload = json.dumps(cls._config_metadata(config), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _metadata_for_config(cls, config):
        return {
            "metadata_version": CHECKPOINT_METADATA_VERSION,
            "config_hash": cls._config_hash(config),
            "config": cls._config_metadata(config),
            "files": {
                "model": CHECKPOINT_MODEL_FILENAME,
                "head": CHECKPOINT_HEAD_FILENAME,
            },
        }

    @classmethod
    def _write_metadata(cls, checkpoint_dir, config):
        metadata_file = Path(checkpoint_dir) / CHECKPOINT_METADATA_FILENAME
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(cls._metadata_for_config(config), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def _checkpoint_root(cls, model_path=None):
        if model_path is None:
            return Path.cwd() / CHECKPOINTS_DIRNAME
        path = Path(model_path).resolve()
        for candidate in (path, *path.parents):
            if candidate.name == CHECKPOINTS_DIRNAME:
                return candidate
        return Path.cwd() / CHECKPOINTS_DIRNAME

    @classmethod
    def _checkpoint_layer_count(cls, weights_file):
        state_dict = torch.load(weights_file, map_location="cpu")
        layer_ids = {
            int(key.split("encoder.layers.")[1].split(".")[0])
            for key in state_dict
            if "encoder.layers." in key
        }
        return len(layer_ids)

    @classmethod
    def _bootstrap_legacy_metadata(cls, checkpoints_root):
        if not checkpoints_root.exists():
            return
        for config_file in checkpoints_root.rglob(CHECKPOINT_CONFIG_FILENAME):
            checkpoint_dir = config_file.parent
            metadata_file = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
            weights_file = checkpoint_dir / CHECKPOINT_MODEL_FILENAME
            if metadata_file.exists() or not weights_file.exists():
                continue

            config = ViTIQAConfig.from_pretrained(str(checkpoint_dir))
            try:
                layer_count = cls._checkpoint_layer_count(weights_file)
            except Exception as exc:
                print(f"[ViTForIQA] Skipping legacy metadata for {checkpoint_dir}: {exc}")
                continue

            if layer_count != config.num_hidden_layers:
                print(
                    "[ViTForIQA] Skipping legacy metadata for "
                    f"{checkpoint_dir}: checkpoint has {layer_count} layers, "
                    f"config requests {config.num_hidden_layers}"
                )
                continue

            cls._write_metadata(checkpoint_dir, config)
            print(f"[ViTForIQA] Wrote legacy metadata to {metadata_file}")

    @classmethod
    def _find_checkpoint_for_config(cls, checkpoints_root, config_hash):
        if not checkpoints_root.exists():
            return None
        for metadata_file in checkpoints_root.rglob(CHECKPOINT_METADATA_FILENAME):
            try:
                with metadata_file.open(encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                continue
            if metadata.get("config_hash") == config_hash:
                return metadata_file.parent
        return None

    @classmethod
    def _new_checkpoint_dir(cls, checkpoints_root, config_hash):
        hash_prefix = config_hash[:CHECKPOINT_HASH_PREFIX_LENGTH]
        base = checkpoints_root / f"{CHECKPOINT_AUTO_DIR_PREFIX}-{hash_prefix}"
        checkpoint_dir = base
        suffix = 2
        while checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            checkpoint_dir = checkpoints_root / f"{base.name}-{suffix}"
            suffix += 1
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @classmethod
    def from_pretrained(cls, model_path=None, torch_dtype=torch.float16, device_map=None, **kwargs):
        checkpoints_root = kwargs.pop("checkpoints_root", None)
        if model_path is None:
            config = ViTIQAConfig(**kwargs)
        else:
            config = ViTIQAConfig.from_pretrained(model_path, **kwargs)

        if checkpoints_root is None:
            checkpoints_root = cls._checkpoint_root(model_path)
        else:
            checkpoints_root = Path(checkpoints_root).resolve()
        cls._bootstrap_legacy_metadata(checkpoints_root)

        config_hash = cls._config_hash(config)
        resolved_path = cls._find_checkpoint_for_config(checkpoints_root, config_hash)
        random_init = resolved_path is None

        if random_init:
            resolved_path = cls._new_checkpoint_dir(checkpoints_root, config_hash)
            config.save_pretrained(str(resolved_path))
            cls._write_metadata(resolved_path, config)
            print(
                f"[ViTForIQA] No checkpoint found for this config; "
                f"starting from randomly initialized weights at {resolved_path}"
            )
        else:
            resolved_path = Path(resolved_path)
            if model_path is None or resolved_path.resolve() != Path(model_path).resolve():
                print(f"[ViTForIQA] Using checkpoint metadata match at {resolved_path}")

        model = cls(config)
        model.resolved_checkpoint_path = str(resolved_path)

        weights_file = resolved_path / CHECKPOINT_MODEL_FILENAME
        if not random_init and weights_file.exists():
            state_dict = torch.load(weights_file, map_location="cpu")
            # Drop head and any legacy normalisation keys (ln.*, head.*) that may
            # be stale in pytorch_model.bin from a prior architecture.
            # head.bin is loaded separately and is authoritative for the head.
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("head.") and not k.startswith("ln.")
                and not k.startswith("feat_bn.")
            }
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                print(f"[ViTForIQA] Unexpected backbone keys: {unexpected}")
            print(f"[ViTForIQA] Backbone weights loaded from {weights_file}")
        elif not random_init:
            print(f"[ViTForIQA] WARNING: checkpoint dir exists but {CHECKPOINT_MODEL_FILENAME} not found — backbone is randomly initialized")
        head_file = resolved_path / CHECKPOINT_HEAD_FILENAME
        if not random_init and head_file.exists():
            head_sd = torch.load(head_file, map_location="cpu", weights_only=True)
            model.load_state_dict(head_sd, strict=False)
            print(f"[ViTForIQA] Head weights loaded from {head_file}")
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.config.save_pretrained(output_dir)
        self._write_metadata(output_dir, self.config)
        torch.save(self.state_dict(), os.path.join(output_dir, CHECKPOINT_MODEL_FILENAME))
