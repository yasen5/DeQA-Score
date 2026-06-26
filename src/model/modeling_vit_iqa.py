import os
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
from .visual_encoder import MplugOwlVisionModel


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
        # Two-layer MLP head.  Input features are L2-normalised before being
        # passed here (see _encode_one / score), so the head sees unit-norm
        # vectors and can use a standard initialisation scale.
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
        """Mean of patch tokens from raw encoder output, bypassing post_layernorm.

        Two design choices here:
        - Patch mean rather than CLS token: the CLS token is trained for image
          captioning and suppresses quality variation in favour of semantic
          consistency.  Averaging the 1024 patch tokens (positions 1:) preserves
          spatial quality signals (sharpness, noise, artefacts) distributed across
          the image.
        - Before post_layernorm: post_layernorm is a per-sample LayerNorm that
          divides by within-sample std (~3035), collapsing cross-image variation
          by ~3000×.  We bypass it and L2-normalise instead (see _encode_one).
        """
        vit = self.vision_model
        hidden = vit.embeddings(images)
        enc_out = vit.encoder(inputs_embeds=hidden, return_dict=True)
        # Mean-pool patch tokens only; skip position 0 (CLS).
        return enc_out.last_hidden_state[:, 1:, :].mean(dim=1)  # (B, hidden)

    def _encode_one(self, images, level_probs=None):
        """Run the ViT on one batch and return (logits, probs, scores, stds, kl_loss)."""
        feat = self._cls_features(images)             # (B, hidden)
        # L2-norm maps each feature to the unit sphere.  This removes the
        # within-sample magnitude (~3000) that would otherwise swamp cross-image
        # quality differences, and works with any batch size (unlike BatchNorm).
        feat = F.normalize(feat, dim=-1)
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
            logits = self.head(F.normalize(cls, dim=-1))
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
