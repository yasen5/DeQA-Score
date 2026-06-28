"""
Measure how much image quality information exists in the ViT backbone's raw
encoder output BEFORE any quality-specific training.

The question: can a linear projection (PCA PC1) of the backbone features
predict ground-truth quality scores, even though the backbone was never
trained on quality labels?

What this script does
---------------------
1. Load the frozen backbone and run it on N images.
2. Collect the raw encoder output (bypassing post_layernorm, which destroys
   cross-image signal — see src/model/modeling_vit_iqa.py for the explanation).
3. Apply one of three pooling strategies to produce a single feature vector
   per image (see --pooling below).
4. L2-normalise each feature vector so magnitudes don't dominate direction.
5. Run PCA, take the first principal component (the direction of maximum
   variance across images).
6. Compute Pearson r and Spearman ρ between PC1 scores and GT quality scores.
7. Save a scatter plot showing the relationship.

Pooling strategies (--pooling)
-------------------------------
  cls          Position-0 token (the global summary trained for captioning).
               May suppress quality variation in favour of semantic consistency.

  patch_mean   Mean of the 1024 patch tokens (positions 1:).
               Preserves spatial quality information — sharpness, noise, and
               artefacts are distributed across the image, so averaging patch
               embeddings captures them better than the CLS aggregation.

  all_mean     Mean of all 1025 tokens (CLS + patches).  Usually close to
               patch_mean because the 1024 patches dominate the average.

  compare      Run all three strategies and print a side-by-side comparison.
               Useful for deciding which pooling to use in the model.

Interpreting results
--------------------
|r| > 0.5   Strong linear relationship — a linear head should learn quickly.
|r| 0.3–0.5 Moderate — head can learn but will benefit from an MLP.
|r| < 0.3   Weak — that pooling does not encode quality linearly;
             an MLP head or backbone fine-tuning is needed.

Usage
-----
    python3 scripts/analyze_pca_quality_correlation.py \\
        --data-path data/Data-DeQA-Score/KADID10K/metas/train_kadid_8k.json \\
        --image-folder data/Data-DeQA-Score \\
        --pooling compare

Optional flags
--------------
    --num-samples 500           Analyse only N samples (default: all).
    --infer-batch 8             Images per forward pass (reduce if OOM).
    --out pca_correlation.png   Output plot path.
    --model-path PATH           Explicit checkpoint (auto-detected by default).
    --no-l2-norm                Skip L2 normalisation before PCA.
    --seed 42                   Random seed for sample shuffling.
"""

import argparse
import os
import random
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor

sys.path.insert(0, ".")
from src.datasets.build_soft_labels.gen_soft_label import SoftLabelSample, load_soft_label_samples
from src.mm_utils import expand2square
from src.model import ViTForIQA


POOLING_CHOICES = ("cls", "patch_mean", "all_mean", "compare")

POOLING_LABELS = {
    "cls":        "CLS token (pos 0)",
    "patch_mean": "Mean of patch tokens (pos 1:)",
    "all_mean":   "Mean of all tokens (pos 0:)",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(data_path, num_samples, seed):
    data = [
        s for s in load_soft_label_samples(data_path)
        if s.level_probs is not None and s.gt_score_norm is not None
    ]
    if not data:
        raise ValueError(f"No usable samples in {data_path}")
    random.seed(seed)
    random.shuffle(data)
    if num_samples is not None:
        data = data[:num_samples]
    return data


def extract_raw_encoder_outputs(
    model,
    samples: List[SoftLabelSample],
    processor,
    bg_color,
    image_folder,
    device,
    infer_batch,
):
    """
    Run the frozen backbone on every sample.

    Returns:
      last_hidden_state : np.ndarray  (N, num_tokens, hidden_size)
                          Raw encoder output before post_layernorm.
                          Tokens: position 0 = CLS, positions 1: = patches.
      gt_scores         : np.ndarray  (N,)  ground-truth quality (1–5)
    """
    model.eval()
    all_hidden, all_scores = [], []

    with torch.inference_mode():
        for i in range(0, len(samples), infer_batch):
            chunk = samples[i:i + infer_batch]
            imgs, scores = [], []
            for s in chunk:
                img_path = os.path.join(image_folder, s.image)
                img = Image.open(img_path).convert("RGB")
                img = expand2square(img, bg_color)
                pv = processor(img, return_tensors="pt")["pixel_values"][0]
                imgs.append(pv)
                scores.append(s.gt_score_norm)

            imgs_t = torch.stack(imgs).to(device=device, dtype=torch.float32)
            vit = model.vision_model
            hidden = vit.embeddings(imgs_t)
            enc_out = vit.encoder(inputs_embeds=hidden, return_dict=True)
            # Shape: (batch, 1 + num_patches, hidden_size) — before post_layernorm
            all_hidden.append(enc_out.last_hidden_state.cpu().numpy())
            all_scores.extend(scores)

            n_done = min(i + infer_batch, len(samples))
            if n_done % 500 < infer_batch or n_done == len(samples):
                print(f"  {n_done}/{len(samples)} images processed")

    return np.concatenate(all_hidden, axis=0), np.array(all_scores, dtype=np.float32)


def pool(hidden_states, pooling):
    """hidden_states: (N, T, D).  Returns (N, D)."""
    if pooling == "cls":
        return hidden_states[:, 0, :]
    if pooling == "patch_mean":
        return hidden_states[:, 1:, :].mean(axis=1)
    if pooling == "all_mean":
        return hidden_states.mean(axis=1)
    raise ValueError(f"Unknown pooling: {pooling}")


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #

def run_pca_correlation(feats_raw, scores, l2_norm):
    """
    PCA on features; correlate each of the top 10 PCs with quality scores.

    Returns a dict including:
        pc1_scores    : (N,) projection onto PC1
        pearson_r     : Pearson r between PC1 and GT quality
        spearman_rho  : Spearman ρ between PC1 and GT quality
        explained_var : fraction of variance explained by PC1
        pc_correlations : list of (pc_num, r, p, explained_var) for PCs 1–10
    """
    feats = feats_raw.copy()
    if l2_norm:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-8, None)

    pca = PCA(n_components=10)
    projections = pca.fit_transform(feats)  # (N, 10)

    pc1_scores = projections[:, 0]
    pearson_r, pearson_p     = stats.pearsonr(pc1_scores, scores)
    spearman_rho, spearman_p = stats.spearmanr(pc1_scores, scores)

    pc_correlations = []
    for k in range(projections.shape[1]):
        r, p = stats.pearsonr(projections[:, k], scores)
        pc_correlations.append((k + 1, r, p, pca.explained_variance_ratio_[k]))

    return {
        "pc1_scores":      pc1_scores,
        "pearson_r":       pearson_r,
        "pearson_p":       pearson_p,
        "spearman_rho":    spearman_rho,
        "spearman_p":      spearman_p,
        "explained_var":   pca.explained_variance_ratio_[0],
        "pc_correlations": pc_correlations,
    }


def print_result(label, result, n_samples):
    r   = result["pearson_r"]
    rho = result["spearman_rho"]
    ev  = result["explained_var"]
    print(f"\n  Pooling : {label}")
    print(f"  Samples : {n_samples}")
    print(f"  PC1 explained variance : {ev*100:.1f}%")
    print(f"  Pearson  r = {r:+.4f}  (p = {result['pearson_p']:.2e})")
    print(f"  Spearman ρ = {rho:+.4f}  (p = {result['spearman_p']:.2e})")
    print()
    print(f"  {'PC':<6} {'Pearson r':>10} {'Expl. var':>10}")
    print(f"  {'-'*28}")
    best_abs_r = max(abs(x[1]) for x in result["pc_correlations"])
    for pc_num, r_k, _, ev_k in result["pc_correlations"]:
        marker = "  ← best" if abs(r_k) == best_abs_r else ""
        print(f"  PC{pc_num:<4} {r_k:>+10.4f} {ev_k*100:>9.1f}%{marker}")


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def make_scatter_plot(pooling_results, scores, n_samples, out_path, l2_norm):
    """
    pooling_results : dict mapping pooling_name -> result dict
    One column per pooling strategy: scatter + regression on top, per-PC bar below.
    """
    pooling_names = list(pooling_results.keys())
    n_cols = len(pooling_names)
    norm_label = "L2-normalised" if l2_norm else "raw"

    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(pooling_names):
        res   = pooling_results[name]
        pc1   = res["pc1_scores"]
        r     = res["pearson_r"]
        rho   = res["spearman_rho"]
        ev    = res["explained_var"]
        label = POOLING_LABELS.get(name, name)

        # ---- Scatter ----
        ax = axes[0, col]
        ax.scatter(pc1, scores, s=5, alpha=0.30, color="#3498db", linewidths=0)
        m, b = np.polyfit(pc1, scores, 1)
        x_line = np.linspace(pc1.min(), pc1.max(), 200)
        ax.plot(x_line, m * x_line + b, color="#e74c3c", linewidth=1.5)
        ax.set_xlabel("PC1 projection", fontsize=10)
        ax.set_ylabel("GT quality (1–5)", fontsize=10)
        ax.set_title(
            f"{label}\nPearson r={r:+.3f}  ρ={rho:+.3f}  PC1 var={ev*100:.1f}%",
            fontsize=9,
        )
        ax.grid(alpha=0.2)

        # ---- Per-PC bar ----
        ax2 = axes[1, col]
        pcs   = res["pc_correlations"]
        pc_rs = [p[1] for p in pcs]
        pc_lbl = [f"PC{p[0]}" for p in pcs]
        best  = max(abs(v) for v in pc_rs)
        colors = ["#e74c3c" if abs(v) == best else "#3498db" for v in pc_rs]
        bars = ax2.bar(pc_lbl, pc_rs, color=colors, edgecolor="white", linewidth=0.4)
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("Pearson r with GT quality", fontsize=9)
        ax2.set_title("Per-PC correlation", fontsize=9)
        ax2.grid(axis="y", alpha=0.2)
        for bar, v in zip(bars, pc_rs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.025 * np.sign(v if v != 0 else 1),
                f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )

    fig.suptitle(
        f"ViT quality signal before training  ({n_samples} images, {norm_label} features)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(args):
    device = get_device()
    print(f"Device: {device}")

    print("Loading model ...")
    # softkl_loss=True selects the checkpoint that has real backbone weights.
    # Without it, from_pretrained matches a different config hash and may land
    # on a directory that has no pytorch_model.bin (randomly initialized backbone),
    # which produces near-zero feature correlations.
    model = ViTForIQA.from_pretrained(args.model_path, torch_dtype=None, softkl_loss=True)
    ckpt_path = getattr(model, "resolved_checkpoint_path", None)
    print(f"Checkpoint: {ckpt_path}")

    # Abort early if no backbone weights were loaded — random features are useless.
    if ckpt_path:
        weights_file = os.path.join(ckpt_path, "pytorch_model.bin")
        if not os.path.exists(weights_file):
            print(
                f"\nERROR: No backbone weights found at {weights_file}.\n"
                "The backbone is randomly initialized, so features carry no quality signal.\n"
                "Pass the correct checkpoint explicitly:\n"
                "  --model-path checkpoints/<your-dir-with-pytorch_model.bin>"
            )
            sys.exit(1)
    model = model.to(device=device, dtype=torch.float32)

    processor = AutoImageProcessor.from_pretrained(args.preprocessor_path)
    bg_color  = tuple(int(x * 255) for x in processor.image_mean)

    print(f"\nLoading samples from {args.data_path} ...")
    samples = load_samples(args.data_path, args.num_samples, args.seed)
    print(f"Analysing {len(samples)} samples")

    print("\nExtracting encoder outputs ...")
    hidden_states, scores = extract_raw_encoder_outputs(
        model, samples, processor, bg_color,
        args.image_folder, device, args.infer_batch,
    )
    # hidden_states: (N, 1025, 1024)
    print(f"Encoder output shape: {hidden_states.shape}  "
          f"(images × tokens × hidden_size)")

    pooling_names = (
        ["cls", "patch_mean", "all_mean"] if args.pooling == "compare"
        else [args.pooling]
    )

    l2_norm = not args.no_l2_norm
    pooling_results = {}
    print("\n" + "=" * 60)
    for name in pooling_names:
        feats  = pool(hidden_states, name)
        result = run_pca_correlation(feats, scores, l2_norm)
        pooling_results[name] = result
        print_result(POOLING_LABELS.get(name, name), result, len(samples))
    print("=" * 60)

    make_scatter_plot(pooling_results, scores, len(samples), args.out, l2_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-path",    required=True,
                        help="JSON file from gen_soft_label.py")
    parser.add_argument("--image-folder", required=True,
                        help="Base directory prepended to each sample's 'image' field")
    parser.add_argument("--model-path",  default=None,
                        help="Checkpoint path (auto-detected from checkpoints/ if omitted)")
    parser.add_argument("--preprocessor-path", default="preprocessor",
                        help="Path to image preprocessor config (default: preprocessor)")
    parser.add_argument("--pooling", default="compare", choices=POOLING_CHOICES,
                        help="Pooling strategy, or 'compare' to show all three (default: compare)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Analyse only the first N samples after shuffling (default: all)")
    parser.add_argument("--infer-batch", type=int, default=8,
                        help="Images per backbone forward pass (reduce if OOM, default: 8)")
    parser.add_argument("--no-l2-norm", action="store_true",
                        help="Skip L2 normalisation before PCA")
    parser.add_argument("--out", default="pca_correlation.png",
                        help="Output scatter plot path (default: pca_correlation.png)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample shuffling (default: 42)")
    args = parser.parse_args()
    main(args)
