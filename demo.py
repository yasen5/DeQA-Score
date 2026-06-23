"""
Demo: run ViTForIQA on 4 images and display a bar chart of the quality distribution.

Images used:
  1. fig/singapore_flyer.jpg        — original (high quality)
  2. fig/boy_colorful.jpg           — original (high quality)
  3. fig/singapore_flyer.jpg        — Gaussian-blurred  (degraded)
  4. fig/boy_colorful.jpg           — salt-and-pepper noise (degraded)

Usage:
    python3 demo.py [--model-path checkpoints/DeQA-Score-Mix3] \
                    [--head-path  checkpoints/DeQA-Score-Mix3/head.bin] \
                    [--out demo.png]
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

sys.path.insert(0, ".")
from src.mm_utils import expand2square
from src.model import ViTForIQA

LEVELS = ["excellent", "good", "fair", "poor", "bad"]
COLORS = ["#2ecc71", "#82e0aa", "#f7dc6f", "#e59866", "#e74c3c"]


# ------------------------------------------------------------------ #
# Image helpers                                                        #
# ------------------------------------------------------------------ #

def blur_image(img: Image.Image, radius: int = 8) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def noise_image(img: Image.Image, amount: float = 0.15) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-255 * amount, 255 * amount, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def load_images():
    base_a = Image.open("fig/singapore_flyer.jpg").convert("RGB")
    base_b = Image.open("fig/boy_colorful.jpg").convert("RGB")
    return [
        ("Singapore Flyer\n(original)",   base_a),
        ("Boy Colorful\n(original)",      base_b),
        ("Singapore Flyer\n(blurred)",    blur_image(base_a)),
        ("Boy Colorful\n(noisy)",         noise_image(base_b)),
    ]


# ------------------------------------------------------------------ #
# Model                                                                #
# ------------------------------------------------------------------ #

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, head_path, device):
    model = ViTForIQA.from_pretrained(model_path, torch_dtype=None)
    if head_path:
        head_sd = torch.load(head_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(head_sd, strict=False)
        if unexpected:
            print(f"[demo] Unexpected head keys: {unexpected}")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


@torch.inference_mode()
def score_images(model, processor, pil_images, device):
    tensors = []
    for img in pil_images:
        img = expand2square(img, tuple(int(x * 255) for x in processor.image_mean))
        t = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
        tensors.append(t)
    batch = torch.stack(tensors).to(device=device, dtype=torch.float32)
    output = model(images=batch)
    probs = torch.softmax(output.logits, dim=-1).cpu().numpy()   # (4, 5)
    scores = (probs * np.array([5, 4, 3, 2, 1])).sum(axis=-1)   # (4,)
    return probs, scores


# ------------------------------------------------------------------ #
# Plot                                                                 #
# ------------------------------------------------------------------ #

def make_plot(titles, pil_images, probs, scores, out_path):
    n = len(pil_images)
    fig = plt.figure(figsize=(4.5 * n, 7))
    gs = gridspec.GridSpec(2, n, height_ratios=[1, 1.4], hspace=0.35, wspace=0.35)

    for i in range(n):
        # ---- thumbnail ----
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(pil_images[i])
        ax_img.axis("off")
        ax_img.set_title(titles[i], fontsize=10, pad=4)

        # ---- bar chart ----
        ax_bar = fig.add_subplot(gs[1, i])
        bars = ax_bar.bar(LEVELS, probs[i], color=COLORS, edgecolor="white", linewidth=0.6)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Probability", fontsize=9)
        ax_bar.set_xlabel("Quality level", fontsize=9)
        ax_bar.tick_params(axis="x", labelsize=8)
        ax_bar.tick_params(axis="y", labelsize=8)
        ax_bar.set_title(f"Score: {scores[i]:.2f} / 5", fontsize=10, pad=4)

        # value labels on bars
        for bar, prob in zip(bars, probs[i]):
            if prob > 0.04:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{prob:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    fig.suptitle("ViTForIQA — Quality Distribution", fontsize=13, y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved demo plot to {out_path}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(args):
    device = get_device()
    print(f"Device: {device}")

    processor = CLIPImageProcessor.from_pretrained(args.preprocessor_path)
    model = load_model(args.model_path, args.head_path, device)

    named_images = load_images()
    titles     = [t for t, _ in named_images]
    pil_images = [img for _, img in named_images]

    probs, scores = score_images(model, processor, pil_images, device)

    for title, score, prob in zip(titles, scores, probs):
        label = title.replace("\n", " ")
        dist  = "  ".join(f"{l}={p:.2f}" for l, p in zip(LEVELS, prob))
        print(f"{label:<35}  score={score:.2f}  [{dist}]")

    make_plot(titles, pil_images, probs, scores, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",       default="checkpoints/DeQA-Score-Mix3")
    parser.add_argument("--head-path",        default="checkpoints/DeQA-Score-Mix3/head.bin")
    parser.add_argument("--preprocessor-path", default="./preprocessor")
    parser.add_argument("--out",              default="demo.png")
    args = parser.parse_args()
    main(args)
