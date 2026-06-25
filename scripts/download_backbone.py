"""
Download the ViT backbone weights from zhiyuanyou/DeQA-Score-Mix3 and save them
into checkpoints/DeQA-Score-Mix3/pytorch_model.bin in ViTForIQA key format.

The HF model stores ViT keys under `model.vision_model.*`.
ViTForIQA expects them under `vision_model.*` (no `model.` prefix).
The dense head (ln, head) is intentionally excluded — it is randomly initialised
and trained from scratch while the ViT backbone is frozen.
"""

import argparse
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.constants import (
    CHECKPOINT_MODEL_FILENAME,
    DEQA_SCORE_MIX3_HF_REPO_ID,
    DEQA_SCORE_MIX3_VIT_SHARD_FILENAME,
    DOWNLOAD_BACKBONE_ARG_SPECS,
)


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading shard 4 (contains all ViT weights)...")
    shard_path = hf_hub_download(
        repo_id=DEQA_SCORE_MIX3_HF_REPO_ID,
        filename=DEQA_SCORE_MIX3_VIT_SHARD_FILENAME,
    )

    print("Loading shard...")
    full_shard = load_file(shard_path, device="cpu")

    print("Extracting and remapping vision_model keys...")
    vit_state_dict = {}
    for k, v in full_shard.items():
        if k.startswith("model.vision_model."):
            new_key = k[len("model."):]   # strip "model." prefix
            vit_state_dict[new_key] = v

    print(f"Extracted {len(vit_state_dict)} ViT parameter tensors")

    out_path = os.path.join(output_dir, CHECKPOINT_MODEL_FILENAME)
    torch.save(vit_state_dict, out_path)
    print(f"Saved backbone weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg_spec in DOWNLOAD_BACKBONE_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    args = parser.parse_args()
    main(args.output_dir)
