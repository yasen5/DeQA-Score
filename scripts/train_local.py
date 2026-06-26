"""
Minimal local training script — no deepspeed required.

Trains the head on a fixed batch drawn from a soft-label JSON file produced by
build_soft_labels/gen_soft_label.py, to confirm:
  - backbone weights load correctly
  - ViT can be frozen (only ln + head train)
  - loss decreases

Usage:
    python3 scripts/train_local.py \\
        --data-path ../../Data-DeQA-Score/KONIQ/metas/train_koniq_7k_new.json \\
        --image-folder ../../Data-DeQA-Score \\
        [--steps 50] [--batch-size 4] [--lr 1e-3]
"""

import argparse
import json
import os
import sys
import tempfile
import types

import torch
from PIL import Image
from transformers import AutoImageProcessor

sys.path.insert(0, ".")
from src.constants import CHECKPOINT_HEAD_FILENAME, IQA_HEAD_STATE_KEYS, TRAIN_LOCAL_ARG_SPECS
from src.model import ViTForIQA


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_real_batch(data_path, image_folder, preprocessor_path, batch_size, device, dtype):
    """Load a fixed batch from a soft-label JSON file (gen_soft_label.py output)."""
    with open(data_path) as f:
        samples = json.load(f)

    samples = [s for s in samples if "level_probs" in s][:batch_size]
    if len(samples) < batch_size:
        raise ValueError(
            f"Not enough samples with level_probs in {data_path}: "
            f"found {len(samples)}, need {batch_size}"
        )

    processor = AutoImageProcessor.from_pretrained(preprocessor_path)

    images, level_probs = [], []
    for s in samples:
        img_path = os.path.join(image_folder, s["image"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt")["pixel_values"][0]
        images.append(pixel_values)
        level_probs.append(s["level_probs"])

    images = torch.stack(images).to(device=device, dtype=dtype)
    level_probs = torch.tensor(level_probs, dtype=dtype, device=device)
    return images, level_probs


def main(args):
    device = get_device()
    print(f"Device: {device}")

    if args.model_path is not None:
        args.model_path = os.path.abspath(args.model_path)

    # from_pretrained already loads head.bin from model_path if it exists
    model = ViTForIQA.from_pretrained(args.model_path, torch_dtype=None, softkl_loss=True)
    args.model_path = getattr(model, "resolved_checkpoint_path", args.model_path)
    if args.save_head is None:
        args.save_head = os.path.join(args.model_path, CHECKPOINT_HEAD_FILENAME)

    model = model.to(device=device, dtype=torch.float32)
    model.train()

    if args.freeze_backbone:
        # Freeze backbone — only train the head (ln + linear)
        for p in model.vision_model.parameters():
            p.requires_grad = False
        # Disable gradient checkpointing on the frozen encoder to avoid spurious warnings
        model.vision_model.encoder.gradient_checkpointing = False

    # Enable KL loss between predicted distribution and soft labels
    model.config.softkl_loss = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params: {trainable_names}")
    print(f"Total trainable: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # Linear warmup: scale LR from 0 → 1 over warmup_steps, then hold at 1
    def lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / args.warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    images, level_probs = load_real_batch(
        args.data_path, args.image_folder, args.preprocessor_path,
        args.batch_size, device, torch.float32,
    )

    print(f"\n{'Step':>5}  {'Loss':>10}  {'LR':>10}")
    print("-" * 30)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        output = model(images=images, level_probs=level_probs)
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"{step:>5}  {loss.item():>10.6f}  {current_lr:>10.2e}")

    print("\nDone.")

    head_sd = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if k in IQA_HEAD_STATE_KEYS
    }

    torch.save(head_sd, args.save_head)
    print(f"Head weights saved to {args.save_head}")

    if not args.freeze_backbone:
        model.save_pretrained(args.model_path)
        print(f"Backbone weights saved to {args.model_path}")

    if args.run_demo:
        import demo as demo_mod

        _tmp_head = None
        head_path = args.save_head
        if head_path is None:
            _tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            _tmp.close()
            head_path = _tmp.name
            _tmp_head = head_path
            torch.save(head_sd, head_path)

        demo_args = types.SimpleNamespace(
            model_path=args.model_path,
            head_path=head_path,
            preprocessor_path=args.preprocessor_path,
            out=args.demo_out,
        )
        print(f"\nRunning demo → {args.demo_out}")
        demo_mod.main(demo_args)

        if _tmp_head:
            os.unlink(_tmp_head)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg_spec in TRAIN_LOCAL_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    args = parser.parse_args()
    main(args)
