"""
Minimal local training script — no deepspeed, no disk images required.

Overfits a small fixed synthetic batch to confirm:
  - backbone weights load correctly
  - ViT is frozen (only ln + head train)
  - loss decreases

Usage:
    python3 scripts/train_local.py [--steps 50] [--batch-size 4] [--lr 1e-3]
"""

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model import ViTForIQA


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_synthetic_batch(batch_size, image_size, device, dtype):
    """Fixed batch of random images + soft quality labels."""
    torch.manual_seed(0)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
    # Random soft labels that sum to 1 per sample
    raw = torch.rand(batch_size, 5, device=device, dtype=dtype)
    level_probs = raw / raw.sum(dim=-1, keepdim=True)
    return images, level_probs


def main(args):
    device = get_device()
    print(f"Device: {device}")

    model = ViTForIQA.from_pretrained(args.model_path, torch_dtype=None)
    model = model.to(device=device, dtype=torch.float32)
    model.train()

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

    image_size = 448   # matches MplugOwlVisionConfig default
    images, level_probs = make_synthetic_batch(args.batch_size, image_size, device, torch.float32)

    print(f"\n{'Step':>5}  {'Loss':>10}")
    print("-" * 18)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        output = model(images=images, level_probs=level_probs)
        loss = output.loss
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            print(f"{step:>5}  {loss.item():>10.6f}")

    print("\nDone. Loss should be decreasing over steps.")

    if args.save_head:
        head_sd = {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if k in {"ln.weight", "ln.bias", "head.weight", "head.bias"}
        }
        torch.save(head_sd, args.save_head)
        print(f"Head weights saved to {args.save_head}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/DeQA-Score-Mix3")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-head", default=None, metavar="PATH",
                        help="Save trained head weights to this path after training")
    args = parser.parse_args()
    main(args)
