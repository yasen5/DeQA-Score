"""
Minimal local training script — no deepspeed required.

Trains ViTForIQA on soft-label quality supervision using end-to-end backbone
fine-tuning with gradient accumulation.

Training design
---------------
- Gradient accumulation (--grad-accum, default 32): each logged "step" averages
  gradients over grad-accum forward passes, giving an effective batch size of
  batch-size × grad-accum (default 4 × 32 = 128).  This stabilises training
  without requiring more GPU memory.
- L2-normalised patch-mean pooling (in the model): removes the ~3000× magnitude
  gap between within-sample and cross-image variation that previously caused the
  head to see identical inputs for all images.

Usage
-----
    python3 scripts/train_local.py \\
        --data-path data/Data-DeQA-Score/KADID10K/metas/train_kadid_8k.json \\
        --image-folder data/Data-DeQA-Score \\
        [--steps 500] [--batch-size 4] [--grad-accum 32] [--lr 1e-3]
"""

import argparse
import os
import random
import signal
import sys
import types

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

sys.path.insert(0, ".")
from src.constants import CHECKPOINT_HEAD_FILENAME, IQA_HEAD_STATE_KEYS, TRAIN_LOCAL_ARG_SPECS
from src.datasets.single_dataset import DataCollatorForSupervisedDataset, SingleDataset
from src.model import ViTForIQA


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args):
    device = get_device()
    print(f"Device: {device}")

    if args.model_path is not None:
        args.model_path = os.path.abspath(args.model_path)

    model = ViTForIQA.from_pretrained(args.model_path, torch_dtype=None, softkl_loss=True)
    args.model_path = getattr(model, "resolved_checkpoint_path", args.model_path)
    if args.save_head is None:
        args.save_head = os.path.join(args.model_path, CHECKPOINT_HEAD_FILENAME)

    model = model.to(device=device, dtype=torch.float32)
    model.train()

    if args.freeze_backbone:
        for p in model.vision_model.parameters():
            p.requires_grad = False
        model.vision_model.encoder.gradient_checkpointing = False
        print("Backbone frozen — training head only")
    else:
        print("Backbone unfrozen — fine-tuning end-to-end")

    model.config.softkl_loss = True

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params ({len(trainable_names)} tensors, {total_trainable:,} values)")
    print(f"  LR: {args.lr:.1e}")
    print(f"  Batch size: {args.batch_size}  Grad accum: {args.grad_accum}"
          f"  → effective batch: {args.batch_size * args.grad_accum}")

    def lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / args.warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_weights():
        if any(p.isnan().any().item() for p in model.parameters()):
            print("WARNING: NaN detected in model weights — skipping save to avoid corrupting checkpoint")
            return
        head_sd = {k: v.cpu() for k, v in model.state_dict().items() if k in IQA_HEAD_STATE_KEYS}
        torch.save(head_sd, args.save_head)
        print(f"Head saved to {args.save_head}")
        if not args.freeze_backbone:
            model.save_pretrained(args.model_path)
            print(f"Backbone saved to {args.model_path}")

    def _sigint_handler(sig, frame):
        print("\nInterrupted — saving weights before exit...")
        save_weights()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    processor = AutoImageProcessor.from_pretrained(args.preprocessor_path)
    data_args_ns = types.SimpleNamespace(
        data_paths=[args.data_path],
        data_weights=[1],
        image_folder=args.image_folder,
        image_processor=processor,
        image_aspect_ratio="pad",
    )
    dataset = SingleDataset(
        data_paths=data_args_ns.data_paths,
        data_weights=data_args_ns.data_weights,
        data_args=data_args_ns,
    )
    collator = DataCollatorForSupervisedDataset()
    if args.set_size is not None:
        if args.set_size > len(dataset):
            raise ValueError(
                f"--set-size {args.set_size} exceeds dataset size {len(dataset)}"
            )
        pool_indices = random.sample(range(len(dataset)), args.set_size)
    else:
        pool_indices = list(range(len(dataset)))
    if len(pool_indices) < args.batch_size:
        raise ValueError(
            f"Not enough samples with level_probs in {args.data_path}: "
            f"found {len(pool_indices)}, need {args.batch_size}"
        )
    print(f"\nDataset: {len(pool_indices)} samples"
          + (f" (subset of {len(dataset)})" if args.set_size is not None else "")
          + f" from {args.data_path}")
    print(f"\n{'Step':>5}  {'Loss':>10}  {'LR':>10}")
    print("-" * 32)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    (line_step,) = ax.plot([], [], alpha=0.4, color="steelblue", label="step loss")
    (line_ema,) = ax.plot([], [], color="steelblue", linewidth=2, label="EMA loss")
    ax.legend()
    plot_steps, plot_losses, plot_ema = [], [], []

    loss_plot_path = os.path.join(args.model_path, "loss_curve.png")

    def update_plot():
        line_step.set_data(plot_steps, plot_losses)
        line_ema.set_data(plot_steps, plot_ema)
        ax.relim()
        ax.autoscale_view()
        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(0.001)
        fig.savefig(loss_plot_path, dpi=100)

    accum = args.grad_accum
    optimizer.zero_grad()
    running_loss = 0.0

    for step in range(1, args.steps + 1):
        # Each logged step accumulates `accum` mini-batches before an update.
        step_loss = 0.0
        for _ in range(accum):
            indices = random.sample(pool_indices, args.batch_size)
            batch = collator([dataset[i] for i in indices])
            images = batch["images"].to(device=device, dtype=torch.float32)
            level_probs = batch["level_probs"].to(device=device, dtype=torch.float32)
            output = model(images=images, level_probs=level_probs)
            # Divide by accum so the accumulated gradient equals the mean loss.
            loss = output.loss / accum
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 1 else step_loss
        if step % args.log_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"{step:>5}  {step_loss:>10.6f}  {current_lr:>10.2e}  "
                  f"(ema {running_loss:.4f})")
            plot_steps.append(step)
            plot_losses.append(step_loss)
            plot_ema.append(running_loss)
            update_plot()

    print("\nDone.")
    save_weights()
    update_plot()
    print(f"Loss curve saved to {loss_plot_path}")
    plt.ioff()
    plt.show()

    if args.run_demo:
        import demo as demo_mod
        demo_args = types.SimpleNamespace(
            model_path=args.model_path,
            head_path=args.save_head,
            preprocessor_path=args.preprocessor_path,
            dataset_json=args.data_path,
            data_root=args.image_folder,
            num_samples=4,
            seed=0,
            out=args.demo_out,
            allowed_indices=pool_indices if args.set_size is not None else None,
        )
        print(f"\nRunning demo → {args.demo_out}")
        demo_mod.main(demo_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for arg_spec in TRAIN_LOCAL_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    args = parser.parse_args()
    main(args)
