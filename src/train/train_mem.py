import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import transformers
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from src.datasets import make_data_module
from src.datasets.utils import DataArguments
from src.model import ViTForIQA
from src.model.configuration_mplug_owl2 import MplugOwlVisionConfig, ViTIQAConfig
from src.train.mplug_owl2_trainer import MPLUGOwl2Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    vit_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a ViTForIQA checkpoint or a directory with ViT weights to fine-tune from."},
    )
    freeze_vision_model: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    level_names: List[str] = field(default_factory=lambda: ["excellent", "good", "fair", "poor", "bad"])
    weight_rank: float = field(default=1.0)
    softkl_loss: bool = field(default=False)
    weight_softkl: float = field(default=1.0)
    weight_next_token: float = field(default=0.05)
    continuous_rating_loss: bool = field(default=True)
    binary_rating_loss: str = field(default="fidelity")
    closeset_rating_loss: bool = field(default=False)
    use_fix_std: bool = field(default=True)
    detach_pred_std: bool = field(default=False)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    group_by_modality_length: bool = field(default=False)
    save_safetensors: bool = False


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ------------------------------------------------------------------ #
    # Build / load model                                                   #
    # ------------------------------------------------------------------ #
    if model_args.vit_model_path is not None:
        rank0_print(f"Loading ViTForIQA from {model_args.vit_model_path}")
        model = ViTForIQA.from_pretrained(
            model_args.vit_model_path, torch_dtype=compute_dtype
        )
    else:
        rank0_print("Initialising ViTForIQA from default config")
        config = ViTIQAConfig()
        model = ViTForIQA(config)
        model = model.to(compute_dtype)

    model.config.use_cache = False

    # ------------------------------------------------------------------ #
    # Loss / training configuration                                        #
    # ------------------------------------------------------------------ #
    model.config.softkl_loss = training_args.softkl_loss
    model.config.weight_softkl = training_args.weight_softkl
    model.config.weight_next_token = training_args.weight_next_token
    model.config.weight_rank = training_args.weight_rank
    model.config.continuous_rating_loss = training_args.continuous_rating_loss
    model.config.binary_rating_loss = training_args.binary_rating_loss
    model.config.closeset_rating_loss = training_args.closeset_rating_loss
    model.config.use_fix_std = training_args.use_fix_std
    model.config.detach_pred_std = training_args.detach_pred_std
    model.config.image_aspect_ratio = data_args.image_aspect_ratio

    # ------------------------------------------------------------------ #
    # Frozen / trainable parameters                                        #
    # ------------------------------------------------------------------ #
    for p in model.parameters():
        p.requires_grad = True

    if model_args.freeze_vision_model:
        rank0_print("Freezing vision model backbone")
        for p in model.vision_model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Image processor & data                                               #
    # ------------------------------------------------------------------ #
    preprocessor_path = model_args.vit_model_path or "openai/clip-vit-large-patch14-336"
    data_args.image_processor = CLIPImageProcessor.from_pretrained(preprocessor_path)
    data_args.is_multimodal = True

    model.vision_model.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    model.ln.to(device=training_args.device)
    model.head.to(device=training_args.device)

    data_module = make_data_module(data_args=data_args)

    trainer = MPLUGOwl2Trainer(
        model=model, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
