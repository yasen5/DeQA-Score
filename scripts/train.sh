#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH

# Path to a ViTForIQA checkpoint to fine-tune from (leave empty to train from scratch).
VIT_MODEL_PATH="../ModelZoo/vit-iqa-pretrained/"

deepspeed --include localhost:$1 --master_port 6688 src/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --vit_model_path $VIT_MODEL_PATH \
    --dataset_type pair \
    --level_names excellent good fair poor bad \
    --softkl_loss True \
    --weight_rank 1.0 \
    --weight_softkl 1.0 \
    --weight_next_token 0.05 \
    --continuous_rating_loss True \
    --closeset_rating_loss True \
    --use_fix_std True \
    --detach_pred_std True \
    --data_paths ../Data-DeQA-Score/KONIQ/metas/train_koniq_7k.json \
                 ../Data-DeQA-Score/SPAQ/metas/train_spaq_9k.json \
                 ../Data-DeQA-Score/KADID10K/metas/train_kadid_8k.json \
    --data_weights 1 1 1 \
    --image_folder ../Data-DeQA-Score/ \
    --output_dir ./checkpoints/vit-iqa-mix3/ \
    --image_aspect_ratio pad \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --freeze_vision_model True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
