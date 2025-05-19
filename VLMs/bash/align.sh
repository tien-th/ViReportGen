#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat
export PYTHONPATH=/path/to/this/repo_folder:/path/to/this/repo_folder/llava/model/multimodal_encoder:$PYTHONPATH
# MODEL_VERSION=llava-med-v1.5-mistral-7b

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v1
########### DO NOT CHANGE ###########

deepspeed --num_gpus=4 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /path/to/base/LLM/model \
    --version $PROMPT_VERSION \
    --data_path /path/to/data/file \
    --eval_data_path /path/to/data/file \
    --image_folder /path/to/image/folder \
    --vision_tower /path/to/pretrained/vision_encoder/model \
    --pretrain_mm_mlp_adapter /path/to/pretrained/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /path/to/output/folder \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

