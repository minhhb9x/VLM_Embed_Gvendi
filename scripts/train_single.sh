#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=1
# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_ddp.py"

# SUBSETS=(
#   "VOC2007"
#   "OK-VQA"
# )

SUBSETS=(
  "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
#   "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
)

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "training/FastVLM-0.5B_base_16_eos_cls" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --kd_loss_type "contrastive" \
    --image_resolution "low" \
    --report_to "none" 