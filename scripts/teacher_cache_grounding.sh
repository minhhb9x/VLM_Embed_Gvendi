#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="teacher_cache.py"

# SUBSETS=(
#   "VOC2007"
#   "OK-VQA"
# )

SUBSETS=(
  # "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA"
  "MSCOCO"
)

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
CUDA_VISIBLE_DEVICES=0 \
torchrun  --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name raghavlite/B3_Qwen2_2B \
    --lora True \
    --lora_r 8 \
    --lora_alpha 64 \
    --model_backbone "qwen2_vl" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "caching/B3_Qwen2_2B_grounding" \
    --per_device_train_batch_size 4 \
    --image_resolution "mid" \
    --seed 42 \
    --normalize False \
    --report_to "none" 