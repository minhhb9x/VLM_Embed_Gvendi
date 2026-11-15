#!/bin/bash

NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_distill_ddp.py"

EOS_PROJECTION_SPACE="student"
TEMPERATURE=0.02
EOS_KD_WEIGHT=0.3
ER_KD_WEIGHT=0.7

SUBSETS=(
  "MSCOCO"
)

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/B3-Qwen2-2B_FastVLM-0.5B_idr_vg" \
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
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.03 \
    --temperature "${TEMPERATURE}" \
    --kd_loss_type "eos_er_combined_loss" \
    --eos_projection_space "${EOS_PROJECTION_SPACE}" \
    --eos_kd_weight "${EOS_KD_WEIGHT}" \
    --er_kd_weight "${ER_KD_WEIGHT}" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5 \
    --ddp_find_unused_parameters True \
    --report_to None
