#!/bin/bash

NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_ddp_muon.py"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/FastVLM-0.5B_talas_1.0_eos_muon_lr8e-5_cls" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --muon_lr 8e-5 \
    --adjust_lr_fn "match_rms_adamw" \
    --muon_weight_decay 0.01 \
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
    --warmup_ratio 0.05 \
    --kd_weight 1.0 \
    --caching_dir "caching/B3_Qwen2_2B_cls" \
    --kd_loss_type "talas" \
    --image_resolution "low" \
    --num_projectors 1 \
    --num_self_kd_layers 3 \
    --projector_lr 5e-5 \
    --report_to None
