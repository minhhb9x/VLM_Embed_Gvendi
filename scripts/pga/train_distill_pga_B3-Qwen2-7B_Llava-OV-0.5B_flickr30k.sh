#!/bin/bash
set -euo pipefail
# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1
PORT=12345

# Chạy script này từ thư mục gốc repository.
TRAIN_SCRIPT="train_distill_ddp.py"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================================================================
# Training
# =========================================================================

torchrun --standalone \
    --nproc_per_node="$NUM_GPUS_PER_NODE" "$TRAIN_SCRIPT" \
    --model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --teacher_model_name "raghavlite/B3_Qwen2_7B" \
    --lora "True" \
    --teacher_lora "True" \
    --lora_r "64" \
    --lora_alpha "64" \
    --teacher_lora_r "8" \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_onevision" \
    --pooling "eos" \
    --dataset_name "nlphuji/flickr30k" \
    --dataset_split "train" \
    --image_dir "vlm2vec_external/flickr30k" \
    --flickr_caption_mode "cycle" \
    --flickr_caption_seed "42" \
    --percent_data "1.0" \
    --output_dir "training/B3-Qwen2-7B_Llava-OV-0.5B_pga_flickr30k" \
    --per_device_train_batch_size "16" \
    --gradient_accumulation_steps "1" \
    --learning_rate "1e-4" \
    --num_train_epochs "1" \
    --bf16 \
    --save_total_limit "5" \
    --logging_steps "1" \
    --save_strategy "epoch" \
    --seed "42" \
    --weight_decay "0.01" \
    --normalize "True" \
    --teacher_normalize "True" \
    --lr_scheduler_type "cosine" \
    --warmup_ratio "0.03" \
    --kd_weight "1" \
    --kd_loss_type "pga" \
    --image_resolution "tiny" \
    --teacher_image_resolution "low" \
    --projector_config_path "config/projector_pga.json" \
    --teacher_hidden_dim "3584" \
    --student_hidden_dim "896" \
    --report_to "none"

# =========================================================================
# Flickr30k retrieval evaluation
# =========================================================================

EVAL_BATCH_SIZE=16
OUTPUT_PATH="./Flickr30k-eval_outputs/B3-Qwen2-7B_Llava-OV-0.5B_pga_flickr30k"

python evaluation/mmeb_baselines/eval_flickr.py \
    --model_name "training/B3-Qwen2-7B_Llava-OV-0.5B_pga_flickr30k/checkpoint-final" \
    --encode_output_path "$OUTPUT_PATH" \
    --lora "True" \
    --lora_r "64" \
    --lora_alpha "64" \
    --pooling "eos" \
    --model_backbone "llava_onevision" \
    --normalize "True" \
    --bf16 \
    --dataset_name "nlphuji/flickr30k" \
    --dataset_split "test" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --image_dir "vlm2vec_external/flickr30k" \
    --image_resolution "tiny" \
    --tgt_prefix_mod \
    --load_pretrained_lora "True" \
    --report_to "none"
