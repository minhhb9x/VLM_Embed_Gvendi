#!/bin/bash
set -euo pipefail
# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
# CACHE_DIR contains qry.pt and pos.pt for each encoded_dir.
CACHE_DIR="${CACHE_DIR:-caching/B3_Qwen2_2B_flickr30k}"
if [[ ! -d "$CACHE_DIR" ]]; then
    printf 'PGA2 teacher cache directory not found: %s\n' "$CACHE_DIR" >&2
    exit 1
fi

# Chạy script này từ thư mục gốc repository.
TRAIN_SCRIPT="train_ddp.py"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================================================================
# Training
# =========================================================================

torchrun --standalone \
    --nproc_per_node="$NUM_GPUS_PER_NODE" "$TRAIN_SCRIPT" \
    --model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --lora "True" \
    --lora_r "64" \
    --lora_alpha "64" \
    --model_backbone "llava_onevision" \
    --pooling "eos" \
    --dataset_name "nlphuji/flickr30k" \
    --dataset_split "train" \
    --image_dir "vlm2vec_external/flickr30k" \
    --flickr_caption_mode "first" \
    --flickr_caption_seed "42" \
    --percent_data "1.0" \
    --output_dir "training/B3-Qwen2-2B_Llava-OV-0.5B_pga2_flickr30k" \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
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
    --kd_loss_type "pga2" \
    --caching_dir "$CACHE_DIR" \
    --teacher_hidden_dim 1536 \
    --image_resolution "tiny" \
    --projector_config_path "config/projector_pga.json" \
    --student_hidden_dim "896" \
    --report_to "none"

# =========================================================================
# Flickr30k retrieval evaluation
# =========================================================================

EVAL_BATCH_SIZE=16
OUTPUT_PATH="./Flickr30k-eval_outputs/B3-Qwen2-2B_Llava-OV-0.5B_pga2_flickr30k"

python eval_flickr.py \
    --model_name "training/B3-Qwen2-2B_Llava-OV-0.5B_pga2_flickr30k/checkpoint-final" \
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
    --load_pretrained_lora "True" \
    --report_to "none"
