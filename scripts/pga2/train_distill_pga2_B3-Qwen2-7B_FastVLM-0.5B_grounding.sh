#!/bin/bash
set -euo pipefail
# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
# CACHE_DIR contains qry.pt and pos.pt for each encoded_dir.
CACHE_DIR="${CACHE_DIR:-caching/B3_Qwen2_7B_grounding}"
if [[ ! -d "$CACHE_DIR" ]]; then
    printf 'PGA2 teacher cache directory not found: %s\n' "$CACHE_DIR" >&2
    exit 1
fi

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_ddp.py"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================

torchrun --standalone \
    --nproc_per_node="$NUM_GPUS_PER_NODE" "$TRAIN_SCRIPT" \
    --model_name "apple/FastVLM-0.5B" \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "MSCOCO" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/B3-Qwen2-7B_FastVLM-0.5B_pga2_grounding" \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
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
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 1 \
    --kd_loss_type "pga2" \
    --caching_dir "$CACHE_DIR" \
    --teacher_hidden_dim 3584 \
    --image_resolution "low" \
    --projector_lr 5e-4 \
    --projector_config_path "config/projector_pga.json" \
    --report_to none



EVAL_SUBSETS=(
    MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing
)


EVAL_BATCH_SIZE=16
OUTPUT_BASE="./MMEB-eval_outputs/B3-Qwen2-7B_FastVLM-0.5B_pga2_grounding"

for BATCH_SIZE in \
    "$EVAL_BATCH_SIZE" \
    "$((EVAL_BATCH_SIZE - 1))" \
    "$((EVAL_BATCH_SIZE - 2))"
do
    if [ "$BATCH_SIZE" -lt 1 ]; then
        continue
    fi

    OUTPUT_PATH="${OUTPUT_BASE}_b${BATCH_SIZE}"

    echo "======================================"
    echo "Evaluating with batch size: $BATCH_SIZE"
    echo "Output: $OUTPUT_PATH"
    echo "======================================"

    rm -rf "$OUTPUT_PATH"

    python eval_mmeb.py \
      --model_name "training/B3-Qwen2-7B_FastVLM-0.5B_pga2_grounding/checkpoint-epoch-0" \
      --encode_output_path "$OUTPUT_PATH" \
      --lora True \
      --lora_r 64 \
      --lora_alpha 64 \
      --pooling eos \
      --model_backbone llava_qwen2 \
      --normalize True \
      --bf16 \
      --dataset_name TIGER-Lab/MMEB-eval \
      --subset_name "${EVAL_SUBSETS[@]}" \
      --dataset_split test \
      --per_device_eval_batch_size "$BATCH_SIZE" \
      --image_dir eval_images/ \
      --tgt_prefix_mod \
      --load_pretrained_lora True \
      --report_to none

done
