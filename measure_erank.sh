#!/usr/bin/env bash
set -euo pipefail

# bash measure_erank.sh student <checkpoint> [max_batches] [batch_size] [num_layers]
# bash measure_erank.sh teacher [teacher_repo] [max_batches] [batch_size] [num_layers]
TYPE="${1:?Usage: bash measure_erank.sh student|teacher [checkpoint] [max_batches] [batch_size] [num_layers]}"
PATH_OR_REPO="${2:-}"
MAX_BATCHES="${3:-5}"
BATCH_SIZE="${4:-8}"
NUM_LAYERS="${5:-1}"
PROJ_DIM="${PROJ_DIM:-512}"
PROJ_BLOCK_SIZE="${PROJ_BLOCK_SIZE:-4096}"

case "$TYPE" in
  student)
    : "${PATH_OR_REPO:?Pass student checkpoint as the second argument}"
    MODEL="apple/FastVLM-0.5B"
    BACKBONE="llava_qwen2"
    RANK=64
    EXTRA=(--checkpoint_path "$PATH_OR_REPO")
    RUN_ID="$(basename "$(dirname "$PATH_OR_REPO")")"
    ;;
  teacher)
    MODEL="${PATH_OR_REPO:-raghavlite/B3_Qwen2_2B}"
    BACKBONE="qwen2_vl"
    RANK=8
    EXTRA=()
    RUN_ID="$(basename "$MODEL")"
    ;;
  *) echo "Expected student or teacher" >&2; exit 1 ;;
esac

python measure_erank.py \
  --model_name "$MODEL" \
  --model_backbone "$BACKBONE" \
  --lora_r "$RANK" --lora_alpha 64 \
  --max_batches "$MAX_BATCHES" \
  --batch_size "$BATCH_SIZE" \
  --num_layers "$NUM_LAYERS" \
  --proj_dim "$PROJ_DIM" \
  --proj_block_size "$PROJ_BLOCK_SIZE" \
  --seed 42 \
  --output "erank_results/${TYPE}_${RUN_ID}_b${MAX_BATCHES}_bs${BATCH_SIZE}_l${NUM_LAYERS}.json" \
  "${EXTRA[@]}"
