#!/usr/bin/env bash
# Run from VLM_Embed_DAVED repository root.
set -euo pipefail

BACKBONE=${1:-qwen3_vl}
MODEL=${2:-Qwen/Qwen3-VL-Embedding-8B}
OUTPUT=${3:-caching/Qwen3VL_8B}
BATCH_SIZE=${4:-4}
RESOLUTION=${5:-mid}
LORA=${6:-False}
MAX_BATCHES=${7:-0}
CHECKPOINT=${8:-}
OVERWRITE=${9:-False}
REQUIRE_ATTENTION=${10:-False}

SUBSETS=(
    ImageNet_1K N24News HatefulMemes VOC2007 SUN397
    OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W
    MSCOCO    
)
CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
    CHECKPOINT_ARGS=(--checkpoint_path "$CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node="${NUM_GPUS_PER_NODE:-1}" teacher_cache_multi.py \
  --model_name "$MODEL" \
  --model_backbone "$BACKBONE" \
  "${CHECKPOINT_ARGS[@]}" \
  --lora "$LORA" \
  --pooling eos \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name "${SUBSETS[@]}" \
  --dataset_split original \
  --image_dir vlm2vec_train/MMEB-train \
  --output_dir "$OUTPUT" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --image_resolution "$RESOLUTION" \
  --seed 42 \
  --normalize False \
  --max_batches "$MAX_BATCHES" \
  --overwrite_cache "$OVERWRITE" \
  --require_attention "$REQUIRE_ATTENTION" \
  --report_to none
