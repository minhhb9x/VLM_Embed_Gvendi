#!/bin/bash

set -e

# ============================================================
# Usage:
# ./run_talas_jepa_cls.sh 1 1 1 1 1.0 0.05
#
# Args:
#   1. use_distill_loss
#   2. use_distill_cse_loss
#   3. use_distill_vison_loss
#   4. use_sigreg_loss
#   5. kd_weight
#   6. sigreg_weight
# ============================================================

USE_DISTILL_LOSS=${1:-True}
USE_DISTILL_CSE_LOSS=${2:-True}
USE_DISTILL_VISON_LOSS=${3:-True}
USE_SIGREG_LOSS=${4:-True}
KD_WEIGHT=${5:-1.0}
SIGREG_WEIGHT=${6:-0.5}
NUM_LAYER=${7:-1}
D_TAU=${8:-0.02}

# ============================================================
# Convert True/False -> 1/0 cho tên folder
# ============================================================

bool_to_python() {
    case "$1" in
        1|true|True|TRUE)
            echo "True"
            ;;
        0|false|False|FALSE)
            echo "False"
            ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}

bool_to_int() {
    case "$1" in
        1|true|True|TRUE)
            echo "1"
            ;;
        0|false|False|FALSE)
            echo "0"
            ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}


# Python values
DISTILL_LOSS_BOOL=$(bool_to_python "$USE_DISTILL_LOSS")
DISTILL_CSE_BOOL=$(bool_to_python "$USE_DISTILL_CSE_LOSS")
DISTILL_VISION_BOOL=$(bool_to_python "$USE_DISTILL_VISON_LOSS")
SIGREG_BOOL=$(bool_to_python "$USE_SIGREG_LOSS")

# Folder values
D_DISTILL=$(bool_to_int "$USE_DISTILL_LOSS")
D_CSE=$(bool_to_int "$USE_DISTILL_CSE_LOSS")
D_VISION=$(bool_to_int "$USE_DISTILL_VISON_LOSS")
D_SIGREG=$(bool_to_int "$USE_SIGREG_LOSS")


# ============================================================
# Tên experiment
# ============================================================

EXP_NAME="talas_jepa_v2_d${D_DISTILL}_cse${D_CSE}_vis${D_VISION}_sig${D_SIGREG}_kd${KD_WEIGHT}_sw${SIGREG_WEIGHT}_l${NUM_LAYER}_dt${D_TAU}"

OUTPUT_DIR="training/Qwen2_7B_FastVLM-0.5B_cls_${EXP_NAME}_M256"
CACHE_DIR="caching/B3_Qwen2_7B_cls"

echo "============================================================"
echo "Experiment:"
echo "  USE_DISTILL_LOSS      = $USE_DISTILL_LOSS"
echo "  USE_DISTILL_CSE_LOSS  = $USE_DISTILL_CSE_LOSS"
echo "  USE_DISTILL_VISON_LOSS= $USE_DISTILL_VISON_LOSS"
echo "  USE_SIGREG_LOSS       = $USE_SIGREG_LOSS"
echo "  KD_WEIGHT             = $KD_WEIGHT"
echo "  SIGREG_WEIGHT         = $SIGREG_WEIGHT"
echo ""
echo "OUTPUT_DIR:"
echo "  $OUTPUT_DIR"
echo "============================================================"


# ============================================================
# 1. TRAIN
# ============================================================

NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_ddp.py"

#teacher model name not affect training. Still keep B3_Qwen2_2B but change the cache_dir to B3_Qwen2_7B

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
    --output_dir "$OUTPUT_DIR" \
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
    --warmup_ratio 0.05 \
    --caching_dir "$CACHE_DIR" \
    --kd_loss_type "talas_jepa" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config_emo.json" \
    --num_self_kd_layers 3 \
    --projector_lr 5e-4 \
    --report_to None \
    --use_distill_loss "$DISTILL_LOSS_BOOL" \
    --use_distill_cse_loss "$DISTILL_CSE_BOOL" \
    --use_distill_vison_loss "$DISTILL_VISION_BOOL" \
    --use_sigreg_loss "$SIGREG_BOOL" \
    --kd_weight "$KD_WEIGHT" \
    --sigreg_weight "$SIGREG_WEIGHT" \
    --num_layers "$NUM_LAYER" \
    --d_cse_temperature "$D_TAU"


# ============================================================
# 2. CHECKPOINT
# ============================================================

MODEL="$OUTPUT_DIR/checkpoint-epoch-0"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: Checkpoint not found:"
    echo "$MODEL"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training finished."
echo "Checkpoint:"
echo "  $MODEL"
echo "============================================================"


# ============================================================
# 3. EVAL
# ============================================================

SUBSETS=(
    "ImageNet-1K"
    "N24News"
    "HatefulMemes"
    "VOC2007"
    "SUN397"
    # "Place365"
    # "ImageNet-A"
    # "ImageNet-R"
    # "ObjectNet"
    # "Country211"
)

EVAL_OUTPUT="./MMEB-eval_outputs/Qwen2_7B_FastVLM-0.5B_cls_${EXP_NAME}_M256/"

python eval_mmeb.py \
    --model_name "$MODEL" \
    --encode_output_path "$EVAL_OUTPUT" \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 8 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none

echo ""
echo "============================================================"
echo "DONE"
echo "Experiment:"
echo "  $EXP_NAME"
echo ""
echo "Train:"
echo "  $OUTPUT_DIR"
echo ""
echo "Eval:"
echo "  $EVAL_OUTPUT"
echo "============================================================"

