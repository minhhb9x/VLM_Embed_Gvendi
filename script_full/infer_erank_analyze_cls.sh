# ============================================================
# Usage:
# ./infer_erank_analyze_cls.sh 1 1 1 1 1.0 0.05
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
SIGREG_WEIGHT=${6:-0.05}
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

MODEL="training/FastVLM-0.5B_cls_${EXP_NAME}/checkpoint-epoch-0"
# MODEL="training/llava_ov-0.5B_cls_${EXP_NAME}/checkpoint-epoch-0"



INFER_SUBSETS=(
    "ImageNet-1K"
#   "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
#   "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  # "ScienceQA" "VizWiz" "GQA" "TextVQA"
)

INFER_SCRIPT="infer_eval_hidden_attention.py"


python $INFER_SCRIPT \
    --model_name $MODEL \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name vlm2vec_eval/MMEB-eval \
    --subset_name "${INFER_SUBSETS[0]}" \
    --dataset_split "test" \
    --image_dir "eval_images/" \
    --tgt_prefix_mod \
    --encode_output_path "infer/FastVLM-0.5B_${EXP_NAME}" \
    --per_device_eval_batch_size 8 \
    --load_pretrained_lora True \
    --report_to None

# analyze erank
echo "normalize"
python ./er_statistic.py \
    --pt_dir "infer/FastVLM-0.5B_${EXP_NAME}"/${INFER_SUBSETS[0]}/query \
    --start_idx 0 \
    --end_idx 49 \
    --normalize \
    --output_file "analyze/FastVLM-0.5B_${EXP_NAME}.txt"

echo "no normalize"
python ./er_statistic.py \
    --pt_dir "infer/FastVLM-0.5B_${EXP_NAME}"/${INFER_SUBSETS[0]}/query \
    --start_idx 0 \
    --end_idx 49 \
    --output_file "analyze/FastVLM-0.5B_${EXP_NAME}.txt"

