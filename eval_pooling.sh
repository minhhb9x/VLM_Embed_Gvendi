SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
    # "ImageNet-1K"
#   "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA"
)

MODEL=training/B3_Qwen2_2B_pooling/checkpoint-epoch-0

python eval_mmeb_pooling.py \
    --model_name $MODEL \
    --encode_output_path ./MMEB-eval_outputs/B3_Qwen2_2B/ \
    --lora True --lora_r 8 --lora_alpha 64 \
    --pooling eos \
    --model_backbone qwen2_vl \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 1 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --modality_gated_pooling True \
    --report_to none