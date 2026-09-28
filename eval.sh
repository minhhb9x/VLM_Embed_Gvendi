SUBSETS=(
  # "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
  # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  "ScienceQA" "VizWiz" "GQA" "TextVQA"
)

# MODEL=training/FastVLM-0.5B_base_16_eos_cls/checkpoint-final
MODEL=meta_train/span_propose_llava_ov_vqa_v2_final


CUDA_VISIBLE_DEVICES=1 python eval_mmeb_2.py \
    --model_name $MODEL \
    --encode_output_path './MMEB-eval_outputs/span_propose_llava_ov_vqa_v2_final_2_24/' \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_onevision_old \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 2 \
    --image_resolution "tiny" \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none