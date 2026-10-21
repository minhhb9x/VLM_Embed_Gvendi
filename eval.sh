SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
  "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  # "ScienceQA" "VizWiz" "GQA" "TextVQA"
)

# MODEL=training/FastVLM-0.5B_base_16_eos_cls/checkpoint-final
MODEL=training/FastVLM-0.5B_cls_gvendi_cse0_grad1_kd-w1.0_grad-w1.0_l1/checkpoint-epoch-0


python eval_mmeb.py \
    --model_name $MODEL \
    --encode_output_path './MMEB-eval_outputs/FastVLM-0.5B_cls_gvendi_cse0_grad1_kd-w1.0_grad-w1.0_l1/' \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none