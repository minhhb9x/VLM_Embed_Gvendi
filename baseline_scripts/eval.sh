SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
  # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  # "ScienceQA" "VizWiz" "GQA" "TextVQA"
)

MODEL=Qwen/Qwen3-VL-Embedding-2B

CUDA_VISIBLE_DEVICES=0 \
python eval_mmeb.py \
    --model_name $MODEL \
    --encode_output_path './MMEB-eval_outputs/Qwen3-VL-Embedding-2B_cls/' \
    --pooling eos \
    --model_backbone qwen3_vl \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 8 \
    --image_resolution mid \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --report_to none