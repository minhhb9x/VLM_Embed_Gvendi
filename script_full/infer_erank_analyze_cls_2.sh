

INFER_SUBSETS=(
    "ImageNet-1K"
)

INFER_SCRIPT="infer_eval_hidden_attention.py"


python $INFER_SCRIPT \
    --model_name "training/FastVLM-0.5B_simcse_cka_cls/checkpoint-epoch-0" \
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
    --encode_output_path "infer/FastVLM-0.5B_simcse_cka_cls" \
    --per_device_eval_batch_size 8 \
    --load_pretrained_lora True \
    --report_to None

# analyze erank
python ./er_statistic.py \
    --pt_dir "infer/FastVLM-0.5B_simcse_cka_cls"/${INFER_SUBSETS[0]}/query \
    --start_idx 0 \
    --end_idx 49 \
    --normalize \
    --output_file "analyze/FastVLM-0.5B_simcse_cka_cls.txt"

