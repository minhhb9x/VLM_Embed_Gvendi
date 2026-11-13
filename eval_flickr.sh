CUDA_VISIBLE_DEVICES=1 python eval_flickr.py \
    --model_name raghavlite/B3_Qwen2_7B \
    --model_backbone qwen2_vl \
    --lora True \
    --lora_r 8 \
    --pooling eos \
    --normalize True \
    --bf16 \
    --dataset_name "nlphuji/flickr30k" \
    --dataset_split test \
    --image_dir "vlm2vec_external/flickr30k" \
    --image_resolution tiny \
    --per_device_eval_batch_size 4 \
    --encode_output_path "MMEB-eval_outputs/B3_Qwen2_7B" \
    --report_to none

# CUDA_VISIBLE_DEVICES=0 python eval_flickr.py \
#     --model_name Qwen/Qwen3-VL-Embedding-8B \
#     --model_backbone qwen3_vl \
#     --pooling eos \
#     --normalize True \
#     --bf16 \
#     --dataset_name "nlphuji/flickr30k" \
#     --dataset_split test \
#     --image_dir "vlm2vec_external/flickr30k" \
#     --image_resolution tiny \
#     --per_device_eval_batch_size 16 \
#     --encode_output_path "MMEB-eval_outputs/Qwen3-VL-Embedding-8B" \
#     --report_to none