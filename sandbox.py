import os
from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS, LLAVA_ONEVISION, \
    Qwen2_VL_process_fn, LLAVA_QWEN2, FastVLM_process_fn, Llava_ONEVISION_process_fn, QWEN3_VL, Qwen3_VL_process_fn
from src.utils import batch_to_device
from PIL import Image
import numpy as np
from src.model.llava.model import LlavaQwen2ForCausalLM
import torch
import math

import torch.nn.functional as F

from transformers.image_transforms import (
    convert_to_rgb,
    resize,
)

# model_args = ModelArguments(
#     model_name='raghavlite/B3_Qwen2_2B',
#     pooling='last',
#     normalize=True,
#     model_backbone='qwen2_vl',
#     lora=True
# )
# process_fn = Qwen2_VL_process_fn
# token_img = QWEN2_VL


# model_args = ModelArguments(
#     model_name='apple/FastVLM-0.5B',
#     pooling='last',
#     normalize=True,
#     model_backbone=LLAVA_QWEN2,
#     lora=True,
# )
# process_fn = FastVLM_process_fn
# token_img = LLAVA_QWEN2


model_args = ModelArguments(
    model_name='llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
    pooling='last',
    normalize=True,
    model_backbone=LLAVA_ONEVISION,
    # lora=True,
)
data_args = DataArguments(
    image_resolution = 'tiny'
)
process_fn = Llava_ONEVISION_process_fn
token_img = LLAVA_ONEVISION


# model_args = ModelArguments(
#     model_name='Qwen/Qwen3-VL-Embedding-8B',
#     pooling='last',
#     normalize=True,
#     model_backbone=QWEN3_VL,
# )
# data_args = DataArguments(
#     image_resolution = 'low'
# )
# process_fn = Qwen3_VL_process_fn
# token_img = QWEN3_VL



processor = load_processor(model_args, None)
model = MMEBModel.build(model_args)
model = model.to('cuda:0', dtype=torch.bfloat16)
model.eval()

processor_inputs = {
    "text": [f'Represent the given image with the following question: What is in the image {VLM_IMAGE_TOKENS[token_img]}',
          f'Represent the given image with the following question: What is in the image'],
    "images": [Image.open('example.jpg').resize((336, 336))],
}

inputs = process_fn(
    processor_inputs,
    processor, 
    # square_padding=True
    )
inputs = batch_to_device(inputs, "cuda")

decoded_texts = processor.tokenizer.batch_decode(
    inputs["input_ids"].cpu(),
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)

print(inputs.keys())
for i, text in enumerate(decoded_texts):
    print(f"\n{'=' * 30} SAMPLE {i} {'=' * 30}")
    print(text)


tok = processor.tokenizer
ids = inputs["input_ids"].cpu()
mask = inputs["attention_mask"].cpu()

tok = processor.tokenizer

prompt = processor.apply_chat_template(
    [
        {"role": "system", "content": "Represent the user's input."},
        {"role": "user", "content": "Hello"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

print("After chat template:", repr(prompt[-80:]))

for add_special in [False, True]:
    ids = tok(
        prompt,
        add_special_tokens=add_special,
    )["input_ids"]

    print(
        f"add_special_tokens={add_special}:",
        tok.convert_ids_to_tokens(ids[-5:]),
    )

# output = model.encode_input(inputs)[0]

# print(output.shape)