import sys

import torch
import torch.nn.functional as F
from PIL import Image

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.model import MMEBModel
from src.model.processor import QWEN3_VL, Qwen3_VL_process_fn
from src.model.vlm_backbone.qwen3_vl_embedding import (
    Qwen3VLForEmbedding,
    Qwen3VLProcessor,
)
from src.model.processor import load_processor


model_name = "Qwen/Qwen3-VL-Embedding-2B"
device = "cuda"


model_args = ModelArguments(
    model_name=model_name,
    pooling='last',
    normalize=True,
    model_backbone='qwen3_vl',
    lora=False,
)

model = MMEBModel.load(model_args).eval().to(device)
processor = load_processor(model_args, None)

image = Image.open("example.jpg").convert("RGB")

inputs = Qwen3_VL_process_fn(
    model_inputs={
        "text": ["<|image_pad|> Represent this image", "<|image_pad|> Represent this image"],
        "images": [[image.resize((128, 128))], [image.resize((128, 64))]],
    },
    processor=processor
)
inputs = {key: value.to(device) for key, value in inputs.items()}

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

tokenizer = processor.tokenizer

for i in range(input_ids.size(0)):
    ids = input_ids[i]
    mask = attention_mask[i].bool()

    valid_ids = ids[mask]
    valid_tokens = tokenizer.convert_ids_to_tokens(valid_ids.tolist())
    decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)

    print(f"\n{'=' * 80}")
    print(f"SAMPLE {i}")
    print(f"{'=' * 80}")

    print("Sequence length:", ids.numel())
    print("Valid length:", mask.sum().item())

    print("\nINPUT IDS:")
    print(valid_ids.tolist())

    print("\nTOKENS:")
    for j, (token_id, token) in enumerate(zip(valid_ids.tolist(), valid_tokens)):
        print(f"{j:4d} | {token_id:6d} | {repr(token)}")

    print("\nDECODED:")
    print(decoded)

with torch.inference_mode():
    pooled_output, _, attention_matrix, output_hidden_states = model.encode_input(inputs)

hidden = output_hidden_states[-1]

print("\n" + "=" * 80)
print("HIDDEN STATE CHECK")
print("=" * 80)

print("input_ids shape:     ", tuple(inputs["input_ids"].shape))
print("attention_mask shape:", tuple(inputs["attention_mask"].shape))
print("hidden shape:        ", tuple(hidden.shape))

print("\nPadded sequence length:")
print("input_ids:", inputs["input_ids"].shape[1])
print("hidden:   ", hidden.shape[1])
print("same:", inputs["input_ids"].shape[1] == hidden.shape[1])

for i in range(hidden.size(0)):
    mask = inputs["attention_mask"][i].bool()

    num_input_tokens = inputs["input_ids"][i].numel()
    num_valid_tokens = mask.sum().item()
    num_hidden_tokens = hidden[i].shape[0]
    num_valid_hidden_tokens = hidden[i][mask].shape[0]

    print(f"\nSAMPLE {i}")
    print(f"input tokens (with padding):  {num_input_tokens}")
    print(f"valid input tokens:           {num_valid_tokens}")
    print(f"hidden tokens (with padding): {num_hidden_tokens}")
    print(f"valid hidden tokens:          {num_valid_hidden_tokens}")

    print(
        "valid lengths match:",
        num_valid_tokens == num_valid_hidden_tokens
    )

print(processor.image_token_id)