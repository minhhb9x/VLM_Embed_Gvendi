import torch

path = "caching/Qwen3VL_8B/ImageNet_1K/0/qry.pt"
cache = torch.load(path, map_location="cpu", weights_only=True)

for key, value in cache.items():
    print(f"\n{key}:")
    if torch.is_tensor(value):
        print(f"shape={tuple(value.shape)}, dtype={value.dtype}")
    print(value)