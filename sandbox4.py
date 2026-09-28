import os
import torch
from tqdm import tqdm

# ROOT = "caching/B3_Qwen2_7B_cls/VOC2007"
# ROOT = "caching/B3_Qwen2_7B_cls/HatefulMemes"
# ROOT = "caching/B3_Qwen2_7B_cls/SUN397"
# ROOT = "caching/B3_Qwen2_7B_cls/ImageNet_1K"
ROOT = "caching/B3_Qwen2_7B_cls/N24News"

folders = sorted(
    [x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x))],
    key=lambda x: int(x) if x.isdigit() else x,
)

bad_files = []

for folder in tqdm(folders):
    for name in ["qry.pt", "pos.pt"]:
        path = os.path.join(ROOT, folder, name)

        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            bad_files.append((folder, name, path, repr(e)))

print(f"\nChecked {len(folders)} folders")
print(f"Bad files: {len(bad_files)}")

for folder, name, path, error in bad_files:
    print(f"\nFolder: {folder}")
    print(f"File:   {name}")
    print(f"Path:   {path}")
    print(f"Error:  {error}")