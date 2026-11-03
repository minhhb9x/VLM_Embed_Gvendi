import os
import torch
from tqdm import tqdm

BASE_ROOT = "caching/Qwen3VL_8B"

subsets = sorted([
    x for x in os.listdir(BASE_ROOT)
    if os.path.isdir(os.path.join(BASE_ROOT, x))
])

bad_files = []
total_folders = 0

for subset in subsets:
    root = os.path.join(BASE_ROOT, subset)

    folders = sorted(
        [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))],
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )

    total_folders += len(folders)
    subset_bad = 0

    for folder in tqdm(folders, desc=subset):
        for name in ["qry.pt", "pos.pt"]:
            path = os.path.join(root, folder, name)

            try:
                torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                bad_files.append((subset, folder, name, path, repr(e)))
                subset_bad += 1

    print(f"{subset}: {len(folders)} folders, {subset_bad} bad files")

print(f"\nChecked {len(subsets)} subsets")
print(f"Total folders: {total_folders}")
print(f"Bad files: {len(bad_files)}")

for subset, folder, name, path, error in bad_files:
    print(f"\nSubset: {subset}")
    print(f"Folder: {folder}")
    print(f"File:   {name}")
    print(f"Path:   {path}")
    print(f"Error:  {error}")