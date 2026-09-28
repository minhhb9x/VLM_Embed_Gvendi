import os
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download


# Danh sách zip file cần tải
files = {
    # "ImageNet_1K.zip": "images_zip/ImageNet_1K.zip",
    # "N24News.zip": "images_zip/N24News.zip",
    # "SUN397.zip": "images_zip/SUN397.zip",
    # "HatefulMemes.zip": "images_zip/HatefulMemes.zip",
    # "VOC2007.zip": "images_zip/VOC2007.zip",
    # "OK-VQA": "images_zip/OK-VQA.zip",
    # "A-OKVQA": "images_zip/A-OKVQA.zip",
    # "DocVQA": "images_zip/DocVQA.zip",
    # "InfographicsVQA": "images_zip/InfographicsVQA.zip",
    # "ChartQA": "images_zip/ChartQA.zip",
    # "Visual7W": "images_zip/Visual7W.zip"
    "MSCOCO": "images_zip/MSCOCO.zip"
}

dataset = "TIGER-Lab/MMEB-train"

download_dir = Path("./downloads")
save_dir = Path("./vlm2vec_train/MMEB-train/images")

download_dir.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)


def unzip_python(zip_path: str | Path, output_dir: str | Path, remove_zip: bool = True):
    """
    Giải nén zip bằng thư viện Python built-in.
    Không cần lệnh unzip, không cần sudo.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    print(f"📦 Unzipping {zip_path} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    if remove_zip:
        zip_path.unlink()

    print(f"✔️ Unzipped & removed {zip_path}")


def get_local_zip_path(repo_path: str) -> Path:
    """
    hf_hub_download với local_dir sẽ lưu theo cấu trúc:
    ./downloads/images_zip/ImageNet_1K.zip
    """
    return download_dir / repo_path


# Tải từng file, nếu zip đã có thì skip
local_paths = []

for name, repo_path in files.items():
    local_zip_path = get_local_zip_path(repo_path)

    if local_zip_path.exists():
        print(f"⏭️ Skip download, already exists: {local_zip_path}")
        local_paths.append(local_zip_path)
        continue

    print(f"⬇️ Downloading {name} ...")

    downloaded = hf_hub_download(
        repo_id=dataset,
        filename=repo_path,
        local_dir=download_dir,
        repo_type="dataset",
    )

    local_paths.append(Path(downloaded))


# Giải nén song song bằng Python threads
max_workers = min(len(local_paths), os.cpu_count() or 4)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(unzip_python, zip_file, save_dir, True)
        for zip_file in local_paths
    ]

    for future in as_completed(futures):
        future.result()

print("🎉 All done!")