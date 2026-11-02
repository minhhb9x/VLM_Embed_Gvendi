"""Local nlphuji/flickr30k CSV adapter; no Hugging Face download is needed.

The CSV's ``split`` column is authoritative. The upstream HF builder exposes
all rows through a single HF split named "test", which is not the test set.
These map-style datasets work with torch DataLoader without subclassing Dataset.
"""
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


IMAGE_INSTRUCTION = "Find an image caption describing the given image."
CAPTION_PREFIX = "Represent the image caption: "


def is_flickr30k(dataset_name):
    return str(dataset_name).lower() in {"nlphuji/flickr30k", "flickr30k"}


def canonical_split(split):
    split = str(split).strip().lower()
    split = {"dev": "val", "valid": "val", "validation": "val"}.get(split, split)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Flickr30k split must be train, val or test, got {split!r}")
    return split


@dataclass(frozen=True)
class Flickr30kRecord:
    filename: str
    image_path: str
    captions: tuple
    split: str


def load_flickr30k_records(image_dir, split, annotation_file=None, *, verify_images=True):
    """Accept either the dataset root or its extracted flickr30k-images folder.

    Resolve the common root/images/images nesting too. Fail explicitly on missing
    images, malformed captions, duplicate filenames or cross-split leakage.
    """
    split = canonical_split(split)
    if not image_dir:
        raise ValueError("Pass --image_dir pointing to your local Flickr30k folder")
    root = Path(image_dir).expanduser().resolve()
    if annotation_file:
        annotation = Path(annotation_file).expanduser().resolve()
    else:
        candidates = [root / "flickr_annotations_30k.csv",
                      root.parent / "flickr_annotations_30k.csv",
                      root.parent.parent / "flickr_annotations_30k.csv"]
        annotation = next((p for p in candidates if p.is_file()), candidates[0])
    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation}. Pass --annotation_file.")

    image_roots = list(dict.fromkeys([
        root, root / "flickr30k-images", root / "flickr30k-images" / "flickr30k-images",
        annotation.parent / "flickr30k-images",
        annotation.parent / "flickr30k-images" / "flickr30k-images",
    ]))
    records, seen = [], {}
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        caption_column = "raw" if "raw" in columns else "caption"
        required = {"filename", "split", caption_column}
        if not required <= columns:
            raise ValueError(f"CSV needs filename, split and raw (JSON captions); got {sorted(columns)}")
        for line, row in enumerate(reader, 2):
            filename = row["filename"].strip()
            if not filename or Path(filename).name != filename:
                raise ValueError(f"CSV line {line}: expected a plain image filename")
            row_split = canonical_split(row["split"])
            if filename in seen:
                raise ValueError(f"Duplicate image {filename!r} in splits {seen[filename]!r}/{row_split!r}")
            seen[filename] = row_split
            if row_split != split:
                continue
            try:
                captions = json.loads(row[caption_column])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"CSV line {line}: captions must be a JSON list") from exc
            if (not isinstance(captions, list) or not captions
                    or any(not isinstance(c, str) or not c.strip() for c in captions)):
                raise ValueError(f"CSV line {line}: expected a nonempty list of nonempty captions")
            image_path = next((p / filename for p in image_roots if (p / filename).is_file()), None)
            if image_path is None:
                if verify_images:
                    raise FileNotFoundError(f"Image {filename!r} not found under {root}; check extracted directory")
                image_path = root / "flickr30k-images" / filename
            records.append(Flickr30kRecord(filename, str(image_path), tuple(captions), split))
    if not records:
        raise ValueError(f"No rows with split={split!r} in {annotation}")
    return records


class Flickr30kPairs:
    """One image per item; rotate its caption each epoch to avoid false negatives.

    Rows have exactly the schema consumed by DistillationDataset/SingleDataset.
    Absolute image paths work with their existing os.path.join image loader.
    """
    def __init__(self, data_args):
        self.records = load_flickr30k_records(
            data_args.image_dir, data_args.dataset_split,
            getattr(data_args, "annotation_file", None),
        )
        fraction = getattr(data_args, "percent_data", 1.0)
        if not 0 < fraction <= 1:
            raise ValueError("percent_data must be in (0, 1]")
        self.records = self.records[:math.ceil(len(self.records) * fraction)]
        self.mode = getattr(data_args, "flickr_caption_mode", "cycle")
        if self.mode not in {"cycle", "first"}:
            raise ValueError("flickr_caption_mode must be cycle or first")
        if getattr(data_args, "caching_dir", None) and self.mode == "cycle":
            raise ValueError("Caption cycling needs online supervision; cached teachers require flickr_caption_mode=first")
        rng = random.Random(getattr(data_args, "flickr_caption_seed", 42))
        self.offsets = [rng.randrange(len(row.captions)) for row in self.records]
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.records[index]
        caption_id = 0 if self.mode == "first" else (self.offsets[index] + self.epoch) % len(row.captions)
        return {
            "qry": "<|image_1|> " + IMAGE_INSTRUCTION,
            "qry_image_path": row.image_path,
            "pos_text": CAPTION_PREFIX + row.captions[caption_id],
            "pos_image_path": "",
            "encoded_dir": f"Flickr30k/{row.split}/{Path(row.filename).stem}_{caption_id}",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check local Flickr30k annotations and image paths")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--annotation_file", default=None)
    args = parser.parse_args()
    for split in ("train", "val", "test"):
        records = load_flickr30k_records(args.image_dir, split, args.annotation_file)
        print(f"{split}: {len(records)} images, {sum(len(r.captions) for r in records)} captions")
