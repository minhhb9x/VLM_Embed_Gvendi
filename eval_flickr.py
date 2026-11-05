"""Evaluate local Flickr30k on the exact nlphuji 1K retrieval test annotations.

Place this file at the repository root, beside eval_mmeb_2.py.
Uses src.flickr30k_data for local CSV/image loading; no dataset download.
Default text inputs are unmodified captions, matching the original FlickrDataset.
--tgt_prefix_mod explicitly adds the current training CAPTION_PREFIX instead.
The annotation check covers filenames, captions and order, not image bytes.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

from collections.abc import Mapping
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Fingerprint of [(filename, captions), ...] from the original test_1k_flickr.csv.
# Serialize with json.dumps(..., ensure_ascii=False, separators=(",", ":")).
# Source: https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval/blob/main/test_1k_flickr.csv
# SHA256 of that source CSV: a427bd870759196b9af09ba9cbd3f0a49c6e6a50c7eafcd9001fdd9f81219105
TEST_ANNOTATION_SHA256 = "874f700f7d48eac7bdd37d14edd4329ad01fea97c7d06c5ede671e8c09b5fd19"


def verify_test_annotations(samples):
    """Fail before model loading if the local test annotations differ."""
    if len(samples) != 1000 or any(len(captions) != 5 for _, _, captions in samples):
        raise ValueError("Expected the original Flickr30k test: 1,000 images, 5 captions each")
    annotations = [(name, captions) for name, _, captions in samples]
    digest = hashlib.sha256(json.dumps(
        annotations, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if digest != TEST_ANNOTATION_SHA256:
        raise ValueError(
            "Local test filenames, caption contents or ordering differ from "
            "nlphuji/flickr_1k_test_image_text_retrieval. "
            "Use the original flickr_annotations_30k.csv without editing captions or row order."
        )
    return digest


def read_local_data(data_args):
    from src.flickr30k_data import load_flickr30k_records

    if data_args.dataset_split != "test":
        raise ValueError("This evaluator requires --dataset_split test")
    records = load_flickr30k_records(
        data_args.image_dir, "test", getattr(data_args, "annotation_file", None)
    )
    samples = [(row.filename, row.image_path, row.captions) for row in records]
    verify_test_annotations(samples)
    return samples


class FlickrDataset(Dataset):
    def __init__(self, samples, modality, image_token, image_resolution=None, caption_prefix=""):
        if modality not in {"image", "text"}:
            raise ValueError("modality must be image or text")
        self.eval_data, self.image_names = [], []
        self.image_resolution = image_resolution
        for name, path, captions in samples:
            if modality == "image":
                text = image_token + " Find an image caption describing the given image."
                # text = image_token 
                self.eval_data.append((text, path))
                self.image_names.append(name)
            else:
                for caption in captions:
                    self.eval_data.append((caption_prefix + caption, None))
                    self.image_names.append(name)

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, index):
        text, path = self.eval_data[index]
        if path is None:
            return text, None
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if self.image_resolution:
            # Preserve the original aspect ratio; resize the longest edge to side.
            side = {"tiny": 336, "low": 448, "mid": 672, "high": 1344}[self.image_resolution]
            width, height = image.size
            scale = side / max(width, height)
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(new_size, resample=Image.Resampling.BICUBIC)
        image.filename = path
        return text, image


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


@torch.no_grad()
def encode(model, loader, device, description):
    embeddings = []
    for batch in tqdm(loader, desc=description):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(qry=to_device(batch, device))
        embeddings.append(output["qry_reps"].detach().float().cpu())
    return torch.cat(embeddings)


def recall_at_k(query, gallery, query_names, gallery_names):
    # Match positives by filename, including all five captions of each image.
    if len(query) != len(query_names) or len(gallery) != len(gallery_names):
        raise ValueError("Embedding and filename counts do not match")
    if not len(query) or not len(gallery):
        raise ValueError("Retrieval query/gallery cannot be empty")
    ranked = torch.argsort(query @ gallery.T, dim=1, descending=True)[:, :10]
    hits = {1: 0, 5: 0, 10: 0}
    for name, indices in zip(query_names, ranked):
        retrieved = [gallery_names[index] for index in indices.tolist()]
        for k in hits:
            hits[k] += name in retrieved[:k]
    metrics = {f"R@{k}": count / len(query) for k, count in hits.items()}
    metrics["P@1"] = metrics["R@1"]
    return metrics


def main():
    from transformers import AutoConfig, HfArgumentParser
    from src.arguments import ModelArguments, DataArguments, TrainingArguments
    from src.model.model import MMEBModel
    from src.data.collator.eval_collator import EvalCollator
    from src.model.processor import load_processor, get_backbone_name, VLM_IMAGE_TOKENS

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.world_size != 1:
        raise ValueError("Run this evaluator with python on one device, not torchrun")
    samples = read_local_data(data_args)
    print("Verified local test annotations: 1,000 images / 5,000 captions; matches nlphuji 1K.")
    if not model_args.model_backbone:
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_args.model_backbone = get_backbone_name(config, model_args.model_type)
    processor = load_processor(model_args, data_args)
    collator = EvalCollator(data_args=data_args, model_args=model_args, processor=processor)
    model = MMEBModel.load(model_args, is_trainable=False)
    model = model.to(training_args.device, dtype=torch.bfloat16).eval()
    output_dir = Path(data_args.encode_output_path or "Flickr30k-eval_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    from src.flickr30k_data import CAPTION_PREFIX
    # prefix = CAPTION_PREFIX if data_args.tgt_prefix_mod else ""
    prefix = "summary:" if data_args.tgt_prefix_mod else ""

    embeddings, names = {}, {}
    for modality in ("image", "text"):
        dataset = FlickrDataset(samples, modality, VLM_IMAGE_TOKENS[model_args.model_backbone],
                                data_args.image_resolution, prefix)
        loader = DataLoader(dataset, batch_size=training_args.per_device_eval_batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=training_args.dataloader_num_workers, collate_fn=collator)
        embeddings[modality] = encode(model, loader, training_args.device, f"Encode {modality}")
        names[modality] = dataset.image_names
        with (output_dir / f"flickr_{data_args.dataset_split}_{modality}.pkl").open("wb") as handle:
            pickle.dump((embeddings[modality].numpy(), names[modality]), handle)

    metrics = {}
    for direction, source, target in (("i2t", "image", "text"), ("t2i", "text", "image")):
        metrics[direction] = recall_at_k(embeddings[source], embeddings[target], names[source], names[target])
        print(direction, metrics[direction])
    result = dict(reference_dataset="nlphuji/flickr_1k_test_image_text_retrieval",
                  annotation_sha256=TEST_ANNOTATION_SHA256,
                  split=data_args.dataset_split, num_images=len(samples),
                  num_captions=len(names["text"]), caption_prefix=prefix,
                  model_name=model_args.model_name, checkpoint_path=model_args.checkpoint_path,
                  image_resolution=data_args.image_resolution, pooling=model_args.pooling,
                  normalize=model_args.normalize, metrics=metrics)
    score_path = output_dir / f"flickr_{data_args.dataset_split}_score.json"
    score_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {score_path}")


if __name__ == "__main__":
    main()
