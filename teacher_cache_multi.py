"""Standalone multi-backbone MMEB teacher cache.

All cache statistics exclude the final non-padding token from image/text.
That final token is still used as the attention query for attention pooling.
The PR formula matches teacher_cache_qwen3.py, but its token set excludes
the final token; PR values may therefore differ from legacy cache files.
"""

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.single_wrapper import SingleCollator, SingleDataset


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_main_process():
    return get_rank() == 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return
    rank = get_local_rank()
    if rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={rank} but only {torch.cuda.device_count()} GPUs available")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


def seed_everything(seed, rank=0):
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(value, device) for value in obj)
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


class DistributedSequentialSampler(Sampler):
    """Shard samples without shuffling, with no duplicated padding items."""

    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_dataloader(model_args, data_args, training_args, processor):
    dataset = SingleDataset(data_args, model_args)
    sampler = (DistributedSequentialSampler(dataset, get_rank(), dist.get_world_size())
               if is_dist() else SequentialSampler(dataset))
    collator = SingleCollator(
        processor=processor, model_args=model_args,
        data_args=data_args, training_args=training_args,
    )
    return DataLoader(
        dataset, batch_size=training_args.per_device_train_batch_size,
        sampler=sampler, collate_fn=collator, drop_last=False, pin_memory=False,
    )


def exact_participation_ratio(tokens, eps=1e-12):
    """Normalized centered token PR, identical to original Qwen3 cache."""
    n, d = tokens.shape
    if n < 2:
        return tokens.new_tensor(float("nan"), dtype=torch.float32)
    x = tokens.to(torch.float64)
    x = x - x.mean(0, keepdim=True)
    gram = x @ x.T
    tr2 = gram.square().sum()
    if tr2 <= eps:
        return tokens.new_tensor(float("nan"), dtype=torch.float32)
    return (gram.diagonal().sum().square() / tr2 / min(n - 1, d)).float()


def normalize_torchrun_args():
    for arg in list(sys.argv):
        if arg.startswith("--local_rank="):
            sys.argv.remove(arg)
            sys.argv.extend(("--local_rank", arg.split("=", 1)[1]))


@dataclass
class CacheArguments:
    max_batches: int = field(default=0, metadata={"help": "0 = all batches"})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite existing qry.pt / pos.pt"})
    require_attention: bool = field(default=False, metadata={"help": "Fail when last-layer attention is unavailable"})


def image_token_id(processor, backbone):
    """Resolve the token occupying the LLM's image positions."""
    if backbone in ("llava_qwen2", "llava_qwen2_old"):
        return -200

    names = {
        "qwen2_vl": "<|image_pad|>", "qwen2_5_vl": "<|image_pad|>",
        "qwen3_vl": "<|image_pad|>",
        "qwen2_vl_tokenselection": "<|image_pad|>",
        "qwen2_5_vl_tokenselection": "<|image_pad|>",
        "llava_onevision": "<image>", "llava_onevision_old": "<image>",
        "llava_next": "<image>",
    }
    if backbone not in names:
        raise ValueError(
            f"Unsupported backbone {backbone!r}: encoder must return "
            "(rep, image_features, attention, hidden_states)."
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    value = getattr(processor, "image_token_id", None)
    if value is None:
        value = getattr(tokenizer, "image_token_id", None)
    if value is None:
        name = getattr(processor, "image_token", None) or names[backbone]
        value = tokenizer.convert_tokens_to_ids(name)
    if not isinstance(value, int) or value < 0 or value == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Cannot resolve image token ID for {backbone}: {value}")
    return value


def position_masks(inputs, hidden_states, image_features, processor, backbone, padding_side=None):
    """Return image/text position indices and last valid index per sample.

    FastVLM inserts its -200 image embeddings inside forward; all other supported
    backbones must have image placeholders expanded in input_ids already.
    """
    ids = inputs["input_ids"]
    valid = inputs.get("attention_mask")
    valid = valid.bool() if valid is not None else torch.ones_like(ids, dtype=torch.bool)
    B, S, _ = hidden_states[-1].shape
    if ids.shape[0] != B:
        raise ValueError("input_ids and hidden_states batch size mismatch")
    image_id = image_token_id(processor, backbone)
    fast = backbone in ("llava_qwen2", "llava_qwen2_old")

    if fast:
        if image_features is None:
            feats = []
        elif isinstance(image_features, torch.Tensor):
            feats = list(image_features) if image_features.ndim == 3 else [image_features]
        else:
            feats = list(image_features)
        if padding_side is None:
            padding_side = getattr(getattr(processor, "tokenizer", processor), "padding_side", "right")
        if padding_side not in ("left", "right"):
            raise ValueError(f"Invalid FastVLM padding side: {padding_side}")
    elif S != ids.shape[1]:
        raise ValueError(
            f"{backbone}: hidden sequence length={S}, input_ids length={ids.shape[1]}. "
            "Image tokens may be expanded inside forward; do not use input_ids masks blindly."
        )

    results, feat_i = [], 0
    for i in range(B):
        token_ids = ids[i, valid[i]]
        if token_ids.numel() == 0:
            raise ValueError(f"Sample {i} has no non-padding tokens")
        labels = torch.where(token_ids == image_id, 2, 1).long()
        if fast:
            loc = torch.where(token_ids == -200)[0]
            if loc.numel() > 1:
                raise ValueError("FastVLM cache supports one image per sample")
            if loc.numel():
                if feat_i >= len(feats) or feats[feat_i] is None:
                    raise ValueError(f"Sample {i}: -200 found without image_features")
                n_image = int(feats[feat_i].shape[0])
                feat_i += 1
                p = int(loc[0].item())
                labels = torch.cat((labels[:p], labels.new_full((n_image,), 2), labels[p + 1:]))
            labels = labels[:S]  # Match FastVLM's truncation to max sequence length
            aligned = labels.new_zeros(S)
            shift = S - labels.numel() if padding_side == "left" else 0
            aligned[shift:shift + labels.numel()] = labels
        else:
            aligned = labels.new_zeros(S)
            aligned[valid[i]] = labels

        used = (aligned != 0).nonzero(as_tuple=True)[0]
        if used.numel() == 0:
            raise ValueError(f"Sample {i} has no tokens after expansion/truncation")
        last = int(used[-1])
        aligned[last] = 0  # Exclude representation token from BOTH modalities.
        results.append((
            (aligned == 2).nonzero(as_tuple=True)[0],
            (aligned == 1).nonzero(as_tuple=True)[0],
            last,
        ))

    if fast and feat_i != len(feats):
        raise ValueError(f"Used {feat_i}/{len(feats)} FastVLM image feature entries")
    return results


def last_attention(attentions, sample_i, last_i, seq_len):
    """Head-averaged attention from the final valid token to every position."""
    if not isinstance(attentions, (tuple, list)) or not attentions:
        return None
    attn = attentions[-1]
    if attn is None or attn.shape[-2:] != (seq_len, seq_len):
        return None
    if attn.ndim == 4:  # [B, heads, S, S]
        return attn[sample_i, :, last_i, :].float().mean(0)
    if attn.ndim == 3:  # [B, S, S]
        return attn[sample_i, last_i, :].float()
    return None


def attention_pool(hidden_tokens, attention_row, indices):
    if hidden_tokens.numel() == 0 or attention_row is None:
        return None
    weights = attention_row[indices].float()
    denom = weights.sum()
    if not torch.isfinite(denom) or denom <= 0:
        return None
    return ((hidden_tokens.float() * weights[:, None]).sum(0) / denom).cpu()


def build_cache_object(rep, hidden_states, attentions, index, positions, require_attention=False):
    image_idx, text_idx, last = positions
    last_hidden = hidden_states[-1][index]
    image_hidden = last_hidden[image_idx]
    text_hidden = last_hidden[text_idx]
    row = last_attention(attentions, index, last, last_hidden.size(0))
    if require_attention and row is None:
        raise RuntimeError(
            "The model did not return last-layer attention. "
            "Use an attention implementation that supports output_attentions=True."
        )

    pr = None
    if image_idx.numel() >= 2:
        pr = torch.stack([
            exact_participation_ratio(layer[index, image_idx])
            for layer in hidden_states
        ]).cpu()

    return {
        "rep": rep[index].detach().float().cpu(),
        "mean_last_img_token": image_hidden.float().mean(0).cpu() if image_idx.numel() else None,
        "mean_last_text_token": text_hidden.float().mean(0).cpu() if text_idx.numel() else None,
        "pr_img_layers": pr,
        # Keep requested spelling: 'polling' is intentionally spelled 'polling',
        # not 'pooling', for compatibility with new cache consumers.
        "attn_polling_last_img_token": attention_pool(image_hidden, row, image_idx),
        "attn_polling_last_text_token": attention_pool(text_hidden, row, text_idx),
    }


@torch.inference_mode()
def run_cache(model, loader, processor, model_args, training_args, cache_args, device):
    output_root = Path(training_args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    backbone = model_args.model_backbone
    padding_side = getattr(getattr(model.encoder, "config", None), "tokenizer_padding_side", None)
    if padding_side not in ("left", "right"):
        padding_side = getattr(getattr(processor, "tokenizer", processor), "padding_side", "right")
    warned_missing_attention = False

    for batch_i, batch in enumerate(tqdm(loader, desc="Teacher cache", disable=not is_main_process())):
        if cache_args.max_batches and batch_i >= cache_args.max_batches:
            break
        paths = [output_root / str(x) for x in batch["encoded_dir"]]
        if not cache_args.overwrite_cache and all(
            (p / "qry.pt").exists() and (p / "pos.pt").exists() for p in paths
        ):
            continue
        batch = to_device(batch, device)

        for side in ("qry", "pos"):
            rep, image_features, attention, hs = model.encode_input(batch[side])
            if not isinstance(hs, (tuple, list)) or not hs:
                raise ValueError(f"{backbone}: expected per-layer hidden states from encode_input")
            positions = position_masks(
                batch[side], hs, image_features, processor, backbone,
                padding_side=padding_side,
            )
            if not warned_missing_attention and (not isinstance(attention, (tuple, list)) or not attention or attention[-1] is None):
                if cache_args.require_attention:
                    raise RuntimeError(f"{backbone} did not return attention matrices")
                print(f"[rank {get_rank()}] NOTE: attention unavailable; attn_polling_* will be None")
                warned_missing_attention = True
            for j, p in enumerate(paths):
                dest = p / f"{side}.pt"
                if dest.exists() and not cache_args.overwrite_cache:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                cache = build_cache_object(rep, hs, attention, j, positions[j], cache_args.require_attention)
                torch.save(cache, dest)
            del rep, image_features, attention, hs


def main():
    normalize_torchrun_args()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CacheArguments))
    model_args, data_args, training_args, cache_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed, get_rank())
    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    model = MMEBModel.load(model_args, is_trainable=False).to(device).eval()
    processor = load_processor(model_args, None)
    loader = build_dataloader(model_args, data_args, training_args, processor)
    if is_main_process():
        print(f"Cache: {model_args.model_name} ({model_args.model_backbone}), "
              f"batches={cache_args.max_batches or 'all'}, output={training_args.output_dir}")
    run_cache(model, loader, processor, model_args, training_args, cache_args, device)
    if is_dist():
        dist.barrier()


if __name__ == "__main__":
    ddp_setup()
    try:
        main()
    finally:
        cleanup_dist()
