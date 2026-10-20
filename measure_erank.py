"""Measure InfoNCE gradient G-Vendi and EOS eRank for a single MMEB model.

Run from the root of VLM_Embed_Gvendi. Gradient vectors are projected
one anchor at a time, without building a dense [batch, parameters] matrix.
"""
import argparse
import json
import math
import random
import re
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.arguments import DataArguments, ModelArguments
from src.distiller import DistillationCollator, DistillationDataset
from src.model.model import MMEBModel
from src.model.processor import load_processor


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def select_lora(model, num_layers):
    cfg = getattr(model.config, "text_config", model.config)
    total = cfg["num_hidden_layers"] if isinstance(cfg, dict) else cfg.num_hidden_layers
    if not 1 <= num_layers <= total:
        raise ValueError(f"num_layers must be in [1, {total}]")
    selected = set(range(total - num_layers, total))

    model.requires_grad_(False)
    params = []
    for name, param in model.named_parameters():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if "lora" in name.lower() and match and int(match.group(1)) in selected:
            param.requires_grad_(True)
            params.append(param)
    if not params:
        raise RuntimeError(f"No LoRA/DoRA parameters in layers {sorted(selected)}")
    print(f"LoRA layers: {sorted(selected)} | parameters: {sum(p.numel() for p in params):,}")
    return params


class RademacherProjection:
    """Fixed projection in canonical flattened-parameter blocks; no full gradient vector."""

    def __init__(self, proj_dim=512, block_size=4096, seed=42):
        self.proj_dim = proj_dim
        self.block_size = block_size
        self.seed = seed

    @torch.no_grad()
    def __call__(self, gradients):
        if not gradients:
            raise ValueError("Empty gradients")
        device = gradients[0].device
        out = torch.zeros(self.proj_dim, device=device, dtype=torch.float32)
        rng = torch.Generator(device=device)
        pending, filled, block_start = [], 0, 0

        def project_block(block, start):
            rng.manual_seed(self.seed + start)
            signs = torch.randint(
                0, 2, (block.numel(), self.proj_dim),
                device=device, generator=rng, dtype=torch.int8,
            ).float().mul_(2).sub_(1)
            out.add_(block @ signs)

        for grad in gradients:
            flat = grad.detach().reshape(-1).float()
            offset = 0
            while offset < flat.numel():
                take = min(self.block_size - filled, flat.numel() - offset)
                pending.append(flat[offset:offset + take])
                filled += take
                offset += take
                if filled == self.block_size:
                    project_block(torch.cat(pending), block_start)
                    block_start += filled
                    pending, filled = [], 0

        if filled:
            project_block(torch.cat(pending), block_start)
        return out / math.sqrt(self.proj_dim)


def spectral_erank(X, center=False):
    X = X.double()
    if center:
        X = X - X.mean(dim=0, keepdim=True)
    eigvals = torch.linalg.eigvalsh(X @ X.T).clamp_min(0)
    probs = eigvals / eigvals.sum().clamp_min(1e-20)
    positive = probs[probs > 0]
    return (-positive * positive.log()).sum().exp().item()


def projected_batch_grads(losses, params, projector):
    """Per-anchor InfoNCE gradients, projected immediately to [B, proj_dim]."""
    rows, norms = [], []
    for i in range(losses.numel()):
        grads = torch.autograd.grad(
            losses[i], params,
            retain_graph=(i < losses.numel() - 1),
            create_graph=False,
            allow_unused=False,
        )
        norms.append(torch.stack([g.detach().float().norm() for g in grads]).norm().item())
        row = projector(grads)
        rows.append(F.normalize(row, dim=0, eps=1e-8).cpu())
        del grads
    return torch.stack(rows), norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--model_backbone", required=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--proj_block_size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="TIGER-Lab/MMEB-train")
    parser.add_argument("--subsets", nargs="+", default=[
        "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"])
    parser.add_argument("--image_dir", default="vlm2vec_train/MMEB-train")
    parser.add_argument("--output", default="erank_results/model.json")
    args = parser.parse_args()
    if args.max_batches < 1 or args.batch_size < 2 or args.proj_block_size < 1:
        parser.error("max_batches >= 1, batch_size >= 2, proj_block_size >= 1 required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_args = ModelArguments(
        model_name=args.model_name, checkpoint_path=args.checkpoint_path,
        model_backbone=args.model_backbone, lora=True, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, pooling="eos", normalize=True,
        teacher_backbone=args.model_backbone,
    )
    data_args = DataArguments(
        dataset_name=args.dataset_name, subset_name=args.subsets,
        dataset_split="original", image_dir=args.image_dir, image_resolution="low",
    )

    dataset = DistillationDataset(data_args, model_args)
    count = min(args.max_batches, len(dataset) // args.batch_size) * args.batch_size
    if not count:
        raise ValueError("Not enough samples for a complete batch")
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    processor = load_processor(model_args, data_args)
    collator = DistillationCollator(
        processor, processor, model_args, data_args, training_args=None,
    )
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    model = MMEBModel.load(model_args, is_trainable=False)
    model.to(device).eval()
    params = select_lora(model, args.num_layers)
    projector = RademacherProjection(args.proj_dim, args.proj_block_size, args.seed)

    # Sanity-check FastVLM text embedding placement, useful for diagnosing CPU/GPU mismatches.
    embedding_modules = [
        (name, mod.weight.device)
        for name, mod in model.named_modules()
        if name.endswith("embed_tokens") and isinstance(mod, nn.Embedding)
    ]
    if embedding_modules:
        print("Token embedding device:", embedding_modules[0][1])

    projected, embeddings, raw_norms = [], [], []
    for step, batch in enumerate(tqdm(loader, desc="G-Vendi")):
        qry = to_device(batch["student_inputs"]["qry"], device)
        pos = to_device(batch["student_inputs"]["pos"], device)
        if step == 0:
            print("Input IDs device:", qry["input_ids"].device)

        q = model.encode_input(qry)[0]
        p = model.encode_input(pos)[0]
        if q.size(0) != p.size(0):
            raise ValueError("Expected one positive per query")
        scores = model.compute_similarity(q, p).float() / args.temperature
        targets = torch.arange(q.size(0), device=scores.device)
        per_anchor_losses = F.cross_entropy(scores, targets, reduction="none")

        rows, norms = projected_batch_grads(per_anchor_losses, params, projector)
        projected.append(rows)
        raw_norms.extend(norms)
        embeddings.append(q.detach().float().cpu())

    G = torch.cat(projected)
    E = torch.cat(embeddings)
    result = {
        "model": args.checkpoint_path or args.model_name,
        "backbone": args.model_backbone,
        "batches": len(loader),
        "batch_size": args.batch_size,
        "samples": G.size(0),
        "num_layers": args.num_layers,
        "proj_dim": args.proj_dim,
        "gradient_gvendi": spectral_erank(G),
        "eos_erank": spectral_erank(E, center=True),
        "mean_raw_gradient_norm": sum(raw_norms) / len(raw_norms),
        "seed": args.seed,
        "subsets": args.subsets,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nBatches={result['batches']} | samples={result['samples']}")
    print(f"Gradient G-Vendi: {result['gradient_gvendi']:.4f}")
    print(f"EOS embedding eRank: {result['eos_erank']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
