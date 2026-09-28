import argparse
import os

import torch
import torch.nn.functional as F


def compute_effective_rank(
    hidden_state: torch.Tensor,
    eps: float = 1e-10,
    normalize_by_min_dim: bool = False,
) -> torch.Tensor:
    x = hidden_state.float()
    n, d = x.shape

    s = torch.linalg.svdvals(x) / torch.sqrt(torch.tensor(n, device=x.device, dtype=x.dtype))
    eigvals = s.square()
    prob = eigvals.clamp_min(eps) / eigvals.sum().clamp_min(eps)
    entropy = -(prob * torch.log(prob)).sum()

    erank = torch.exp(entropy)
    if normalize_by_min_dim:
        erank = erank / min(n, d)

    return erank


def get_image_token_slice(obj: dict, hidden_state: torch.Tensor) -> slice:
    """Locate the contiguous image-token block in a padding-free sequence."""
    num_image_tokens = int(obj.get("num_image_tokens", 0))
    num_valid_tokens = int(obj.get("num_valid_tokens", hidden_state.size(1)))

    if hidden_state.size(1) != num_valid_tokens:
        raise ValueError(
            f"Saved hidden length ({hidden_state.size(1)}) does not match "
            f"num_valid_tokens ({num_valid_tokens})."
        )

    if bool(obj.get("last_image_token", False)):
        image_end = num_valid_tokens - int(bool(obj.get("has_eos_id", False)))
        image_start = image_end - num_image_tokens
    else:
        image_start = 0
        image_end = num_image_tokens

    if image_start < 0 or image_end > num_valid_tokens or image_end <= image_start:
        raise ValueError(
            f"Invalid image-token range [{image_start}, {image_end}) for "
            f"num_valid_tokens={num_valid_tokens} and num_image_tokens={num_image_tokens}."
        )

    return slice(image_start, image_end)


def extract_text_tokens(obj: dict, hidden_state: torch.Tensor, image_slice: slice) -> torch.Tensor:
    """
    Remove the image-token block and keep the text-token sequence.

    If num_text_tokens excludes a terminal EOS, the EOS is removed too.
    If num_text_tokens includes EOS, all non-image tokens are kept.
    """
    before_image = hidden_state[:, :image_slice.start, :]
    after_image = hidden_state[:, image_slice.stop:, :]
    text_hidden = torch.cat([before_image, after_image], dim=1)

    num_text_tokens = int(obj.get("num_text_tokens", 0))
    if num_text_tokens <= 0:
        return text_hidden

    if text_hidden.size(1) == num_text_tokens:
        return text_hidden

    has_eos_id = bool(obj.get("has_eos_id", False))
    if has_eos_id and text_hidden.size(1) == num_text_tokens + 1:
        return text_hidden[:, :num_text_tokens, :]

    raise ValueError(
        f"Extracted {text_hidden.size(1)} non-image tokens but num_text_tokens={num_text_tokens}. "
        f"has_eos_id={has_eos_id}."
    )


def load_hidden_layers(
    pt_path: str,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    obj = torch.load(pt_path, map_location="cpu")

    num_image_tokens = int(obj.get("num_image_tokens", 0))
    if num_image_tokens <= 0:
        return None, None, None

    hidden_state = obj["hidden_state"].float()
    image_slice = get_image_token_slice(obj, hidden_state)

    image_hidden_layers = hidden_state[:, image_slice, :]
    text_hidden_layers = extract_text_tokens(obj, hidden_state, image_slice)

    if image_hidden_layers.size(1) != num_image_tokens:
        raise ValueError(
            f"Extracted {image_hidden_layers.size(1)} image tokens from {pt_path}, "
            f"expected {num_image_tokens}."
        )

    if text_hidden_layers.size(1) <= 0:
        raise ValueError(f"No text tokens extracted from {pt_path}.")

    if normalize:
        image_hidden_layers = F.normalize(image_hidden_layers, p=2, dim=-1)
        text_hidden_layers = F.normalize(text_hidden_layers, p=2, dim=-1)

    last_token_all_layers = hidden_state[:, -1, :].clone()
    return image_hidden_layers, text_hidden_layers, last_token_all_layers


def compute_per_sample_layer_eranks(
    token_hidden_layers: torch.Tensor,
    device: torch.device,
    normalize_by_min_dim: bool = False,
) -> torch.Tensor:
    """Input [num_layers, num_tokens, hidden_dim] -> output [num_layers]."""
    layer_eranks = []

    for layer_hidden in token_hidden_layers:
        erank = compute_effective_rank(
            layer_hidden.to(device),
            normalize_by_min_dim=normalize_by_min_dim,
        )
        layer_eranks.append(erank.cpu())

    return torch.stack(layer_eranks, dim=0)


def compute_dataset_layer_eranks(
    hidden_samples: list[torch.Tensor],
    device: torch.device,
    normalize_by_min_dim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each layer compute:
      1. all_token_erank: concatenate all tokens across samples.
      2. mean_pooled_erank: mean-pool tokens within each sample, then eRank across samples.
    """
    all_token_layer_eranks = []
    mean_pooled_layer_eranks = []
    num_layers = hidden_samples[0].size(0)

    for layer_idx in range(num_layers):
        all_tokens = torch.cat([hidden[layer_idx] for hidden in hidden_samples], dim=0).to(device)
        all_token_erank = compute_effective_rank(
            all_tokens,
            normalize_by_min_dim=normalize_by_min_dim,
        ).cpu()
        all_token_layer_eranks.append(all_token_erank)

        mean_pooled = torch.stack(
            [hidden[layer_idx].mean(dim=0) for hidden in hidden_samples],
            dim=0,
        ).to(device)
        mean_pooled_erank = compute_effective_rank(
            mean_pooled,
            normalize_by_min_dim=normalize_by_min_dim,
        ).cpu()
        mean_pooled_layer_eranks.append(mean_pooled_erank)

    return (
        torch.stack(all_token_layer_eranks, dim=0),
        torch.stack(mean_pooled_layer_eranks, dim=0),
    )


def summarize_per_sample_eranks(per_sample_layer_eranks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stacked = torch.stack(per_sample_layer_eranks, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return stacked, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", default="infer/FastVLM-0.5B_talas_1.0_eos_cls/ImageNet-1K/query")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=49)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize image/text tokens along hidden dimension before eRank.")
    parser.add_argument(
        "--normalize_by_min_dim",
        action="store_true",
        help="Divide effective rank by min(n, d). If omitted, return raw effective rank.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    image_per_sample_eranks = []
    text_per_sample_eranks = []
    image_hidden_samples = []
    text_hidden_samples = []
    last_token_all_layers_samples = []
    loaded_files = []

    for idx in range(args.start_idx, args.end_idx + 1):
        pt_path = os.path.join(args.pt_dir, f"{idx:08d}.pt")

        if not os.path.exists(pt_path):
            print(f"Skip missing file: {pt_path}")
            continue

        image_hidden_layers, text_hidden_layers, last_token_all_layers = load_hidden_layers(
            pt_path,
            normalize=args.normalize,
        )

        if image_hidden_layers is None:
            print(f"Skip no-image file: {pt_path}")
            continue

        image_per_sample_eranks.append(
            compute_per_sample_layer_eranks(
                image_hidden_layers,
                device,
                normalize_by_min_dim=args.normalize_by_min_dim,
            )
        )
        text_per_sample_eranks.append(
            compute_per_sample_layer_eranks(
                text_hidden_layers,
                device,
                normalize_by_min_dim=args.normalize_by_min_dim,
            )
        )

        image_hidden_samples.append(image_hidden_layers)
        text_hidden_samples.append(text_hidden_layers)
        last_token_all_layers_samples.append(last_token_all_layers)
        loaded_files.append(pt_path)

    if not image_per_sample_eranks:
        raise RuntimeError("No image/text tokens loaded.")

    image_per_sample_eranks, image_per_sample_mean, image_per_sample_std = summarize_per_sample_eranks(image_per_sample_eranks)
    text_per_sample_eranks, text_per_sample_mean, text_per_sample_std = summarize_per_sample_eranks(text_per_sample_eranks)

    image_all_token_eranks, image_mean_pooled_eranks = compute_dataset_layer_eranks(
        image_hidden_samples,
        device,
        normalize_by_min_dim=args.normalize_by_min_dim,
    )
    text_all_token_eranks, text_mean_pooled_eranks = compute_dataset_layer_eranks(
        text_hidden_samples,
        device,
        normalize_by_min_dim=args.normalize_by_min_dim,
    )

    last_token_all_layers_samples = torch.stack(last_token_all_layers_samples, dim=0)
    num_samples, num_hidden_layers, hidden_dim = last_token_all_layers_samples.shape

    last_token_layer_eranks_raw = []
    last_token_layer_eranks_norm = []

    for layer_idx in range(num_hidden_layers):
        layer_last_tokens = last_token_all_layers_samples[:, layer_idx, :]

        raw_erank = compute_effective_rank(
            layer_last_tokens.to(device),
            normalize_by_min_dim=args.normalize_by_min_dim,
        ).cpu()
        last_token_layer_eranks_raw.append(raw_erank)

        layer_last_tokens_norm = F.normalize(layer_last_tokens, p=2, dim=-1)
        norm_erank = compute_effective_rank(
            layer_last_tokens_norm.to(device),
            normalize_by_min_dim=args.normalize_by_min_dim,
        ).cpu()
        last_token_layer_eranks_norm.append(norm_erank)

    last_token_layer_eranks_raw = torch.stack(last_token_layer_eranks_raw, dim=0)
    last_token_layer_eranks_norm = torch.stack(last_token_layer_eranks_norm, dim=0)
    num_layers = image_hidden_samples[0].size(0)

    print(f"Loaded files: {len(loaded_files)}")
    print(f"Effective-rank normalization: {'min(n, d)' if args.normalize_by_min_dim else 'none'}")
    print(f"Per-sample image effective rank shape: {tuple(image_per_sample_eranks.shape)}")
    print(f"Per-sample text effective rank shape: {tuple(text_per_sample_eranks.shape)}")
    print(f"Last-token all-layers batch shape: {tuple(last_token_all_layers_samples.shape)}")
    print(f"Num samples: {num_samples}")
    print(f"Num hidden layers: {num_hidden_layers}")
    print(f"Hidden dim: {hidden_dim}")

    print("\n========================================")
    print("IMAGE TOKEN EFFECTIVE RANK")
    print("========================================")
    print("\nImage effective rank per layer:")

    for layer_idx, mean_erank, std_erank, all_token_erank, mean_pooled_erank in zip(
        range(num_layers),
        image_per_sample_mean,
        image_per_sample_std,
        image_all_token_eranks,
        image_mean_pooled_eranks,
    ):
        print(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_erank={all_token_erank.item():.6f}, "
            f"mean_pooled_erank={mean_pooled_erank.item():.6f}"
        )

    print(f"\nLast layer image per-sample mean effective rank: {image_per_sample_mean[-1].item():.6f}")
    print(f"Last layer image all-token effective rank: {image_all_token_eranks[-1].item():.6f}")
    print(f"Last layer image mean-pooled effective rank: {image_mean_pooled_eranks[-1].item():.6f}")

    print("\n========================================")
    print("TEXT TOKEN EFFECTIVE RANK")
    print("========================================")
    print("\nText effective rank per layer:")

    for layer_idx, mean_erank, std_erank, all_token_erank, mean_pooled_erank in zip(
        range(num_layers),
        text_per_sample_mean,
        text_per_sample_std,
        text_all_token_eranks,
        text_mean_pooled_eranks,
    ):
        print(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_erank={all_token_erank.item():.6f}, "
            f"mean_pooled_erank={mean_pooled_erank.item():.6f}"
        )

    print(f"\nLast layer text per-sample mean effective rank: {text_per_sample_mean[-1].item():.6f}")
    print(f"Last layer text all-token effective rank: {text_all_token_eranks[-1].item():.6f}")
    print(f"Last layer text mean-pooled effective rank: {text_mean_pooled_eranks[-1].item():.6f}")

    print("\n========================================")
    print("LAST TOKEN EFFECTIVE RANK")
    print("========================================")
    print("\nLast-token effective rank across samples per hidden layer:")

    for layer_idx, raw_erank, norm_erank in zip(
        range(num_hidden_layers),
        last_token_layer_eranks_raw,
        last_token_layer_eranks_norm,
    ):
        print(
            f"  layer {layer_idx:02d}: "
            f"raw={raw_erank.item():.6f}, "
            f"normalized={norm_erank.item():.6f}"
        )

    print("\nLast hidden layer - last token:")
    print(f"  raw effective rank:        {last_token_layer_eranks_raw[-1].item():.6f}")
    print(f"  normalized effective rank: {last_token_layer_eranks_norm[-1].item():.6f}")


if __name__ == "__main__":
    main()
