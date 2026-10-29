import numpy as np
import torch

from src.model.model import MMEBModel
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, print_master, QWEN2_5_VL, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, \
    COLPALI, INTERN_VL3, LLAVA_ONEVISION, LLAVA_QWEN2

def count_clean_text_tokens(inputs, special_ids_list):
    """
    Đếm số lượng token hợp lệ:
    1. Giá trị token phải >= 0 (loại bỏ -200, -100...)
    2. Giá trị token không nằm trong special_ids_list (loại bỏ CLS, SEP...)
    """
    input_ids = inputs['input_ids']
    
    if not isinstance(special_ids_list, torch.Tensor):
        # Nếu special_ids_list là list python thường, chuyển thành tensor
        special_ids_tensor = torch.tensor(special_ids_list, device=input_ids.device)
    else:
        # Nếu đã là tensor, đảm bảo cùng device
        special_ids_tensor = special_ids_list.to(input_ids.device)

    valid_index_mask = input_ids >= 0 
    content_mask = ~torch.isin(input_ids, special_ids_tensor)

    final_mask = valid_index_mask & content_mask

    return final_mask.sum(dim=1)

def get_hidden_text_vision(hidden_state, num_text_token, num_vision_token, attention_mask):
    '''
    Get hidden states for text and vision tokens separately
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        num_vision_token: int, number of vision tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        vision_hidden_state = hidden_state[-(num_vision_token+num_text_token): -num_text_token, :]
        text_hidden_state = hidden_state[-num_text_token:, :]
    else:
        vision_hidden_state = hidden_state[:num_vision_token, :]
        text_hidden_state = hidden_state[num_vision_token: num_vision_token + num_text_token, :]
   
    return text_hidden_state, vision_hidden_state

def get_attn_text_vision(attention, num_text_token, num_vision_token, attention_mask):
    '''
    Get attention for text and vision tokens separately
    Args:
        attention: tensor, the output attention from the model
        num_text_token: int, number of text tokens
        num_vision_token: int, number of vision tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        vision_attention = attention[-(num_vision_token+num_text_token): -num_text_token]
        text_attention = attention[-num_text_token:]
    else:
        vision_attention = attention[:num_vision_token]
        text_attention = attention[num_vision_token: num_vision_token + num_text_token]
   
    return text_attention, vision_attention

def pooling(last_hidden_state, attention_mask, mode='eos', normalize=True):
    if mode == 'last' or mode == 'eos':
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        batch_size = last_hidden_state.shape[0]
        if left_padding:
            # Get the vectors at the last position
            reps = last_hidden_state[torch.arange(batch_size), -1, :]
        else:
            # Calculate last 1 position in the original tensor
            max_length = last_hidden_state.size(1)
            invert_mask = (attention_mask == 0).long()
            num_padding_tokens = invert_mask.sum(dim=1)
            eos_indices_positive = max_length - num_padding_tokens - 1
            # Get the vectors at the last 1 position of each attention mask
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), eos_indices_positive]
    else:
        raise NotImplementedError
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

def get_unpadded_hidden(hidden_state, num_text_token, num_vision_token, attention_mask):
    '''
    Get hidden states for unpadded tokens (both text and vision)
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        num_vision_token: int, number of vision tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        unpadded_hidden_state = hidden_state[-(num_vision_token + num_text_token):, :]
    else:
        unpadded_hidden_state = hidden_state[: (num_vision_token + num_text_token), :]
   
    return unpadded_hidden_state

def get_hidden_text(hidden_state, num_text_token, attention_mask):
    '''
    Get hidden states for text tokens
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        text_hidden_state = hidden_state[-num_text_token:, :]
    else:
        text_hidden_state = hidden_state[: num_text_token, :]
   
    return text_hidden_state

def get_grid_size(model: MMEBModel, inputs):
    if model.model_backbone == LLAVA_QWEN2:
        vision_tower = model.encoder.get_vision_tower()
        vision_config = vision_tower.config
        patch_size = vision_config['image_cfg']['patch_size']
        grid_sizes = []
        if 'images' not in inputs:
            return grid_sizes
        for image in inputs['images']:
            if image is None:
                continue
            h, w = image.shape[-2:]
            grid_h = h // patch_size
            grid_w = w // patch_size
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes
    
    elif model.model_backbone in [QWEN2_VL, QWEN2_5_VL]:
        vision_config = model.config.vision_config
        merge_size = vision_config.spatial_merge_size
        grid_sizes = []
        for shape in inputs['image_grid_thw']:
            if shape is None:
                continue
            h, w = shape[0, -2:]
            grid_h = (h // merge_size).item()
            grid_w = (w // merge_size).item()
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes

def build_center_relative_grid(h, w, device=None, dtype=torch.float32):
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def extract_modality_tokens(
    tokenizer, inputs, image_features, attention_matrix,
    output_hidden_states, backbone, *, padding_side=None,
):
    """Return one dict per sample: text/image [L,N,D], last_token [L,D].

    L excludes the initial embedding layer. Tokens retain their original order.
    ``image_features`` is only required for FastVLM samples containing -200.
    ``padding_side`` can override the model-side padding used by FastVLM.
    """
    proc = tokenizer
    tok = getattr(proc, "tokenizer", proc)
    hs = getattr(output_hidden_states, "hidden_states", output_hidden_states)
    if not isinstance(hs, (tuple, list)) or not hs:
        raise ValueError("output_hidden_states must be a nonempty tuple/list")

    attns = attention_matrix if isinstance(attention_matrix, (tuple, list)) else None
    # Hugging Face normally returns: embedding state + one state per layer.
    layers = hs[1:] if len(hs) > 1 and (attns is None or len(hs) == len(attns) + 1) else hs
    B, S, _ = layers[0].shape
    device = layers[0].device
    ids = inputs["input_ids"].to(device)
    valid = inputs.get("attention_mask")
    valid = valid.to(device).bool() if valid is not None else torch.ones_like(ids, dtype=torch.bool)
    if ids.shape != valid.shape or ids.shape[0] != B:
        raise ValueError("Batch mismatch between input_ids, attention_mask, hidden states")

    fast = backbone in {"llava_qwen2", "llava_qwen2_old"}
    if fast:
        image_id = -200
        if image_features is None:
            feats = []
        elif torch.is_tensor(image_features):
            feats = list(image_features) if image_features.ndim == 3 else [image_features]
        else:
            feats = list(image_features)

        if padding_side is None:
            has_left = bool(((~valid[:, 0]) & valid[:, -1]).any())
            has_right = bool((valid[:, 0] & (~valid[:, -1])).any())
            if has_left and has_right:
                raise ValueError("Mixed left/right padding; set padding_side explicitly")
            padding_side = "left" if has_left else "right" if has_right else getattr(tok, "padding_side", "right")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
    else:
        names = {
            "internvl_chat": "<IMG_CONTEXT>",
            "llava_next": "<image>",
            "llava_onevision": "<image>",
            "llava_onevision_old": "<image>",
        }
        if backbone.startswith("qwen") or backbone in {"gme", "lamra"}:
            token_name = "<|image_pad|>"
        else:
            token_name = names.get(backbone)
        if token_name is None:
            raise ValueError(f"Unsupported backbone: {backbone}")
        image_id = getattr(proc, "img_context_token_id", None) if backbone == "internvl_chat" else None
        if image_id is None:
            image_id = getattr(proc, "image_token_id", None)
        if image_id is None:
            image_id = getattr(tok, "image_token_id", None)
        if image_id is None:
            image_id = tok.convert_tokens_to_ids(getattr(proc, "image_token", token_name))
        if not isinstance(image_id, int) or image_id < 0 or image_id == getattr(tok, "unk_token_id", None):
            raise ValueError(f"Could not resolve image token ID for {backbone}")
        if S != ids.shape[1]:
            raise ValueError(f"{backbone}: hidden length {S} != input_ids length {ids.shape[1]}")

    results, cursor = [], 0
    for b in range(B):
        token_ids = ids[b, valid[b]]
        if token_ids.numel() == 0:
            raise ValueError(f"Sample {b} has no valid tokens")
        # 0 = padding/last, 1 = text (including special tokens), 2 = image.
        labels = torch.ones_like(token_ids, dtype=torch.long)

        if fast:
            locations = torch.where(token_ids == image_id)[0]
            if locations.numel() > 1:
                raise ValueError("FastVLM currently expects <=1 image placeholder per sample")
            if locations.numel():
                if cursor >= len(feats) or feats[cursor] is None:
                    raise ValueError("FastVLM image_features missing for a sample with -200")
                count = feats[cursor].shape[0]
                cursor += 1
                pos = int(locations[0])
                labels = torch.cat([labels[:pos], labels.new_full((count,), 2), labels[pos + 1:]])
            # llava_arch.py truncates the expanded sequence from the right.
            labels = labels[:S]
            n = labels.numel()
            expanded = labels.new_zeros(S)
            offset = S - n if padding_side == "left" else 0
            expanded[offset:offset + n] = labels
            last = offset + n - 1
        else:
            labels[token_ids == image_id] = 2
            expanded = labels.new_zeros(S)
            expanded[valid[b]] = labels
            last = int(torch.where(valid[b])[0][-1])

        expanded[last] = 0  # Last valid token is the representation token.
        text_idx = torch.where(expanded == 1)[0]
        image_idx = torch.where(expanded == 2)[0]
        item = {
            "text": torch.stack([h[b, text_idx] for h in layers]),
            "image": torch.stack([h[b, image_idx] for h in layers]),
            "last_token": torch.stack([h[b, last] for h in layers]),
            "text_indices": text_idx,
            "image_indices": image_idx,
            "last_index": last,
            "text_attention": None,
            "image_attention": None,
        }
        if attns is not None and len(attns) == len(layers) and all(
            a is not None and a.ndim == 4 and a.shape[-2:] == (S, S) for a in attns
        ):
            rows = torch.stack([a[b].float().mean(0)[last] for a in attns])
            item["text_attention"] = rows[:, text_idx]
            item["image_attention"] = rows[:, image_idx]
        results.append(item)

    if fast and cursor != len(feats):
        raise ValueError(f"FastVLM used {cursor}/{len(feats)} image feature entries")
    return results
