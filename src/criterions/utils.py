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