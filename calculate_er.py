import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
import time

import dataclasses

from sklearn.pipeline import islice

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.single_wrapper import SingleWrapper, SingleCollator, SingleDataset

from transformers import HfArgumentParser, AutoConfig


from src.model.model import MMEBModel
from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor, COLPALI
from torch.nn.utils.rnn import pad_sequence
import shutil 
import random

def delete_pycache(root='.'):
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname == '__pycache__':
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting: {full_path}")
                try:
                    shutil.rmtree(full_path)
                except:
                    print(">>>>>", "Module not exists", full_path, flush=True)
                    pass
delete_pycache()

def seed_everything(seed: int, rank: int = 0):
    seed = seed + rank  # quan trọng trong DDP

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Nếu bạn muốn deterministic (chậm hơn, đôi khi lỗi với một số ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Bắt buộc với một số ops CUDA mới (matmul, conv...)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

def prepare_dataset(data_args, model_args):
    dataset = SingleDataset(data_args, model_args)
    return dataset

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    else:
        if hasattr(obj, 'to') and callable(obj.to):
            return obj.to(device)
        return obj

@contextmanager
def time_block(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[Timer] {name}: {elapsed:.4f}s")

def compute_effective_rank(
        hidden_state: torch.Tensor, # [N, D]
        eps: float = 1e-10,
    ) -> torch.Tensor:
    X = hidden_state.float() 
    N = X.size(0)
    s = torch.linalg.svdvals(X) / torch.sqrt(torch.tensor(N))
    eigvals = s * s
    prob = eigvals.clamp(min=eps) / eigvals.sum()
    entropy = -(prob * torch.log(prob)).sum()
    effective_rank = torch.exp(entropy) / N
    return effective_rank.to(dtype=hidden_state.dtype)

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

def get_eranks(model, tokenizer, input):
    attention_mask = input['attention_mask'] # [b, seq_len]
    batch_size = attention_mask.size(0)
    output = model.encode_input(input)
    reps, image_features, attentions, hidden_states = output
    special_ids = torch.tensor(
        list(
            set(
                list(tokenizer.added_tokens_encoder.values()) +
                tokenizer.all_special_ids
            )
        ),
        device=input['input_ids'].device,
        dtype=torch.long
    )
    text_tokens = count_clean_text_tokens(input, special_ids)
    image_feature_ers = []
    hidden_state_ers = []

    for i in range(batch_size):
        num_vision_token = 0
        if image_features:
            image_feature_ers.append(compute_effective_rank(image_features[i]).item())
            num_vision_token = image_features[i].size(0)
        last_unpadded_hidden, _ = get_hidden_text_vision(
            hidden_states[-1][i],
            text_tokens[i].item(),
            num_vision_token,
            attention_mask[i]
        )
        hidden_state_ers.append(compute_effective_rank(last_unpadded_hidden).item())
    return image_feature_ers, hidden_state_ers

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed) 

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    train_dataset = prepare_dataset(data_args, model_args)

    is_main_process = training_args.local_rank in [-1, 0]

    os.makedirs(data_args.encode_output_path, exist_ok=True)
    
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    # model = MMEBModel.build(model_args)
    # if model_args.load_pretrained_lora:
    #     model.encoder.merge_and_unload()
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    train_dataset = prepare_dataset(data_args, model_args)
    collator = SingleCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    # random_sample = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        # sampler=random_sample,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
    )

    qry_hidden_ers = []
    qry_image_feature_ers = []
    pos_hidden_ers = []
    pos_image_feature_ers = []

    for batch in tqdm(islice(train_dataloader, 1000), 
                      desc="Encoding for Effective Rank",
                      disable=not is_main_process, 
                      total=min(len(train_dataloader), 1000)):
        batch = to_device(batch, training_args.device)
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # qry_output = model.encode_input(batch['qry'])
                image_feature_ers, hidden_state_ers = get_eranks(model, processor.tokenizer, batch['qry'])
                qry_image_feature_ers.extend(image_feature_ers)
                qry_hidden_ers.extend(hidden_state_ers)
            # print_rank(f"Batch {batch_idx}: Qry Effective Rank = {effective_rank.item():.4f}")
        
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # pos_output = model.encode_input(batch['pos'])
                image_feature_ers, hidden_state_ers = get_eranks(model, processor.tokenizer, batch['pos'])
                pos_image_feature_ers.extend(image_feature_ers)
                pos_hidden_ers.extend(hidden_state_ers)
            # print_rank(f"Batch {batch_idx}: Pos Effective Rank = {effective_rank.item():.4f}")
    
    qry_hidden_ers_mean = np.mean(qry_hidden_ers)
    qry_image_feature_ers_mean = np.mean(qry_image_feature_ers)
    pos_hidden_ers_mean = np.mean(pos_hidden_ers)
    pos_image_feature_ers_mean = np.mean(pos_image_feature_ers)

    if is_main_process:
        print(f"Qry Hidden Effective Rank: {qry_hidden_ers_mean:.4f}")
        print(f"Qry Image Feature Effective Rank: {qry_image_feature_ers_mean:.4f}")
        print(f"Pos Hidden Effective Rank: {pos_hidden_ers_mean:.4f}")
        print(f"Pos Image Feature Effective Rank: {pos_image_feature_ers_mean:.4f}")
    
        encode_qry_hidden_path = os.path.join(data_args.encode_output_path, f"qry_hidden_ers.json")
        encode_qry_image_feature_path = os.path.join(data_args.encode_output_path, f"qry_image_feature_ers.json")
        encode_pos_hidden_path = os.path.join(data_args.encode_output_path, f"pos_hidden_ers.json")
        encode_pos_image_feature_path = os.path.join(data_args.encode_output_path, f"pos_image_feature_ers.json")

        # Lưu list (để vẽ histogram sau)
        with open(encode_qry_hidden_path, "w", encoding="utf-8") as f:
            json.dump(qry_hidden_ers, f)

        with open(encode_qry_image_feature_path, "w", encoding="utf-8") as f:
            json.dump(qry_image_feature_ers, f)

        with open(encode_pos_hidden_path, "w", encoding="utf-8") as f:
            json.dump(pos_hidden_ers, f)

        with open(encode_pos_image_feature_path, "w", encoding="utf-8") as f:
            json.dump(pos_image_feature_ers, f)

        # Lưu mean riêng (txt)
        with open(os.path.join(data_args.encode_output_path, "er_mean.txt"), "w") as f:
            f.write(f"qry_hidden_er_mean: {qry_hidden_ers_mean}\n")
            f.write(f"qry_image_feature_er_mean: {qry_image_feature_ers_mean}\n")
            f.write(f"pos_hidden_er_mean: {pos_hidden_ers_mean}\n")
            f.write(f"pos_image_feature_er_mean: {pos_image_feature_ers_mean}\n")
    


if __name__ == "__main__":
    main()