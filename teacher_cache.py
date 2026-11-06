import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.single_wrapper import SingleCollator, SingleDataset
from src.utils import print_rank


def seed_everything(seed: int, rank: int = 0):
    seed = seed + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = get_local_rank()
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Lower --nproc_per_node or set CUDA_VISIBLE_DEVICES to expose more GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


def to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def release_memory(device):
    gc.collect()
    if torch.cuda.is_available():
        if device is not None and torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() 
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def prepare_dataset(data_args, model_args):
    return SingleDataset(data_args, model_args)


def get_special_ids_for_text_count(tokenizer):
    if tokenizer is None:
        return set()

    eos_ids = tokenizer.eos_token_id
    eos_ids = [] if eos_ids is None else eos_ids if isinstance(eos_ids, list) else [eos_ids]
    return set(getattr(tokenizer, "all_special_ids", [])) - set(eos_ids)


def count_text_tokens(inputs, special_ids_tensor, idx):
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return 0
    valid_mask = input_ids.ge(0) & inputs["attention_mask"].bool()
    if special_ids_tensor.numel() > 0:
        valid_mask = valid_mask & ~torch.isin(input_ids, special_ids_tensor.to(input_ids.device))
    return int(valid_mask[idx].sum().item())


def count_image_tokens(image_features, idx):
    if image_features is None or idx >= len(image_features) or image_features[idx] is None:
        return 0
    return int(image_features[idx].size(0))


def get_clean_token_slice(attention_mask, hidden_len, num_text_tokens, num_image_tokens):
    num_valid_tokens = num_text_tokens + num_image_tokens
    if num_valid_tokens <= 0:
        raise ValueError("No valid token found for hidden/attention inference.")
    if num_valid_tokens > hidden_len:
        raise ValueError(
            f"num_valid_tokens={num_valid_tokens} exceeds hidden_len={hidden_len}. "
            f"num_image_tokens={num_image_tokens}, num_text_tokens={num_text_tokens}."
        )

    left_padding = bool((attention_mask[0].eq(0) & attention_mask[-1].eq(1)).item())
    if left_padding:
        return slice(hidden_len - num_valid_tokens, hidden_len)
    return slice(0, num_valid_tokens)


def split_last_hidden_tokens(last_hidden_state, token_slice, num_image_tokens, num_text_tokens):
    clean_hidden = last_hidden_state[token_slice, :]
    image_hidden = clean_hidden[:num_image_tokens, :]
    text_hidden = clean_hidden[num_image_tokens:num_image_tokens + num_text_tokens, :]
    return image_hidden, text_hidden


def mean_or_none(hidden_tokens):
    if hidden_tokens.numel() == 0:
        return None
    return hidden_tokens.float().mean(dim=0).cpu()


def build_cache_obj(rep, inputs, hidden_states, image_features, special_ids_tensor, idx):
    num_image_tokens = count_image_tokens(image_features, idx)
    num_text_tokens = count_text_tokens(inputs, special_ids_tensor, idx)
    last_hidden_state = hidden_states[-1][idx]
    token_slice = get_clean_token_slice(
        inputs["attention_mask"][idx],
        last_hidden_state.size(0),
        num_text_tokens,
        num_image_tokens,
    )
    image_hidden, text_hidden = split_last_hidden_tokens(
        last_hidden_state,
        token_slice,
        num_image_tokens,
        num_text_tokens,
    )

    return to_device({
        "rep": rep[idx].float(),
        "mean_last_img_token": mean_or_none(image_hidden),
        "mean_last_text_token": mean_or_none(text_hidden),
    }, 'cpu')


class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TeacherCacheRunner:
    def __init__(self, teacher, train_data, model_args, training_args):
        self.device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        self.train_data = train_data
        self.model_args = model_args
        self.training_args = training_args

    @torch.inference_mode()
    def run_once(self):
        progress_bar = tqdm(
            total=len(self.train_data),
            desc="Teacher cache",
            disable=not is_main_process(),
        )

        device = self.device
        processor = load_processor(self.model_args, None)
        special_ids = get_special_ids_for_text_count(processor.tokenizer)
        special_ids_tensor = torch.tensor(sorted(special_ids), device=device, dtype=torch.long)

        for batch in self.train_data:
            encoded_dirs = batch.get("encoded_dir")
            if encoded_dirs is None:
                raise KeyError("Batch is missing encoded_dir. Make sure SingleDataset.__getitem__ and SingleCollator return it.")
            
            skip_batch = True
            for encoded_dir in encoded_dirs:
                output_qry_dir = os.path.join(self.training_args.output_dir, encoded_dir, "qry.pt")
                output_pos_dir = os.path.join(self.training_args.output_dir, encoded_dir, "pos.pt")
                
                if not (os.path.exists(output_qry_dir) and os.path.exists(output_pos_dir)):
                    skip_batch = False
                    break
            
            if skip_batch:
                progress_bar.update(1)
                continue

            batch = to_device(batch, self.device)

            with torch.no_grad():
                qry_output = self.teacher.encode_input(batch["qry"])
                pos_output = self.teacher.encode_input(batch["pos"])

            qry_reps, qry_image_features, _, qry_hidden_states = qry_output
            pos_reps, pos_image_features, _, pos_hidden_states = pos_output

            if len(encoded_dirs) != qry_reps.size(0) or len(encoded_dirs) != pos_reps.size(0):
                raise ValueError(
                    f"encoded_dir count ({len(encoded_dirs)}) does not match encoded batch size "
                    f"(qry={qry_reps.size(0)}, pos={pos_reps.size(0)})."
                )

            for i, encoded_dir in enumerate(encoded_dirs):
                output_qry_dir = os.path.join(self.training_args.output_dir, encoded_dir, "qry.pt")
                output_pos_dir = os.path.join(self.training_args.output_dir, encoded_dir, "pos.pt")

                if os.path.exists(output_qry_dir) and os.path.exists(output_pos_dir):
                    continue

                os.makedirs(os.path.dirname(output_qry_dir), exist_ok=True)
                os.makedirs(os.path.dirname(output_pos_dir), exist_ok=True)

                qry_obj = build_cache_obj(
                    qry_reps,
                    batch["qry"],
                    qry_hidden_states,
                    qry_image_features,
                    special_ids_tensor,
                    i,
                )
                pos_obj = build_cache_obj(
                    pos_reps,
                    batch["pos"],
                    pos_hidden_states,
                    pos_image_features,
                    special_ids_tensor,
                    i,
                )

                torch.save(qry_obj, output_qry_dir)
                torch.save(pos_obj, output_pos_dir)

            progress_bar.update(1)
            del qry_reps
            del pos_reps
            del qry_hidden_states
            del pos_hidden_states
            del batch
            release_memory(self.device)

        progress_bar.close()
        return

def build_dataloader(model_args, data_args, training_args):
    train_dataset = prepare_dataset(data_args, model_args)

    if is_dist():
        sampler = DistributedSequentialSampler(
            train_dataset,
            rank=get_rank(),
            world_size=get_world_size(),
        )
    else:
        sampler = SequentialSampler(train_dataset)

    collator = SingleCollator(
        processor=load_processor(model_args, None),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    return DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
    )


def normalize_torchrun_args():
    for arg in list(sys.argv):
        if arg.startswith("--local_rank="):
            local_rank = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.extend(["--local_rank", local_rank])


def main():
    normalize_torchrun_args()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    seed_everything(training_args.seed, rank=get_rank())

    teacher = MMEBModel.load(model_args, is_trainable=False)
    for n, p in teacher.named_parameters():
        p.requires_grad = False
        
    train_dataloader = build_dataloader(model_args, data_args, training_args)

    print_rank(f"Loaded {len(train_dataloader.dataset)} training samples")
    print_rank(f"Running teacher inference over {len(train_dataloader)} batches")

    runner = TeacherCacheRunner(teacher, train_dataloader, model_args, training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)
    runner.run_once()

    
    if is_dist():
        dist.barrier()


@record
def entrypoint():
    ddp_setup()
    try:
        main()
    finally:
        cleanup_dist()


if __name__ == "__main__":
    entrypoint()
