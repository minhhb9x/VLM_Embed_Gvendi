import json
from src.single_wrapper import SingleWrapper, SingleCollator, SingleDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time 
import os
import sys
from tqdm import tqdm 
import math
import wandb 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
# Todo

def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    return optimizer

def prepare_dataset(data_args, model_args):
    dataset = SingleDataset(data_args, model_args)
    return dataset

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

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

def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(self, model_wrapper, train_data, optimizer, lr_scheduler, criterion, 
                 model_args, training_args, data_args):
        print_rank("Initializing Trainer...")
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model_wrapper = model_wrapper.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        
        self.model_wrapper = DDP(self.model_wrapper, device_ids=[self.gpu_id])

        # <--- [THÊM] Logic kiểm tra report_to="wandb"
        self.use_wandb = False
        if is_main_process():
            # Kiểm tra xem report_to có tồn tại và chứa wandb không
            report_to = getattr(training_args, "report_to", [])
            if report_to is None: report_to = []
            if isinstance(report_to, str):
                report_to = [report_to]
            
            if "wandb" in report_to:
                self.use_wandb = True
    
    def _debug_batch_devices(self, obj, prefix=""):
        if obj is None:
            print(f"{prefix}Value: None")
            return
        
        try:
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor device: {obj.device}, shape: {obj.shape}")
            elif isinstance(obj, dict):
                if len(obj) == 0:
                    print(f"{prefix}Empty dict")
                for k, v in obj.items():
                    self._debug_batch_devices(v, prefix=f"{prefix}{k}.")
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    print(f"{prefix}Empty {type(obj).__name__}")
                for i, v in enumerate(obj):
                    self._debug_batch_devices(v, prefix=f"{prefix}[{i}].")
            else:
                print(f"{prefix}Type: {type(obj).__name__}, Value: {obj}")
        except Exception as e:
            print(f"{prefix}ERROR: {e}")
        
    def run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        losses = []
        
        # Tính tổng số bước (steps) trong epoch để log step
        steps_per_epoch = len(self.train_data.dataset) // self.training_args.per_device_train_batch_size // self.training_args.gradient_accumulation_steps // dist.get_world_size()

        progress_bar = tqdm(total=steps_per_epoch, 
                            desc=f"Epoch {epoch}",
                            disable=not dist.get_rank() == 0)
        for batch_idx, batch in enumerate(self.train_data):
            batch = to_device(batch, self.device)
            loss_dict = self.model_wrapper(self.criterion, batch)
            loss = loss_dict['loss'] / self.training_args.gradient_accumulation_steps

            losses.append(loss.detach().item() * self.training_args.gradient_accumulation_steps)
            
            batch_loss = sum(losses) / len(losses)
            
            loss.backward()
            if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
                if is_main_process():
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{batch_loss:.4f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                    })
                    progress_bar.update(1)

                    # <--- [THÊM] Log metrics vào wandb
                    if self.use_wandb:
                        # Log loss trung bình (cumulative average) hoặc loss tức thời (instant)
                        # Ở đây mình log loss trung bình tích lũy giống như progress bar
                        wandb.log({
                            "train/loss": batch_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + ((batch_idx + 1) / self.training_args.gradient_accumulation_steps) / steps_per_epoch
                        })
                
            torch.cuda.empty_cache()
        progress_bar.close()
        
    def train(self):
        # <--- [THÊM] Khởi tạo wandb run
        if self.use_wandb:
           
            all_config = {}
            if self.model_args: all_config.update(vars(self.model_args))
            if self.data_args: all_config.update(vars(self.data_args))
            if self.training_args: all_config.update(vars(self.training_args))

            wandb.init(
                project="VLM_Embed_Teacher_Pooling",
                config=all_config,
                reinit=True
            )

        for epoch in range(self.training_args.num_train_epochs):
            self.run_epoch(epoch)
            if is_main_process() and self.training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-epoch-{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)

                model = self.model_wrapper.module.model
                model.encoder.save_pretrained(ckpt_dir)

                if hasattr(model.encoder.model.model, 'mm_projector'):
                    projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
                    torch.save(model.encoder.model.model.mm_projector.state_dict(), projector_dir)

                model_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if model_config:
                    model_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)
                try:
                    processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                    if processor:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning: Could not save processor: {e}")
                
                pooling_dir = os.path.join(ckpt_dir, 'modality_gated_pooling.pth')
                pool_v = model.encoder.pool_v
                pool_t = model.encoder.pool_t
                torch.save({
                    'pool_v': pool_v.state_dict(),
                    'pool_t': pool_t.state_dict(),
                }, pooling_dir)
                print_rank(f"Saved modality gated pooling to {pooling_dir}")

                print_rank(f"Saved checkpoint to {ckpt_dir}")

        if is_main_process():
            final_ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-final")
            os.makedirs(final_ckpt_dir, exist_ok=True)
            model = self.model_wrapper.module.model
            model.encoder.save_pretrained(final_ckpt_dir)

            if hasattr(model.encoder.model.model, 'mm_projector'):
                projector_dir =  os.path.join(final_ckpt_dir, "mm_projector.pth")
                torch.save(model.encoder.model.model.mm_projector.state_dict(), projector_dir)
                
            model_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
            tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
            if model_config:
                model_config.save_pretrained(final_ckpt_dir)
            if tokenizer:
                tokenizer.save_pretrained(final_ckpt_dir)
            try:
                processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if processor:
                    processor.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning: Could not save processor: {e}")
            print_rank(f"Saved final model to {final_ckpt_dir}")
            
            pooling_dir = os.path.join(final_ckpt_dir, 'modality_gated_pooling.pth')
            pool_v = model.encoder.pool_v
            pool_t = model.encoder.pool_t
            torch.save({
                'pool_v': pool_v.state_dict(),
                'pool_t': pool_t.state_dict(),
            }, pooling_dir)
            print_rank(f"Saved modality gated pooling to {pooling_dir}")

            if self.use_wandb:
                wandb.finish()
                
def main():
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            local_rank = int(arg.split("=")[-1])
            sys.argv.remove(arg)
            sys.argv.append(f"--local_rank")
            sys.argv.append(f"{local_rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    
    model_wrapper = SingleWrapper(model_args, training_args)
    train_dataset = prepare_dataset(data_args, model_args)
    dist_sampler = DistributedSampler(train_dataset, shuffle=True)

    model_wrapper.eval()

    if model_args.frozen_backbone:
        for p in model_wrapper.parameters():
            p.requires_grad = False

    for p in model_wrapper.model.encoder.pool_v.parameters():
        p.requires_grad = True

    for p in model_wrapper.model.encoder.pool_t.parameters():
        p.requires_grad = True

    model_wrapper.model.encoder.pool_v.train()
    model_wrapper.model.encoder.pool_t.train()

    for n, p in model_wrapper.named_parameters():
        if p.requires_grad:  # thường chỉ là LoRA
            p.data = p.data.to(torch.bfloat16)
    
    collator = SingleCollator(
        processor=model_wrapper.get_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
    )
    
    for n, p in model_wrapper.model.named_parameters():
        if p.requires_grad:
            print(f'Trainable param: {n}')

    optimizer = AdamW(
        list(model_wrapper.model.encoder.pool_v.parameters()) +
        list(model_wrapper.model.encoder.pool_t.parameters()),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(f"Len of train dataset: {len(train_dataloader.dataset)}")
    total_steps = (len(train_dataloader.dataset) // (training_args.per_device_train_batch_size * dist.get_world_size()) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs
    # if model_args.projector_config_path is not None:
    #     optimizer = distiller.add_optimizer_param_group(optimizer)

    print("Number of trainable parameters:", sum(p.numel() for p in optimizer.param_groups[0]['params'] if p.requires_grad))

    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        from transformers import get_constant_schedule_with_warmup
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
        )
    criterion = build_criterion(training_args)
    trainer = Trainer(model_wrapper, train_dataloader, optimizer, lr_scheduler, criterion, 
                      model_args, training_args, data_args)
    trainer.train()
    
if __name__ == "__main__":
    ddp_setup()
    main()
    destroy_process_group()

    # Must have param model_args.load_pretrained_lora if train on pretrained LoRA weights