import json
from src.distiller import Distiller, DistillationCollator, DistillationDataset
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

import random
import numpy as np

def seed_everything(seed: int, rank: int = 0):
    seed = seed + rank  # quan trọng trong DDP

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Nếu bạn muốn deterministic (chậm hơn, đôi khi lỗi với một số ops)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Bắt buộc với một số ops CUDA mới (matmul, conv...)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    dataset = DistillationDataset(data_args, model_args)
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
    def __init__(self, distiller, train_data, optimizer, lr_scheduler, criterion, 
                 model_args, training_args, data_args):
        print_rank("Initializing Trainer...")
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.distiller = distiller.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        
        self.distiller = DDP(self.distiller, device_ids=[self.gpu_id])

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
        losses, contrastive_losses, kd_losses = [], [], []
        kd_rkd_losses, ot_losses, kd_dtw_losses = [], [], []
        kd_mse_losses, kd_penultimate_losses = [], []
        span_losses, cross_modal_losses = [], []
        
        # Tính tổng số bước (steps) trong epoch để log step
        steps_per_epoch = len(self.train_data.dataset) // self.training_args.per_device_train_batch_size // self.training_args.gradient_accumulation_steps // dist.get_world_size()
        
        progress_bar = tqdm(total=steps_per_epoch, 
                            desc=f"Epoch {epoch}",
                            disable=not dist.get_rank() == 0)
        for batch_idx, batch in enumerate(self.train_data):
            batch = to_device(batch, self.device)
            loss_dict = self.distiller(self.criterion, batch)
            loss = loss_dict['loss'] / self.training_args.gradient_accumulation_steps
            kd_loss = loss_dict.get('kd_loss', torch.tensor(0.0))
            span_loss = loss_dict.get('span_loss', torch.tensor(0.0))
            contrastive_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0))
            kd_rkd_loss = loss_dict.get('kd_loss_rkd', torch.tensor(0.0))
            cross_modal_loss = loss_dict.get('cross_modal_loss', torch.tensor(0.0))
            ot_loss = loss_dict.get('ot_loss', torch.tensor(0.0))
            kd_dtw_loss = loss_dict.get('kd_loss_dtw', torch.tensor(0.0))
            kd_mse_loss = loss_dict.get('kd_mse_loss', torch.tensor(0.0))
            kd_penultimate_loss = loss_dict.get('kd_penultimate_loss', torch.tensor(0.0))

            losses.append(loss.detach().item() * self.training_args.gradient_accumulation_steps)
            contrastive_losses.append(contrastive_loss.detach().item())
            kd_losses.append(kd_loss.detach().item())
            span_losses.append(span_loss.detach().item())
            kd_rkd_losses.append(kd_rkd_loss.detach().item())
            cross_modal_losses.append(cross_modal_loss.detach().item())
            ot_losses.append(ot_loss.detach().item())
            kd_dtw_losses.append(kd_dtw_loss.detach().item())
            kd_mse_losses.append(kd_mse_loss.detach().item())
            kd_penultimate_losses.append(kd_penultimate_loss.detach().item())
            
            batch_loss = sum(losses) / len(losses)
            batch_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
            batch_kd_loss = sum(kd_losses) / len(kd_losses)
            batch_span_loss = sum(span_losses) / len(span_losses)
            batch_kd_rkd_loss = sum(kd_rkd_losses) / len(kd_rkd_losses)
            batch_cross_modal_loss = sum(cross_modal_losses) / len(cross_modal_losses)
            batch_ot_loss = sum(ot_losses) / len(ot_losses)
            batch_kd_dtw_loss = sum(kd_dtw_losses) / len(kd_dtw_losses)
            batch_kd_loss_mse = sum(kd_mse_losses) / len(kd_mse_losses)
            batch_kd_penultimate_loss = sum(kd_penultimate_losses) / len(kd_penultimate_losses)
            
            loss.backward()
            if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
                if is_main_process():
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{batch_loss:.4f}",
                        'kd_loss': f"{batch_kd_loss:.4f}",
                        'contrastive_loss': f"{batch_contrastive_loss:.4f}",
                        'kd_rkd_loss': f"{batch_kd_rkd_loss:.4f}",
                        'cross_modal_loss': f"{batch_cross_modal_loss:.4f}",
                        'ot_loss': f"{batch_ot_loss:.4f}",
                        'kd_dtw_loss': f"{batch_kd_dtw_loss:.4f}",
                        'kd_loss_mse': f"{batch_kd_loss_mse:.4f}",
                        'kd_penultimate_loss': f"{batch_kd_penultimate_loss:.4f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                    })
                    progress_bar.update(1)

                    # <--- [THÊM] Log metrics vào wandb
                    if self.use_wandb:
                        # Log loss trung bình (cumulative average) hoặc loss tức thời (instant)
                        # Ở đây mình log loss trung bình tích lũy giống như progress bar
                        wandb.log({
                            "train/loss": batch_loss,
                            "train/kd_loss": batch_kd_loss,
                            "train/contrastive_loss": batch_contrastive_loss,
                            "train/kd_rkd_loss": batch_kd_rkd_loss,
                            "train/ot_loss": batch_ot_loss,
                            "train/kd_dtw_loss": batch_kd_dtw_loss,
                            "train/kd_loss_mse": batch_kd_loss_mse,
                            "train/kd_penultimate_loss": batch_kd_penultimate_loss,
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
                project="VLM_Embed_distill",
                config=all_config,
                reinit=True
            )

        for epoch in range(self.training_args.num_train_epochs):
            self.run_epoch(epoch)
            if is_main_process() and self.training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-epoch-{epoch}")
                projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                student = self.distiller.module.student
                student.encoder.save_pretrained(ckpt_dir)
                if self.model_args.model_backbone in ["llava_onevision", "llava_two_vision"]:
                    torch.save(student.encoder.model.multi_modal_projector.state_dict(), projector_dir)
                else:
                    # if hasattr(student.encoder.model.model, 'mm_projector'):
                    torch.save(student.encoder.model.model.mm_projector.state_dict(), projector_dir)

                student_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if student_config:
                    student_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)
                try:
                    processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                    if processor:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning: Could not save processor: {e}")
                print_rank(f"Saved checkpoint to {ckpt_dir}")

                ## for evaluation
                print(f"Start evaluating student at epoch {epoch}")
            dist.barrier()

        if is_main_process():
            final_ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-final")
            projector_dir =  os.path.join(final_ckpt_dir, "mm_projector.pth")
            os.makedirs(final_ckpt_dir, exist_ok=True)
            student = self.distiller.module.student
            student.encoder.save_pretrained(final_ckpt_dir)
            if self.model_args.model_backbone in ["llava_onevision", "llava_two_vision"]:
                torch.save(student.encoder.model.multi_modal_projector.state_dict(), projector_dir)
            else:
                # if hasattr(student.encoder.model.model, 'mm_projector'):
                torch.save(student.encoder.model.model.mm_projector.state_dict(), projector_dir)
            student_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
            tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
            if student_config:
                student_config.save_pretrained(final_ckpt_dir)
            if tokenizer:
                tokenizer.save_pretrained(final_ckpt_dir)
            try:
                processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if processor:
                    processor.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning: Could not save processor: {e}")
            print_rank(f"Saved final model to {final_ckpt_dir}")
            
            if self.use_wandb:
                wandb.finish()

        dist.barrier()
                
                
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
    
    rank = dist.get_rank()
    seed_everything(training_args.seed, rank=rank) 
    
    distiller = Distiller(model_args, training_args)
    train_dataset = prepare_dataset(data_args, model_args)
    dist_sampler = DistributedSampler(train_dataset, shuffle=True, seed=training_args.seed)
    for n, p in distiller.student.named_parameters():
        if p.requires_grad:  # thường chỉ là LoRA
            p.data = p.data.to(torch.bfloat16)
    
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
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
    num_trainable_vision = 0
    for n, p in distiller.student.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True
            
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True
            
        if "lm_head" in n:
            p.requires_grad = False
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
            num_trainable_vision += p.numel()
    print_rank(f"Number of trainable vision parameters: {num_trainable_vision}")
    
    optimizer = AdamW(
        distiller.student.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(f"Len of train dataset: {len(train_dataloader.dataset)}")
    total_steps = (len(train_dataloader.dataset) // (training_args.per_device_train_batch_size * dist.get_world_size()) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs
    # if model_args.projector_config_path is not None:
    optimizer = distiller.add_optimizer_param_group(optimizer)

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
    trainer = Trainer(distiller, train_dataloader, optimizer, lr_scheduler, criterion, 
                      model_args, training_args, data_args)
    trainer.train()
    
if __name__ == "__main__":
    ddp_setup()
    main()
    destroy_process_group()