import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List
from scipy.optimize import linear_sum_assignment

class RademacherProjection(nn.Module):
    def __init__(self, param_dim: int, proj_dim: int = 512,
                 seed: int = 42, block_size: int = 4096):
        super().__init__()
        self.param_dim  = param_dim
        self.proj_dim   = proj_dim
        self.seed       = seed
        self.block_size = block_size
        self.scale      = proj_dim ** -0.5

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        B, D = g.shape
        out  = torch.zeros(B, self.proj_dim, device=g.device, dtype=g.dtype)
        rng  = torch.Generator(device=g.device)
        for start in range(0, D, self.block_size):
            end = min(start + self.block_size, D)
            rng.manual_seed(self.seed + start)
            Pi  = (torch.randint(0, 2, (end - start, self.proj_dim),
                                 generator=rng, device=g.device,
                                 dtype=g.dtype) * 2 - 1)
            out += g[:, start:end] @ Pi
        return out * self.scale

class GVENDILoss(nn.Module):
    def __init__(self, args):
        super(GVENDILoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
        self.teacher_grads = False
        self.teacher_grad_params = []
        self.student_grad_params = []

        self.grad_dim = 512

        self.grad_stu_proj = nn.Linear(self.grad_dim, self.grad_dim)
        with torch.no_grad():
            self.grad_stu_proj.weight.normal_(mean=0.0, std=1e-3)

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def _tail_params(self, model: nn.Module) -> List[nn.Parameter]:
        named = [(n, p) for n, p in model.named_parameters()
                if p.requires_grad and any(k in n for k in self.extracted_layer_ids)]

        params = []
        for _, p in reversed(named):
            params.append(p)
        return params

    def _extract_grads(
        self,
        loss: Tensor,
        params: List[nn.Parameter],
        create_graph: bool = False,
        retain_graph: bool = False,
    ):
        """
        loss: scalar InfoNCE loss
        params: selected LoRA parameters

        Returns:
            G: [1, P]
            P = total number of selected parameters
        """
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=params,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=False,
        )

        g = torch.cat([
            grad.reshape(-1).float()
            for grad in grads
        ])

        return g.unsqueeze(0)  # [1, P]

    def get_num_layers(self,model):
        config = model.config

        if hasattr(config, "text_config"):
            text_config = config.text_config
            if isinstance(text_config, dict):
                return text_config["num_hidden_layers"]
            return text_config.num_hidden_layers

        return config.num_hidden_layers

    def _init_grad_(self, distiller):
        teacher_model = distiller.teacher
        student_model = distiller.student
        # Extract LoRA parameters for teacher and student models
        if not self.teacher_grads:
            num_teacher_layers = self.get_num_layers(teacher_model.encoder) - 1
            self.extracted_layer_ids = [f"layers.{num_teacher_layers - 1 - i}."
                for i in range(self.args.num_layers)]

            for name, param in teacher_model.named_parameters():
                if ("lora" in name.lower() and any(k in name for k in self.extracted_layer_ids)):
                    param.requires_grad_(True)
                    self.teacher_grad_params.append(param)

            tea_grad_dim = sum(p.numel() for p in self.teacher_grad_params)
            self._tea_rademacher_proj = RademacherProjection(param_dim=tea_grad_dim, proj_dim=self.grad_dim)

            num_student_layers = self.get_num_layers(student_model.encoder) - 1
            self.extracted_layer_ids = [f"layers.{num_student_layers - 1 - i}."
                for i in range(self.args.num_layers)]

            for name, param in student_model.named_parameters():
                if ("lora" in name.lower() and any(k in name for k in self.extracted_layer_ids)):
                    self.student_grad_params.append(param)
            stu_grad_dim = sum(p.numel() for p in self.student_grad_params)
            self._stu_rademacher_proj = RademacherProjection(param_dim=stu_grad_dim, proj_dim=self.grad_dim)


            self.teacher_grads = True

    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']

        
        batch_size = student_qry_input['input_ids'].size(0)

        self._init_grad_(distiller)

        teacher_model.eval()
        teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
        teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
        teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
        teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            all_teacher_qry_reps = teacher_qry_reps
            all_teacher_pos_reps = teacher_pos_reps

        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

            
        teacher_scores = teacher_model.compute_similarity(all_teacher_qry_reps, all_teacher_pos_reps)
        teacher_scores = teacher_scores.view(all_teacher_qry_reps.size(0), -1)
        teacher_contrastive_loss = F.cross_entropy(teacher_scores, target)


        grad_loss = torch.zeros_like(contrastive_loss)

        if self.args.use_distill_grad_loss:
            teacher_grads = self._extract_grads(
                teacher_contrastive_loss,
                self.teacher_grad_params,
                create_graph=False,
                retain_graph=False
            )

            student_grads = self._extract_grads(
                contrastive_loss,
                self.student_grad_params,
                create_graph=True,
                retain_graph=True
            )

            
            projected_teacher_grads = self._tea_rademacher_proj(teacher_grads)
            projected_student_grads = self._stu_rademacher_proj(student_grads)
            self.grad_stu_proj.to(projected_student_grads.device)
            aligned_projected_student_grads = self.grad_stu_proj(projected_student_grads)

            grad_loss = F.mse_loss(aligned_projected_student_grads, projected_teacher_grads)
        
        loss = contrastive_loss + self.args.grad_weight * grad_loss

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'grad_loss': grad_loss
        }
                        