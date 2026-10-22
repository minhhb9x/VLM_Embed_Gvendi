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

        self.grad_stu_proj = nn.Linear(self.grad_dim, self.grad_dim, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.grad_stu_proj.weight)

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

    def _extract_grads_old(
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


    def _extract_grads(
        self,
        losses: Tensor,
        params: List[nn.Parameter],
        create_graph: bool = False,
    ):
        """
        losses: [B] - per-anchor InfoNCE losses

        Returns:
            G: [B, P]
        """
        B = losses.size(0)

        grad_outputs = torch.eye(B, device=losses.device, dtype=losses.dtype)

        grads = torch.autograd.grad(
            outputs=losses,
            inputs=params,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )

        G = torch.cat([g.reshape(B, -1).float() for g in grads], dim=-1) # [B, P]

        return G

    
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
            num_teacher_layers = self.get_num_layers(teacher_model.encoder)
            # print(f"Number of teacher layers: {num_teacher_layers}")
            self.extracted_layer_ids = [f"layers.{num_teacher_layers - 1 - i}."
                for i in range(self.args.num_layers)]

            for name, param in teacher_model.named_parameters():
                if ("lora" in name.lower() and any(k in name for k in self.extracted_layer_ids)):
                    param.requires_grad_(True)
                    self.teacher_grad_params.append(param)

            tea_grad_dim = sum(p.numel() for p in self.teacher_grad_params)
            self._tea_rademacher_proj = RademacherProjection(param_dim=tea_grad_dim, proj_dim=self.grad_dim)

            num_student_layers = self.get_num_layers(student_model.encoder)
            # print(f"Number of student layers: {num_student_layers}")
            self.extracted_layer_ids = [f"layers.{num_student_layers - 1 - i}."
                for i in range(self.args.num_layers)]

            for name, param in student_model.named_parameters():
                if ("lora" in name.lower() and any(k in name for k in self.extracted_layer_ids)):
                    self.student_grad_params.append(param)
            stu_grad_dim = sum(p.numel() for p in self.student_grad_params)
            self._stu_rademacher_proj = RademacherProjection(param_dim=stu_grad_dim, proj_dim=self.grad_dim)

            device = self.student_grad_params[0].device
            self.grad_stu_proj = self.grad_stu_proj.to(device)

            self.teacher_grads = True

    def cross_model_infonce(self, sq, sp, tq, tp, tau=0.02):
        tq = F.normalize(tq.float(), dim=-1)
        tp = F.normalize(tp.float(), dim=-1)

        sq = F.normalize(sq.float(), dim=-1)
        sp = F.normalize(sp.float(), dim=-1)

        labels = torch.arange(sq.size(0), device=sq.device)

        loss_q = F.cross_entropy(sq @ tp.T / tau, labels)
        loss_p = F.cross_entropy(tq @ sp.T / tau, labels)

        return (loss_q + loss_p) / 2

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
        scores = scores.view(all_student_qry_reps.size(0), -1) / self.distiller.temperature
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        student_losses = F.cross_entropy(scores, target, reduction="none")
        contrastive_loss = student_losses.mean()

            
        teacher_scores = teacher_model.compute_similarity(all_teacher_qry_reps, all_teacher_pos_reps)
        teacher_scores = teacher_scores.view(all_teacher_qry_reps.size(0), -1) / self.distiller.temperature
        teacher_losses = F.cross_entropy(teacher_scores, target, reduction="none")

        grad_loss = torch.zeros_like(contrastive_loss)

        if self.args.use_distill_grad_loss:
            teacher_grads = self._extract_grads(
                teacher_losses,
                self.teacher_grad_params,
                create_graph=False
            ).detach()

            student_grads = self._extract_grads(
                student_losses,
                self.student_grad_params,
                create_graph=True
            )

            # Project the gradients
            projected_teacher_grads = self._tea_rademacher_proj(teacher_grads)
            projected_student_grads = self._stu_rademacher_proj(student_grads)

            projected_teacher_grads = F.normalize(
                projected_teacher_grads.detach().float(),
                p=2, dim=-1,eps=1e-8
            )

            projected_student_grads = F.normalize(
                projected_student_grads.float(), p=2, dim=-1, eps=1e-8
            )

            Gt = F.normalize(projected_teacher_grads.detach(),
                dim=-1, eps=1e-8
            )

            Gs = F.normalize(projected_student_grads,
                dim=-1, eps=1e-8
            )

            # Gradient Gram matrices
            Kt = Gt @ Gt.T
            Ks = Gs @ Gs.T

            B = Ks.size(0)

            if B > 1:
                mask = ~torch.eye(B, device=Ks.device, dtype=torch.bool)
                grad_loss = ((Ks - Kt).square()[mask].mean())
            else:
                grad_loss = Ks.sum() * 0.0


        kd_loss = torch.zeros_like(contrastive_loss)
        if self.args.use_distill_cse_loss:
            projectors = distiller.projectors
            proj_student_qry_reps = projectors['s2t'](all_student_qry_reps)
            proj_student_pos_reps = projectors['s2t'](all_student_pos_reps)

            kd_loss = self.cross_model_infonce(
                proj_student_qry_reps,
                proj_student_pos_reps,
                all_teacher_qry_reps.detach(),
                all_teacher_pos_reps.detach(),
            )

        
        loss = contrastive_loss + self.args.grad_weight * grad_loss + self.args.kd_weight * kd_loss

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'grad_loss': grad_loss
        }
                        