import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision


class EMSigRegKDLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kd_loss_weight = args.kd_weight

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, tensor: Tensor):
        if self.world_size == 1:
            return tensor

        tensor = tensor.contiguous()
        all_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, tensor)
        all_tensors[self.process_rank] = tensor
        return torch.cat(all_tensors, dim=0)

    def sigreg(self, x: torch.Tensor, num_slices: int = 128):
        device = x.device
        projection_seed = random.randint(0, 2**63 - 1) if self.process_rank == 0 else 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        generator = torch.Generator(device=device)
        generator.manual_seed(projection_seed)

        projections = torch.randn(x.size(1), num_slices, generator=generator, device=device, dtype=x.dtype)
        projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype)
        expected_cf = torch.exp(-0.5 * t.square())
        empirical_cf = torch.exp(1j * (x @ projections).unsqueeze(-1) * t).mean(dim=0)

        if self.world_size > 1:
            dist.all_reduce(empirical_cf, op=dist.ReduceOp.SUM)
            empirical_cf = empirical_cf / self.world_size

        error = (empirical_cf - expected_cf).abs().square().mul(expected_cf)
        sigreg_per_slice = torch.trapezoid(error, t, dim=1) * x.size(0) * self.world_size
        return sigreg_per_slice.mean()

    def _compute_em_loss(
        self,
        student_model,
        teacher_model,
        student_inputs,
        teacher_inputs,
        student_image_features,
        teacher_image_features,
        student_hidden_states,
        teacher_hidden_states,
        num_text_tokens,
        reference,
    ):
        loss_vsd = reference.new_zeros(())
        loss_vlad = reference.new_zeros(())

        if student_image_features is None or teacher_image_features is None:
            return loss_vsd, loss_vlad

        image_index = 0
        batch_size = student_inputs['input_ids'].size(0)

        for i in range(batch_size):
            if image_index >= len(student_image_features) or image_index >= len(teacher_image_features):
                break
            if student_image_features[image_index] is None or teacher_image_features[image_index] is None:
                continue

            num_text_token = num_text_tokens[i].item()
            num_student_vision_token = student_image_features[image_index].size(0)
            num_teacher_vision_token = teacher_image_features[image_index].size(0)

            # right padding -> em_kd.py; left padding -> em_kd_llava_ov.py
            student_text_hidden, student_vision_hidden = get_hidden_text_vision(
                student_hidden_states[-1][i], num_text_token, num_student_vision_token,
                student_inputs['attention_mask'][i]
            )
            teacher_text_hidden, teacher_vision_hidden = get_hidden_text_vision(
                teacher_hidden_states[-1][i], num_text_token, num_teacher_vision_token,
                teacher_inputs['attention_mask'][i]
            )

            student_vision_logits = student_model.encoder.lm_head(student_vision_hidden)
            teacher_vision_logits = teacher_model.encoder.lm_head(teacher_vision_hidden)

            matching_cost = torch.sum(
                torch.abs(teacher_vision_logits.unsqueeze(1) - student_vision_logits.unsqueeze(0)), dim=-1
            ).float().detach().cpu().numpy()
            teacher_indices, student_indices = linear_sum_assignment(matching_cost)
            teacher_indices = torch.tensor(teacher_indices, dtype=torch.long, device=teacher_vision_logits.device)
            student_indices = torch.tensor(student_indices, dtype=torch.long, device=student_vision_logits.device)

            matched_student_logits = student_vision_logits[student_indices]
            matched_teacher_logits = teacher_vision_logits[teacher_indices]
            loss_vsd = loss_vsd + F.mse_loss(matched_student_logits, matched_teacher_logits)

            matched_student_hidden = student_vision_hidden[student_indices]
            matched_teacher_hidden = teacher_vision_hidden[teacher_indices]
            student_affinity = F.cosine_similarity(
                matched_student_hidden.unsqueeze(1), student_text_hidden.unsqueeze(0), dim=-1
            )
            teacher_affinity = F.cosine_similarity(
                matched_teacher_hidden.unsqueeze(1), teacher_text_hidden.unsqueeze(0), dim=-1
            )
            loss_vlad = loss_vlad + F.smooth_l1_loss(student_affinity, teacher_affinity)
            image_index += 1

        return loss_vsd, loss_vlad

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']

        teacher_tokenizer = distiller.get_teacher_processor().tokenizer
        teacher_special_ids = torch.tensor(teacher_tokenizer.all_special_ids, device=teacher_qry_input['input_ids'].device)
        num_text_qry_tokens = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
        num_text_pos_tokens = count_clean_text_tokens(teacher_pos_input, teacher_special_ids)
        batch_size = student_qry_input['input_ids'].size(0)

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, _, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, _, teacher_pos_hidden_states = teacher_pos_output

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, _, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, _, student_pos_hidden_states = student_pos_output

        all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
        all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = F.cross_entropy(scores / distiller.temperature, target)

        qry_vsd, qry_vlad = self._compute_em_loss(
            student_model, teacher_model, student_qry_input, teacher_qry_input,
            student_qry_image_features, teacher_qry_image_features,
            student_qry_hidden_states, teacher_qry_hidden_states,
            num_text_qry_tokens, student_qry_reps
        )
        pos_vsd, pos_vlad = self._compute_em_loss(
            student_model, teacher_model, student_pos_input, teacher_pos_input,
            student_pos_image_features, teacher_pos_image_features,
            student_pos_hidden_states, teacher_pos_hidden_states,
            num_text_pos_tokens, student_pos_reps
        )

        em_kd_loss = (0.25 * (qry_vsd + pos_vsd) + 25 * (qry_vlad + pos_vlad)) / batch_size
        representation_kd_loss = 0.5 * (
            F.mse_loss(student_qry_reps, distiller.projectors['t2s'](teacher_qry_reps))
            + F.mse_loss(student_pos_reps, distiller.projectors['t2s'](teacher_pos_reps))
        )

        sigreg_qry_loss = self.sigreg(student_qry_reps)
        sigreg_pos_loss = self.sigreg(student_pos_reps)
        sigreg_loss = sigreg_qry_loss + sigreg_pos_loss

        loss = (
            0.5 * contrastive_loss + 0.5 * representation_kd_loss + em_kd_loss
            + self.args.sigreg_weight * sigreg_loss
        )

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': em_kd_loss,
            'sigreg_loss': sigreg_loss,
            'sigreg_qry_loss': sigreg_qry_loss,
            'sigreg_pos_loss': sigreg_pos_loss,
        }
