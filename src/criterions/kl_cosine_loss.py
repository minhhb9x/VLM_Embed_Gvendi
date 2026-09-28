import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


class KLCosineLoss(nn.Module):
    def __init__(self, args):
        super(KLCosineLoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
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
    

    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher

        if getattr(self, "student_processor", None) is None:
            self.student_processor = distiller.get_student_processor()
        if getattr(self, "teacher_processor", None) is None:
            self.teacher_processor = distiller.get_teacher_processor()

        # student_processor = self.student_processor
        # teacher_processor = self.teacher_processor

        # student_tokenizer = student_processor.tokenizer
        # teacher_tokenizer = teacher_processor.tokenizer
        
        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, _, _, _ = teacher_qry_output
            teacher_pos_reps, _, _, _ = teacher_pos_output
        
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, _, _, _ = student_qry_output
        student_pos_reps, _, _, _ = student_pos_output
        
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
        scores = scores.view(all_student_qry_reps.size(0), -1) # [B, B]
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

        tea_scores = teacher_model.compute_similarity(all_teacher_qry_reps, all_teacher_pos_reps)
        tea_scores = tea_scores.view(all_teacher_qry_reps.size(0), -1) # [B, B]

        # -------- KD loss: KLDiv --------
        T = self.distiller.temperature

        stu_log_prob = F.log_softmax(scores.float() / T, dim=-1)
        tea_prob = F.softmax(tea_scores.float() / T, dim=-1)

        kd_loss = F.kl_div(stu_log_prob, tea_prob, 
                           reduction="batchmean").to(dtype=all_student_qry_reps.dtype) * (T * T)

        loss = contrastive_loss + self.kd_loss_weight * kd_loss

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': kd_loss,
        }