import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimilarityMatrixDistillationLoss(nn.Module):
    def __init__(self, args):
        super(SimilarityMatrixDistillationLoss, self).__init__()
        self.args = args
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
        return torch.cat(all_tensors, dim=0)

    def _contrastive_loss(self, model, model_wrapper, qry_reps, pos_reps):
        scores = model.compute_similarity(qry_reps, pos_reps)
        scores = scores.view(qry_reps.size(0), -1)

        if pos_reps.size(0) % qry_reps.size(0) != 0:
            raise ValueError(
                f"Expected num_pos to be a multiple of num_qry, got "
                f"num_qry={qry_reps.size(0)} and num_pos={pos_reps.size(0)}"
            )

        target_per_qry = pos_reps.size(0) // qry_reps.size(0)
        target = torch.arange(
            0,
            qry_reps.size(0) * target_per_qry,
            target_per_qry,
            device=scores.device,
            dtype=torch.long,
        )

        return F.cross_entropy(scores / model_wrapper.temperature, target)

    def _similarity_matrix(self, reps: Tensor):
        reps = F.normalize(reps, p=2, dim=-1)
        return reps @ reps.T

    def _similarity_matrix_distillation_loss(
        self,
        student_repr: Tensor,
        teacher_repr: Tensor,
    ):
        student_similarity = self._similarity_matrix(student_repr)
        teacher_similarity = self._similarity_matrix(teacher_repr).detach()

        if student_similarity.shape != teacher_similarity.shape:
            raise ValueError(
                f"Similarity matrix shape mismatch: "
                f"student={student_similarity.shape}, "
                f"teacher={teacher_similarity.shape}"
            )

        n = student_similarity.size(0)
        off_diagonal_mask = ~torch.eye(
            n,
            dtype=torch.bool,
            device=student_similarity.device,
        )

        return F.mse_loss(
            student_similarity[off_diagonal_mask],
            teacher_similarity[off_diagonal_mask],
        )

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model

        student_qry_input = input_data["qry"]
        student_pos_input = input_data["pos"]

        teacher_qry_reps = input_data.get("teacher_qry_rep")
        teacher_pos_reps = input_data.get("teacher_pos_rep")
        if teacher_qry_reps is None or teacher_pos_reps is None:
            raise ValueError(
                "SimilarityMatrixDistillationLoss requires cached "
                "teacher_qry_rep and teacher_pos_rep. Please set "
                "data_args.caching_dir."
            )

        student_qry_reps, _, _, _ = student_model.encode_input(student_qry_input)
        student_pos_reps, _, _, _ = student_model.encode_input(student_pos_input)

        teacher_qry_reps = teacher_qry_reps.to(student_qry_reps.device)
        teacher_pos_reps = teacher_pos_reps.to(student_pos_reps.device)

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

        contrastive_loss = self._contrastive_loss(
            student_model,
            model_wrapper,
            all_student_qry_reps,
            all_student_pos_reps,
        )

        student_repr = torch.cat(
            [all_student_qry_reps, all_student_pos_reps],
            dim=0,
        )
        teacher_repr = torch.cat(
            [all_teacher_qry_reps, all_teacher_pos_reps],
            dim=0,
        )

        kd_loss = self._similarity_matrix_distillation_loss(
            student_repr,
            teacher_repr,
        )
        loss = contrastive_loss + self.kd_loss_weight * kd_loss

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }
