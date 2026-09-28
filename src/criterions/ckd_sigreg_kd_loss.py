import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision, pooling


class CKDSigRegLoss(nn.Module):
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

    def _dist_gather_tensor(self, t: Tensor):
        if self.world_size == 1:
            return t

        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t

        return torch.cat(all_tensors, dim=0)

    def sketched_participation_ratio_erank(self, z_list_first: list[torch.Tensor], 
                                           z_list_last: list[torch.Tensor],
                                           num_slices: int = 256, 
                                           min_valid_tokens: int = 4, eps: float = 1e-8):
 
        B = len(z_list_last)
        if B == 0:
            return 0.0

        device, dtype = z_list_last[0].device, z_list_last[0].dtype
        D = z_list_last[0].shape[-1]

        def _pad_and_mask(z_list):
            lengths = torch.tensor([x.size(0) for x in z_list], device=device)
            z_padded = pad_sequence(z_list, batch_first=True, padding_value=0.0)
            N_max = z_padded.size(1)
            idx = torch.arange(N_max, device=device).unsqueeze(0)
            mask = idx < lengths.unsqueeze(1)
            return z_padded, mask, lengths

        z0_padded, mask0, len0 = _pad_and_mask(z_list_first)
        zL_padded, maskL, lenL = _pad_and_mask(z_list_last)

        z0_padded = z0_padded.detach().float()
        zL_padded = zL_padded.float()

        A = torch.randn(D, num_slices, device=device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(eps)   # [D, M]

        def _participation_ratio(z_padded, mask, lengths):
            mask_f = mask.unsqueeze(-1).float()                                        # [B, N_max, 1]
            n_valid = lengths.clamp(min=2).float()                                     # [B]

            # Center (giống bước 0 cũ) — bắt buộc để proj mang đúng ý nghĩa "centered"
            mean = (z_padded * mask_f).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
            z_c = (z_padded - mean) * mask_f                                           # [B, N_max, D], padding = 0

            # --- Bậc 1: trace(Cov) — CHÍNH XÁC như code cũ ---
            proj = z_c @ A                                                             # [B, N_max, M]
            var_proj = (proj ** 2).sum(dim=1) / (n_valid - 1.0).unsqueeze(-1)          # [B, M] = a^T Cov a
            trace1 = D * var_proj.mean(dim=-1)                                         # [B]  ≈ trace(Cov)

            # --- Bậc 2: trace(Cov^2) — THÊM MỘT MATMUL, KHÔNG LẶP ---
            # Cov @ a_m  =  Z_c^T @ (Z_c @ a_m) / (n-1)   với mỗi m cùng lúc:
            Cov_A = torch.bmm(z_c.transpose(1, 2), proj) / (n_valid - 1.0).view(-1, 1, 1)  # [B, D, M]
            term2 = (Cov_A ** 2).sum(dim=1)                                            # [B, M] = a^T Cov^2 a
            trace2 = D * term2.mean(dim=-1)                                            # [B]  ≈ trace(Cov^2)

            pr = trace1.pow(2) / trace2.clamp_min(eps)                                 # [B]  participation ratio
            valid = lengths >= min_valid_tokens
            return pr, valid

        z0_normed = z0_padded
        zL_normed = zL_padded

        pr0, valid0 = _participation_ratio(z0_normed, mask0, len0)
        prL, validL = _participation_ratio(zL_normed, maskL, lenL)

        valid = valid0 & validL
        if not valid.any():
            return zL_padded.sum() * 0.0

        loss_per_sample = F.relu(pr0 - prL)

        return loss_per_sample[valid].mean().to(dtype)

    def _compute_modality_distill(self, student_hidden_states, image_features, 
                                  text_token_counts, attention_mask):
        """
        Hàm này chỉ còn nhiệm vụ trích xuất text và vision representations 
        của student, cùng với việc tính toán SIGReg loss.
        """

        batch_size = attention_mask.size(0)
        last_layer_idx = len(student_hidden_states) - 1
        layers = [0, int(last_layer_idx / 2), int(4 * last_layer_idx / 5), last_layer_idx]
        
        stu_img_tokens = {l: [] for l in layers}
        stu_text_reps = []
        
        cur_idx_img = 0
        for i in range(batch_size):
            num_vision_token = 0
            if image_features is not None and cur_idx_img < len(image_features):
                num_vision_token = image_features[cur_idx_img].size(0)
                cur_idx_img += 1
            
            text_last_hidden, img_last_hidden = get_hidden_text_vision(
                student_hidden_states[last_layer_idx][i],
                text_token_counts[i].item(),
                num_vision_token,
                attention_mask[i]
            )
            stu_text_reps.append(text_last_hidden.mean(dim=0))
            
            if num_vision_token > 0:
                for l in layers:
                    _, img_hidden = get_hidden_text_vision(
                        student_hidden_states[l][i],
                        text_token_counts[i].item(),
                        num_vision_token,
                        attention_mask[i]
                    )
                    stu_img_tokens[l].append(img_hidden)

        # 1. Gom representations của Text
        stacked_stu_text_reps = torch.stack(stu_text_reps, dim=0)

        # 2. Gom representations của Vision và tính SIGReg
        stu_img_final_reps = None
        sigreg_final = 0.0
        
        if len(stu_img_tokens[last_layer_idx]) > 0:
            stu_img_final_reps = torch.stack([x.mean(dim=0) for x in stu_img_tokens[last_layer_idx]], dim=0) 
            sigreg_erank_loss = 0.0

            k_layers = 0
            for l in layers[1:-1]:
                k_layers += 1
                # eos_query = pooling(student_hidden_states[l], attention_mask, 
                #                     mode='eos', normalize=True).detach()
                # total_sigreg += self.sigreg_dualview(stu_img_tokens[l], eos_query, 
                #                                      tau=0.1, alpha=0.9)
                # total_sigreg += self.sigreg_sinkhorn(stu_img_tokens[l], concept_queries)

                sigreg_erank_loss += self.sketched_participation_ratio_erank(stu_img_tokens[0], 
                                                                             stu_img_tokens[l])

            if self.args.use_sigreg_loss:
                sigreg_final = sigreg_erank_loss  / max(1, k_layers)

        return stacked_stu_text_reps, stu_img_final_reps, sigreg_final


    def ckd_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student_diff = student.unsqueeze(1) - student.unsqueeze(0)
        teacher_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)

        return F.mse_loss(student_diff, teacher_diff)

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer 
        projectors = model_wrapper.projectors

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None

        tea_text_qry_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_text_token'] is not None else None
        tea_text_pos_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_text_token'] is not None else None
        
        if getattr(self, 'world_size', 1) > 1:
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
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        # =====================================================
        # Project teacher to student embedding dimension
        # =====================================================
        projected_teacher_qry_reps = projectors["t2s"](teacher_qry_reps)
        projected_teacher_pos_reps = projectors["t2s"](teacher_pos_reps)

        all_projected_teacher_qry_reps = self._dist_gather_tensor(projected_teacher_qry_reps)
        all_projected_teacher_pos_reps = self._dist_gather_tensor(projected_teacher_pos_reps)

        # =====================================================
        # Comparative Knowledge Distillation
        #
        # z = [query embeddings, positive embeddings]
        #
        # L_CKD = MSE(
        #     z_s[i] - z_s[j],
        #     z_t[i] - z_t[j]
        # )
        # =====================================================
        student_ckd_reps = torch.cat([all_student_qry_reps, all_student_pos_reps], dim=0)
        teacher_ckd_reps = torch.cat([all_projected_teacher_qry_reps, all_projected_teacher_pos_reps], dim=0)

        kd_loss = self.ckd_loss(student_ckd_reps, teacher_ckd_reps)

        # =====================================================
        # SIGReg
        # =====================================================
        
        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
                 - set([student_tokenizer.eos_token_id])),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        # Trích xuất Representations từ QRY
        _, _, qry_sigreg = self._compute_modality_distill(
            student_hidden_states=student_qry_hidden_states, 
            image_features=student_qry_image_features,
            text_token_counts=num_student_text_qry_tokens, 
            attention_mask=student_qry_input['attention_mask']
        )

        # Trích xuất Representations từ POS
        _, _, pos_sigreg = self._compute_modality_distill(
            student_hidden_states=student_pos_hidden_states, 
            image_features=student_pos_image_features,
            text_token_counts=num_student_text_pos_tokens, 
            attention_mask=student_pos_input['attention_mask']
        )

        SIGReg = torch.zeros_like(contrastive_loss)
        num_sigreg_components = 0

        # print("SIGReg QRY:", qry_sigreg)
        # print("SIGReg POS:", pos_sigreg)


        if student_qry_image_features is not None:
            SIGReg += qry_sigreg
            num_sigreg_components += 1
        if student_pos_image_features is not None:
            SIGReg += pos_sigreg
            num_sigreg_components += 1

        # =====================================================
        # Total
        # =====================================================
        if num_sigreg_components > 0:
            sigreg_loss = SIGReg / num_sigreg_components
        else:
            sigreg_loss = torch.tensor(0.0, device=contrastive_loss.device)

        loss = contrastive_loss + self.kd_loss_weight * kd_loss + self.args.sigreg_weight * sigreg_loss

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "sigreg_loss": sigreg_loss,
        }