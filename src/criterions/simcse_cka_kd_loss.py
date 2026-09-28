import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text, get_hidden_text_vision, pooling
import random
import os


class SimcseCKA(nn.Module):
    def __init__(self, args):
        super(SimcseCKA, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def linear_cka_loss(
        self, student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-8,
    ):
        X = student.float()
        Y = teacher.float()

        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        cross = X.T @ Y
        hsic = cross.square().sum()

        xx = X.T @ X
        yy = Y.T @ Y

        norm_x = torch.sqrt(xx.square().sum() + eps)
        norm_y = torch.sqrt(yy.square().sum() + eps)

        cka = hsic / (norm_x * norm_y + eps)

        return 1.0 - cka

    def distillcse_kd_loss(
            self, S1, S2,
                T1, T2,
                tau=0.05,):
        """
        Distill teacher similarity distribution over in-batch negatives.

        Student and teacher dimensions do not need to match because
        distillation is applied to pairwise similarity matrices.
        """
        S1 = F.normalize(S1.float(), p=2, dim=-1)

        S2 = F.normalize(S2.float(), p=2, dim=-1)

        T1 = F.normalize(T1.float(), p=2, dim=-1)

        T2 = F.normalize(T2.float(), p=2, dim=-1,)

        s_logits = (S1 @ S2.transpose(0, 1)) / tau

        t_logits = (T1 @ T2.transpose(0, 1)) / tau

        # Positive query-passage pairs are on the diagonal.
        # DistillCSE KD here focuses on the negative distribution.
        mask = torch.eye(s_logits.size(0), device=s_logits.device, dtype=torch.bool,)

        s_logits = s_logits.masked_fill(mask, torch.finfo(s_logits.dtype).min,)

        t_logits = t_logits.masked_fill( mask,torch.finfo(t_logits.dtype).min,)

        teacher_probs = F.softmax(t_logits,dim=1,).detach()

        student_log_probs = F.log_softmax(s_logits,dim=1,)

        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean",)
    

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_processor = distiller.get_student_processor()
        student_tokenizer = student_processor.tokenizer

        teacher_processor = distiller.get_teacher_processor()
        teacher_tokenizer = teacher_processor.tokenizer

        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        with torch.no_grad():
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
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
        
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)
        
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
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        kd_simcse = 0.0

        kd_simcse += self.distillcse_kd_loss(all_student_qry_reps, all_student_pos_reps,
                                            all_teacher_qry_reps, all_teacher_pos_reps)

        ##################################


        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
                 - set([student_tokenizer.eos_token_id])),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        teacher_special_ids = torch.tensor(
            list(set(list(teacher_tokenizer.added_tokens_encoder.values()) + teacher_tokenizer.all_special_ids) 
                 - set([teacher_tokenizer.eos_token_id])),
            device=teacher_qry_input['input_ids'].device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        num_teacher_text_qry_tokens = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
        num_teacher_text_pos_tokens = count_clean_text_tokens(teacher_pos_input, teacher_special_ids)

        num_vision_layer_kd = 5

        vision_loss = 0.0

        for layer_idx in range(1, num_vision_layer_kd+1):
            cur_idx_qry_img = 0
            cur_idx_pos_img = 0

            stu_img_qry_reps = []
            stu_img_pos_reps = []

            tea_img_qry_reps = []
            tea_img_pos_reps = []

            # loop qua từng layer
            for i in range(batch_size):
                # --- Xử lý QUERY Image ---
                if student_qry_image_features is not None and teacher_qry_image_features is not None:
                    # Kiểm tra index hợp lệ
                    if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                        stu_feat_qry = student_qry_image_features[cur_idx_qry_img]
                        tea_feat_qry = teacher_qry_image_features[cur_idx_qry_img]

                        num_stu_vision_token = stu_feat_qry.size(0)
                        num_tea_vision_token = tea_feat_qry.size(0)
                    
                    
                        stu_qry_hidden_state_i = student_qry_hidden_states[-layer_idx][i]
                        tea_qry_hidden_state_i = teacher_qry_hidden_states[-layer_idx][i]

                        _, last_stu_img_hidden_state = get_hidden_text_vision(
                            stu_qry_hidden_state_i,
                            num_student_text_qry_tokens[i].item(),
                            num_stu_vision_token,
                            student_qry_input['attention_mask'][i]
                        )
                        _, last_tea_img_hidden_state = get_hidden_text_vision(
                            tea_qry_hidden_state_i,
                            num_teacher_text_qry_tokens[i].item(),
                            num_tea_vision_token,
                            teacher_qry_input['attention_mask'][i]
                        )

                        stu_img_qry_reps.append(last_stu_img_hidden_state.mean(dim=0))
                        tea_img_qry_reps.append(last_tea_img_hidden_state.mean(dim=0))

                        cur_idx_qry_img += 1

                if student_pos_image_features is not None and teacher_pos_image_features is not None:
                    if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                        stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                        tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                        num_stu_vision_token = stu_feat_pos.size(0)
                        num_tea_vision_token = tea_feat_pos.size(0)

                        stu_pos_hidden_state_i = student_pos_hidden_states[-layer_idx][i]
                        tea_pos_hidden_state_i = teacher_pos_hidden_states[-layer_idx][i]

                        _, last_stu_img_hidden_state = get_hidden_text_vision(
                            stu_pos_hidden_state_i,
                            num_student_text_pos_tokens[i].item(),
                            num_stu_vision_token,
                            student_pos_input['attention_mask'][i]
                        )
                        _, last_tea_img_hidden_state = get_hidden_text_vision(
                            tea_pos_hidden_state_i,
                            num_teacher_text_pos_tokens[i].item(),
                            num_tea_vision_token,
                            teacher_pos_input['attention_mask'][i]
                        )

                        stu_img_pos_reps.append(last_stu_img_hidden_state.mean(dim=0))
                        tea_img_pos_reps.append(last_tea_img_hidden_state.mean(dim=0))

                        cur_idx_pos_img += 1

            if len(stu_img_qry_reps) > 0:
                stu_img_qry_reps = torch.stack(stu_img_qry_reps, dim=0)
                tea_img_qry_reps = torch.stack(tea_img_qry_reps, dim=0)
                vision_loss += self.linear_cka_loss(stu_img_qry_reps, tea_img_qry_reps)

            if len(stu_img_pos_reps) > 0:
                stu_img_pos_reps = torch.stack(stu_img_pos_reps, dim=0)
                tea_img_pos_reps = torch.stack(tea_img_pos_reps, dim=0)
                vision_loss += self.linear_cka_loss(stu_img_pos_reps, tea_img_pos_reps)

        vision_loss = vision_loss / num_vision_layer_kd

        if len(stu_img_qry_reps) > 0 and len(stu_img_pos_reps) > 0:
            vision_loss = vision_loss / 2
        
        loss_distill = kd_simcse + vision_loss 

        loss = contrastive_loss + self.kd_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }
