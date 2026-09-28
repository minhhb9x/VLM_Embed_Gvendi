import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text, get_unpadded_hidden

class EffectiveRankLoss(nn.Module):
    def __init__(self, args):
        super(EffectiveRankLoss, self).__init__()
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
    
    def compute_effective_rank(
        self,
        hidden_state: torch.Tensor, # [N, D]
        eps: float = 1e-10,
    ) -> torch.Tensor:
        X = hidden_state.float() 
        N = X.size(0)
        s = torch.linalg.svdvals(X) / torch.sqrt(torch.tensor(N))
        eigvals = s * s 
        prob = eigvals.clamp(min=eps) / eigvals.sum()        
        entropy = -(prob * torch.log(prob)).sum()
        effective_rank = torch.exp(entropy) / N

        return effective_rank.to(dtype=hidden_state.dtype)

    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher

        if getattr(self, "student_processor", None) is None:
            self.student_processor = distiller.get_student_processor()
        if getattr(self, "teacher_processor", None) is None:
            self.teacher_processor = distiller.get_teacher_processor()

        student_processor = self.student_processor
        teacher_processor = self.teacher_processor

        student_tokenizer = student_processor.tokenizer
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

        loss_distill = 0.0

        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        student_special_ids = torch.tensor(
            list(
                set(
                    list(student_tokenizer.added_tokens_encoder.values()) +
                    student_tokenizer.all_special_ids
                )
            ),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        teacher_special_ids = torch.tensor(
            list(
                set(
                    list(teacher_tokenizer.added_tokens_encoder.values()) +
                    teacher_tokenizer.all_special_ids
                )
            ),
            device=teacher_qry_input['input_ids'].device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        num_teacher_text_qry_tokens = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
        num_teacher_text_pos_tokens = count_clean_text_tokens(teacher_pos_input, teacher_special_ids)
        
        loss_vision_er = 0.0
        loss_last_text_er = 0.0

        for i in range(batch_size):
            # --- Xử lý QUERY Image ---
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                # Kiểm tra index hợp lệ
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    stu_feat_qry = student_qry_image_features[cur_idx_qry_img]
                    tea_feat_qry = teacher_qry_image_features[cur_idx_qry_img]
                    loss_vision_er += nn.L1Loss()(self.compute_effective_rank(stu_feat_qry), 
                                                  self.compute_effective_rank(tea_feat_qry))

                    last_stu_text_hidden_state, _ = get_hidden_text_vision(
                        student_qry_hidden_states[-1][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat_qry.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    last_tea_text_hidden_state, _ = get_hidden_text_vision(
                        teacher_qry_hidden_states[-1][i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat_qry.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    loss_last_text_er += nn.L1Loss()(
                        self.compute_effective_rank(last_stu_text_hidden_state),
                        self.compute_effective_rank(last_tea_text_hidden_state)
                    )

                    cur_idx_qry_img += 1
            # no vision tokens
            else:
                last_stu_text_hidden_state = get_hidden_text(
                    student_qry_hidden_states[-1][i],
                    num_student_text_qry_tokens[i].item(),
                    student_qry_input['attention_mask'][i]
                )

                last_tea_text_hidden_state = get_hidden_text(
                    teacher_qry_hidden_states[-1][i],
                    num_teacher_text_qry_tokens[i].item(),
                    teacher_qry_input['attention_mask'][i]
                )

                loss_last_text_er += nn.L1Loss()(
                    self.compute_effective_rank(last_stu_text_hidden_state),
                    self.compute_effective_rank(last_tea_text_hidden_state) 
                )

            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    loss_vision_er += nn.L1Loss()(self.compute_effective_rank(stu_feat_pos), 
                                                  self.compute_effective_rank(tea_feat_pos))

                    last_stu_text_hidden_state, _ = get_hidden_text_vision(
                        student_pos_hidden_states[-1][i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    last_tea_text_hidden_state, _ = get_hidden_text_vision(
                        teacher_pos_hidden_states[-1][i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    loss_last_text_er += nn.L1Loss()(
                        self.compute_effective_rank(last_stu_text_hidden_state),
                        self.compute_effective_rank(last_tea_text_hidden_state) 
                    )

                    cur_idx_pos_img += 1
            # no vision tokens
            else:
                last_stu_text_hidden_state = get_hidden_text(
                    student_pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    student_pos_input['attention_mask'][i]
                )

                last_tea_text_hidden_state = get_hidden_text(
                    teacher_pos_hidden_states[-1][i],
                    num_teacher_text_pos_tokens[i].item(),
                    teacher_pos_input['attention_mask'][i]
                )

                loss_last_text_er += nn.L1Loss()(
                    self.compute_effective_rank(last_stu_text_hidden_state),
                    self.compute_effective_rank(last_tea_text_hidden_state)
                )

        loss_vision_er = loss_vision_er / (cur_idx_qry_img + cur_idx_pos_img + 1e-8)
        loss_last_text_er = loss_last_text_er / (2*batch_size + 1e-8)
        
        # stu_er_qry = self.compute_effective_rank(all_student_qry_reps)
        # stu_er_pos = self.compute_effective_rank(all_student_pos_reps)
        # tea_er_qry = self.compute_effective_rank(all_teacher_qry_reps)
        # tea_er_pos = self.compute_effective_rank(all_teacher_pos_reps)

        # loss_distill = 0.5 * (F.relu(tea_er_qry - stu_er_qry) + 
        #                       F.relu(tea_er_pos - stu_er_pos))

        loss_distill = 0.5 * (loss_vision_er + loss_last_text_er)

        loss = contrastive_loss + self.kd_loss_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }