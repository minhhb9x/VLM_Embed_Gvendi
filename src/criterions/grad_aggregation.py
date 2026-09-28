import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text, get_unpadded_hidden

class GradPoolingLoss(nn.Module):
    def __init__(self, args):
        super(GradPoolingLoss, self).__init__()
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
    
    def calculate_grad_aggregation_pooling(self, hidden_state, hidden_grad):
        hidden_grad = hidden_grad.norm(p=2, dim=-1) # [seqlen]
        hidden_grad = hidden_grad / (hidden_grad.sum() + 1e-8) # [seqlen]
        pooling = (hidden_state * hidden_grad.unsqueeze(-1)).sum(dim=0) # [hidden_dim]
        return pooling

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

        for p in teacher_model.encoder.lm_head.parameters():
            p.requires_grad = True

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
        
        stu_scores = student_qry_reps * student_pos_reps # [b, d] * [b, d] -> [b, d]
        stu_scores = stu_scores.sum(dim=-1) # [b, d] -> [b]
        tea_scores = teacher_qry_reps * teacher_pos_reps # [b, d] * [b, d] -> [b, d]
        tea_scores = tea_scores.sum(dim=-1) # [b, d] -> [b]

        stu_qry_grads = torch.autograd.grad(stu_scores.sum(), student_qry_hidden_states[-2], retain_graph=True)[0] # [b, seq_len, hidden_dim]
        tea_qry_grads = torch.autograd.grad(tea_scores.sum(), teacher_qry_hidden_states[-2], retain_graph=True)[0] # [b, seq_len, hidden_dim]  

        stu_pos_grads = torch.autograd.grad(stu_scores.sum(), student_pos_hidden_states[-2], retain_graph=True)[0] # [b, seq_len, hidden_dim]
        tea_pos_grads = torch.autograd.grad(tea_scores.sum(), teacher_pos_hidden_states[-2], retain_graph=True)[0] # [b, seq_len, hidden_dim]

        stu_qry_grads = stu_qry_grads.detach()
        tea_qry_grads = tea_qry_grads.detach()
        stu_pos_grads = stu_pos_grads.detach()
        tea_pos_grads = tea_pos_grads.detach()

        loss_distill = 0.0

        loss_vision = 0.0
        loss_last_text = 0.0

        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        for i in range(batch_size):
            # --- Xử lý QUERY Image ---
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                # Kiểm tra index hợp lệ
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    stu_feat_qry = student_qry_image_features[cur_idx_qry_img]
                    tea_feat_qry = teacher_qry_image_features[cur_idx_qry_img]

                    penultimate_stu_text_hidden_state, penultimate_stu_vision_hidden_state = get_hidden_text_vision(
                        student_qry_hidden_states[-2][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat_qry.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    penultimate_stu_text_grads, penultimate_stu_vision_grads = get_hidden_text_vision(
                        stu_qry_grads[i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat_qry.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    penultimate_tea_text_hidden_state, penultimate_tea_vision_hidden_state = get_hidden_text_vision(
                        teacher_qry_hidden_states[-2][i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat_qry.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    penultimate_tea_text_grads, penultimate_tea_vision_grads = get_hidden_text_vision(
                        tea_qry_grads[i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat_qry.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    stu_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_stu_text_hidden_state, penultimate_stu_text_grads)
                    stu_pooled_vision = self.calculate_grad_aggregation_pooling(penultimate_stu_vision_hidden_state, penultimate_stu_vision_grads)
                    tea_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_tea_text_hidden_state, penultimate_tea_text_grads)
                    tea_pooled_vision = self.calculate_grad_aggregation_pooling(penultimate_tea_vision_hidden_state, penultimate_tea_vision_grads)

                    tea_proj_pooled_text = self.distiller.projectors['t2s_txt'](tea_pooled_text)
                    tea_proj_pooled_vision = self.distiller.projectors['t2s_img'](tea_pooled_vision)

                    loss_vision += F.mse_loss(stu_pooled_vision, tea_proj_pooled_vision)
                    loss_last_text += F.mse_loss(stu_pooled_text, tea_proj_pooled_text)

                    cur_idx_qry_img += 1
            # no vision tokens
            else:
                penultimate_stu_text_hidden_state = get_hidden_text(
                    student_qry_hidden_states[-2][i],
                    num_student_text_qry_tokens[i].item(),
                    student_qry_input['attention_mask'][i]
                )

                penultimate_stu_text_grads = get_hidden_text(
                    stu_qry_grads[i],
                    num_student_text_qry_tokens[i].item(),
                    student_qry_input['attention_mask'][i]
                )

                penultimate_tea_text_hidden_state = get_hidden_text(
                    teacher_qry_hidden_states[-2][i],
                    num_teacher_text_qry_tokens[i].item(),
                    teacher_qry_input['attention_mask'][i]
                )

                penultimate_tea_text_grads = get_hidden_text(
                    tea_qry_grads[i],
                    num_teacher_text_qry_tokens[i].item(),
                    teacher_qry_input['attention_mask'][i]
                )

                stu_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_stu_text_hidden_state, penultimate_stu_text_grads)
                tea_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_tea_text_hidden_state, penultimate_tea_text_grads)
                tea_proj_pooled_text = self.distiller.projectors['t2s_txt'](tea_pooled_text)
                loss_last_text += F.mse_loss(stu_pooled_text, tea_proj_pooled_text)

            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    penultimate_stu_text_hidden_state, penultimate_stu_vision_hidden_state = get_hidden_text_vision(
                        student_pos_hidden_states[-2][i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    penultimate_stu_text_grads, penultimate_stu_vision_grads = get_hidden_text_vision(
                        stu_pos_grads[i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    penultimate_tea_text_hidden_state, penultimate_tea_vision_hidden_state = get_hidden_text_vision(
                        teacher_pos_hidden_states[-2][i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    penultimate_tea_text_grads, penultimate_tea_vision_grads = get_hidden_text_vision(
                        tea_pos_grads[i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    stu_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_stu_text_hidden_state, penultimate_stu_text_grads)
                    stu_pooled_vision = self.calculate_grad_aggregation_pooling(penultimate_stu_vision_hidden_state, penultimate_stu_vision_grads)
                    tea_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_tea_text_hidden_state, penultimate_tea_text_grads)
                    tea_pooled_vision = self.calculate_grad_aggregation_pooling(penultimate_tea_vision_hidden_state, penultimate_tea_vision_grads)

                    tea_proj_pooled_text = self.distiller.projectors['t2s_txt'](tea_pooled_text)
                    tea_proj_pooled_vision = self.distiller.projectors['t2s_img'](tea_pooled_vision)

                    loss_vision += F.mse_loss(stu_pooled_vision, tea_proj_pooled_vision)
                    loss_last_text += F.mse_loss(stu_pooled_text, tea_proj_pooled_text)
                    
                    cur_idx_pos_img += 1
            # no vision tokens
            else:
                penultimate_stu_text_hidden_state = get_hidden_text(
                    student_pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    student_pos_input['attention_mask'][i]
                )

                penultimate_stu_text_grads = get_hidden_text(
                    stu_pos_grads[i],
                    num_student_text_pos_tokens[i].item(),
                    student_pos_input['attention_mask'][i]
                )

                penultimate_tea_text_hidden_state = get_hidden_text(
                    teacher_pos_hidden_states[-1][i],
                    num_teacher_text_pos_tokens[i].item(),
                    teacher_pos_input['attention_mask'][i]
                )

                penultimate_tea_text_grads = get_hidden_text(
                    tea_pos_grads[i],
                    num_teacher_text_pos_tokens[i].item(),
                    teacher_pos_input['attention_mask'][i]
                )

                stu_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_stu_text_hidden_state, penultimate_stu_text_grads)
                tea_pooled_text = self.calculate_grad_aggregation_pooling(penultimate_tea_text_hidden_state, penultimate_tea_text_grads)
                tea_proj_pooled_text = self.distiller.projectors['t2s_txt'](tea_pooled_text)
                loss_last_text += F.mse_loss(stu_pooled_text, tea_proj_pooled_text)
       
        loss_vision = loss_vision / (cur_idx_qry_img + cur_idx_pos_img + 1e-8)
        loss_last_text = loss_last_text / batch_size

        loss_distill = (loss_vision + loss_last_text) / 2 

        loss = contrastive_loss + self.kd_loss_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }