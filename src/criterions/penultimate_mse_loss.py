import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .utils import get_grid_size, get_hidden_text_vision, build_center_relative_grid

class PenultimateMSELoss(nn.Module):
    def __init__(self, args):
        super(PenultimateMSELoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
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
    
    def _penultimate_loss(self, projectors, # Projectors 
                            stu_hidden_state, tea_hidden_state,
                            num_stu_text_tok, num_tea_text_tok,  # Hidden state tại layer x của sample i
                            stu_img_feat, tea_img_feat,
                            stu_grid_size, tea_grid_size,
                            stu_attention_mask, tea_attention_mask
                            ):
            num_tokens_vision_stu = stu_img_feat.size(0) if stu_img_feat is not None else 0
            num_tokens_vision_tea = tea_img_feat.size(0) if tea_img_feat is not None else 0
            
            _, stu_vision_hidden_state = get_hidden_text_vision(
                stu_hidden_state,
                num_text_token=num_stu_text_tok,
                num_vision_token=num_tokens_vision_stu,
                attention_mask=stu_attention_mask
            )
            _, tea_vision_hidden_state = get_hidden_text_vision(
                tea_hidden_state,
                num_text_token=num_tea_text_tok,
                num_vision_token=num_tokens_vision_tea,
                attention_mask=tea_attention_mask
            )
            
            projected_tea_vision_hidden_state = projectors['t2s_img'](tea_vision_hidden_state)

            norm_stu_vision_hidden_state = F.normalize(stu_vision_hidden_state, dim=-1) # (Ns, D)
            norm_projected_tea_vision_hidden_state = F.normalize(projected_tea_vision_hidden_state, dim=-1) # (Nt, D)

            c2 = torch.cdist(norm_stu_vision_hidden_state, 
                             norm_projected_tea_vision_hidden_state) # (Ns, Nt)


            stu_grid = build_center_relative_grid(stu_grid_size[0], stu_grid_size[1],
                                            device=stu_hidden_state.device,
                                            dtype=stu_hidden_state.dtype).reshape(-1, 2)
        
            tea_grid = build_center_relative_grid(tea_grid_size[0], tea_grid_size[1],
                                            device=tea_hidden_state.device,
                                            dtype=tea_hidden_state.dtype).reshape(-1, 2)

            c1 = (stu_grid.unsqueeze(1) - tea_grid.unsqueeze(0)).abs().sum(dim=-1) # (N_s, N_t)
            c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-6)
            c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-6)

            total_cost = c1 + 0.001 * c2
            matched_tea_idx = total_cost.argmin(dim=1) # (N_s,)

            matched_norm_tea_vision_state = norm_projected_tea_vision_hidden_state[matched_tea_idx, :] # (N_s, D)


            loss = self.mse_loss(stu_vision_hidden_state, matched_norm_tea_vision_state)
            
            return loss

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher
        projectors = distiller.projectors

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
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # KD loss on representations 
        projected_teacher_qry_reps = F.normalize(projectors["t2s"](teacher_qry_reps), p=2, dim=-1)
        projected_teacher_pos_reps = F.normalize(projectors["t2s"](teacher_pos_reps), p=2, dim=-1)
        kd_mse_loss_reps = 0.25 * (self.mse_loss(student_qry_reps, projected_teacher_qry_reps) + self.mse_loss(student_pos_reps, projected_teacher_pos_reps) + \
                                        self.mse_loss(student_qry_reps, projected_teacher_pos_reps) + self.mse_loss(student_pos_reps, projected_teacher_qry_reps))
        
        penultimate_loss = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        # KD loss on penultimate layer
        student_special_ids = torch.tensor(student_tokenizer.all_special_ids, device=student_qry_input['input_ids'].device)
        teacher_special_ids = torch.tensor(teacher_tokenizer.all_special_ids, device=teacher_qry_input['input_ids'].device)

        num_student_text_qry_tokens = (~torch.isin(student_qry_input['input_ids'], 
                                                   student_special_ids)).sum(dim=1)
        num_student_text_pos_tokens = (~torch.isin(student_pos_input['input_ids'], 
                                                   student_special_ids)).sum(dim=1)

        num_teacher_text_qry_tokens = (~torch.isin(teacher_qry_input['input_ids'], 
                                                   teacher_special_ids)).sum(dim=1)
        num_teacher_text_pos_tokens = (~torch.isin(teacher_pos_input['input_ids'], 
                                                   teacher_special_ids)).sum(dim=1)
        
        stu_qry_vision_grid_sizes = get_grid_size(student_model, student_qry_input)
        tea_qry_vision_grid_sizes = get_grid_size(teacher_model, teacher_qry_input)

        stu_pos_vision_grid_sizes = get_grid_size(student_model, student_pos_input)
        tea_pos_vision_grid_sizes = get_grid_size(teacher_model, teacher_pos_input)

        for i in range(batch_size):
            stu_feat = None
            tea_feat = None
            if student_qry_image_features is not None and teacher_qry_image_features is not None \
                and cur_idx_qry_img < len(student_qry_image_features):
                stu_feat = student_qry_image_features[cur_idx_qry_img]
                tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                penultimate_loss += self._penultimate_loss(
                    projectors,
                    stu_hidden_state=student_qry_hidden_states[-2][i],
                    tea_hidden_state=teacher_qry_hidden_states[-2][i],
                    num_stu_text_tok=num_student_text_qry_tokens[i].item(),
                    num_tea_text_tok=num_teacher_text_qry_tokens[i].item(),
                    stu_img_feat=stu_feat,
                    tea_img_feat=tea_feat,
                    stu_grid_size=stu_qry_vision_grid_sizes[cur_idx_qry_img],
                    tea_grid_size=tea_qry_vision_grid_sizes[cur_idx_qry_img],
                    stu_attention_mask=student_qry_input['attention_mask'][i],
                    tea_attention_mask=teacher_qry_input['attention_mask'][i]
                )
                cur_idx_qry_img += 1

            stu_feat = None
            tea_feat = None
            if student_pos_image_features is not None and teacher_pos_image_features is not None \
                and cur_idx_pos_img < len(student_pos_image_features):
                stu_feat = student_pos_image_features[cur_idx_pos_img]
                tea_feat = teacher_pos_image_features[cur_idx_pos_img]
                penultimate_loss += self._penultimate_loss(
                    projectors,
                    stu_hidden_state=student_pos_hidden_states[-2][i],
                    tea_hidden_state=teacher_pos_hidden_states[-2][i],
                    num_stu_text_tok=num_student_text_pos_tokens[i].item(),
                    num_tea_text_tok=num_teacher_text_pos_tokens[i].item(),
                    stu_img_feat=stu_feat,
                    tea_img_feat=tea_feat,
                    stu_grid_size=stu_pos_vision_grid_sizes[cur_idx_pos_img],
                    tea_grid_size=tea_pos_vision_grid_sizes[cur_idx_pos_img],
                    stu_attention_mask=student_pos_input['attention_mask'][i],
                    tea_attention_mask=teacher_pos_input['attention_mask'][i]
                )
                cur_idx_pos_img += 1
       
        penultimate_loss = penultimate_loss / (cur_idx_qry_img + cur_idx_pos_img + 1e-8) # mean loss over proceesed images

        kd_loss = 0.5 * kd_mse_loss_reps + 0.5 * penultimate_loss
        loss = contrastive_loss + kd_loss * self.kd_loss_weight

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_mse_loss': kd_mse_loss_reps,
            'kd_loss': kd_loss,
            'kd_penultimate_loss': penultimate_loss
        }