import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .utils import count_clean_text_tokens, get_grid_size, get_hidden_text_vision, build_center_relative_grid

class VisionRKDLoss(nn.Module):
    def __init__(self, args):
        super(VisionRKDLoss, self).__init__()
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
    
    def _vision_alignment_loss(self, 
                              student_reps, teacher_reps,            # Vector embedding eos
                              stu_hidden_state, tea_hidden_state,    # Hidden state tại layer cuối của sample i
                              num_stu_text_tok, num_tea_text_tok,    # Số lượng token text
                              stu_img_feat, tea_img_feat,            # Tensor vision features cụ thể của ảnh đang xét
                              stu_grid_size, tea_grid_size,          # Grid size (H, W)
                              stu_attention_mask, tea_attention_mask):   # Attention masks
    
        # 1. Lấy kích thước token vision thực tế
        num_tokens_vision_stu = stu_img_feat.size(0)
        num_tokens_vision_tea = tea_img_feat.size(0)

        # 2. Tách vision hidden states từ chuỗi sequence (loại bỏ text)
        _, last_stu_vision_state = get_hidden_text_vision(
            stu_hidden_state, 
            num_stu_text_tok, 
            num_tokens_vision_stu, 
            attention_mask=stu_attention_mask
        )

        _, last_tea_vision_state = get_hidden_text_vision(
            tea_hidden_state, 
            num_tea_text_tok, 
            num_tokens_vision_tea, 
            attention_mask=tea_attention_mask
        )

        if student_reps.dim() == 1: student_reps = student_reps.unsqueeze(0) # (1, D)
        if teacher_reps.dim() == 1: teacher_reps = teacher_reps.unsqueeze(0) # (1, D)

        last_stu_vision_norm = F.normalize(last_stu_vision_state, p=2, dim=-1) # (N_s, D)
        last_tea_vision_norm = F.normalize(last_tea_vision_state, p=2, dim=-1) # (N_t, D)

        last_stu_vision_eos_sim = last_stu_vision_norm @ F.normalize(student_reps, p=2, dim=-1).T  # (N_s, 1)
        last_tea_vision_eos_sim = last_tea_vision_norm @ F.normalize(teacher_reps, p=2, dim=-1).T  # (N_t, 1)

        c2 = (last_stu_vision_eos_sim - last_tea_vision_eos_sim.T).abs()  # (N_s, N_t)

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
        matched_last_tea_tea_vision_state = last_tea_vision_state[matched_tea_idx, :] # (N_s, )


        e_stu = last_stu_vision_state - student_reps  # (N_s, D)
        e_tea = matched_last_tea_tea_vision_state - teacher_reps # (N_s, D)

        e_stu_norm = F.normalize(e_stu, p=2, dim=1) # (N_s, D)
        e_tea_norm = F.normalize(e_tea, p=2, dim=1) # (N_s, D)

        cos_matrix_stu = torch.matmul(e_stu_norm, e_stu_norm.t()) # (N_s, N_s)
        cos_matrix_tea = torch.matmul(e_tea_norm, e_tea_norm.t()) # (N_s, N_s)

        huber_loss = nn.HuberLoss(delta=1.0, reduction='mean')
        angle_loss = huber_loss(cos_matrix_stu, cos_matrix_tea.detach())

        return angle_loss


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
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        
        loss_distill = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0


        student_special_ids = torch.tensor(student_tokenizer.all_special_ids, device=student_qry_input['input_ids'].device)
        teacher_special_ids = torch.tensor(teacher_tokenizer.all_special_ids, device=teacher_qry_input['input_ids'].device)

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        num_teacher_text_qry_tokens = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
        num_teacher_text_pos_tokens = count_clean_text_tokens(teacher_pos_input, teacher_special_ids)
        
        stu_qry_vision_grid_sizes = get_grid_size(student_model, student_qry_input)
        tea_qry_vision_grid_sizes = get_grid_size(teacher_model, teacher_qry_input)

        stu_pos_vision_grid_sizes = get_grid_size(student_model, student_pos_input)
        tea_pos_vision_grid_sizes = get_grid_size(teacher_model, teacher_pos_input)
        
        for i in range(batch_size):
            # --- Xử lý QUERY Image ---
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                # Kiểm tra index hợp lệ
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                    
                    # Kiểm tra feature không phải None
                    if stu_feat is not None and tea_feat is not None:
                        vision_rkd_loss = self._vision_alignment_loss(
                            student_reps=student_qry_reps[i],
                            teacher_reps=teacher_qry_reps[i],
                            stu_hidden_state=student_qry_hidden_states[-1][i],
                            tea_hidden_state=teacher_qry_hidden_states[-1][i],
                            num_stu_text_tok=num_student_text_qry_tokens[i].item(),
                            num_tea_text_tok=num_teacher_text_qry_tokens[i].item(),
                            stu_img_feat=stu_feat,
                            tea_img_feat=tea_feat,
                            stu_grid_size=stu_qry_vision_grid_sizes[cur_idx_qry_img],
                            tea_grid_size=tea_qry_vision_grid_sizes[cur_idx_qry_img],
                            stu_attention_mask=student_qry_input['attention_mask'][i],
                            tea_attention_mask=teacher_qry_input['attention_mask'][i]
                        )
                        loss_distill += vision_rkd_loss
                        cur_idx_qry_img += 1

            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    if stu_feat_pos is not None and tea_feat_pos is not None:
                        vision_rkd_loss = self._vision_alignment_loss(
                            student_reps=student_pos_reps[i],
                            teacher_reps=teacher_pos_reps[i],
                            stu_hidden_state=student_pos_hidden_states[-1][i],  # Lưu ý dùng pos hidden states
                            tea_hidden_state=teacher_pos_hidden_states[-1][i],
                            num_stu_text_tok=num_student_text_pos_tokens[i].item(), # Lưu ý dùng pos tokens count
                            num_tea_text_tok=num_teacher_text_pos_tokens[i].item(),
                            stu_img_feat=stu_feat_pos,
                            tea_img_feat=tea_feat_pos,
                            stu_grid_size=stu_pos_vision_grid_sizes[cur_idx_pos_img], # Lưu ý dùng pos grid size
                            tea_grid_size=tea_pos_vision_grid_sizes[cur_idx_pos_img],
                            stu_attention_mask=student_pos_input['attention_mask'][i],
                            tea_attention_mask=teacher_pos_input['attention_mask'][i]
                        )
                        loss_distill += vision_rkd_loss
                        cur_idx_pos_img += 1
        
        loss_distill = loss_distill / (cur_idx_qry_img + cur_idx_pos_img + 1e-8) # mean loss over proceesed images
        loss = contrastive_loss + loss_distill * self.kd_loss_weight

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }