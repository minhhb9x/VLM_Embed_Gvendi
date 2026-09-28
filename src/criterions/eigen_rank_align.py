import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text

class ERAlign(nn.Module):
    def __init__(self, args):
        super(ERAlign, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
        
        # KLD Loss thường dùng reduction='batchmean' cho xác suất
        self.kld_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

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
    
    def get_unpadded_hidden(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        valid_indices = attention_mask.nonzero(as_tuple=True)[0]
        unpadded_hidden_state = hidden_state[valid_indices, :]
        return unpadded_hidden_state
    
    def compute_spectral_prob(
        self,
        hidden_state: torch.Tensor,
        eps: float = 1e-8,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Tính phân phối xác suất dựa trên năng lượng của Singular Values.
        Returns: Tensor xác suất [K], sum(prob) = 1.
        """
        # SVD yêu cầu float32 để ổn định
        X = hidden_state.float() 

        # Singular values: s_i được sắp xếp giảm dần tự động bởi torch
        s = torch.linalg.svdvals(X)   # [min(N, D)]

        # Eigenvalues của X^T X (Energy)
        eigvals = s.pow(2)

        if top_k is not None and eigvals.numel() > top_k:
            eigvals = eigvals[:top_k]

        # Tránh chia cho 0 hoặc giá trị quá nhỏ
        eigvals = eigvals.clamp(min=eps)

        # Normalize thành phân phối xác suất (Probability Distribution)
        prob = eigvals / eigvals.sum()

        return prob

    def wasserstein_1d_loss(self, stu_prob: Tensor, tea_prob: Tensor) -> Tensor:
        # cumulative mass
        stu_cdf = torch.cumsum(stu_prob, dim=-1)
        tea_cdf = torch.cumsum(tea_prob, dim=-1)

        # Pad về max length (zero-padding ở tail là hợp lệ trong OT)
        max_len = max(stu_cdf.shape[-1], tea_cdf.shape[-1])

        stu_cdf = F.pad(stu_cdf, (0, max_len - stu_cdf.shape[-1]), value=1.0)
        tea_cdf = F.pad(tea_cdf, (0, max_len - tea_cdf.shape[-1]), value=1.0)

        return torch.mean(torch.abs(stu_cdf - tea_cdf))

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
        
        # --- Contrastive Loss Part ---
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
        
        # --- Spectral KLD Loss Part ---
        # Note: Không cần dùng alpha để scale giá trị rank nữa vì ta so sánh phân phối xác suất (tổng = 1)
        
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
        
        loss_vision_eigen_rank = 0.0
        loss_text_eigen_rank = 0.0

        # Accumulators để tránh chia cho 0 nếu batch rỗng (dù hiếm)
        count_vision = 0
        count_text = 0

        for i in range(batch_size):
            # 1. QUERY Processing
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    # --- Vision ---
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                    
                    stu_prob = self.compute_spectral_prob(stu_feat)
                    tea_prob = self.compute_spectral_prob(tea_feat)
                    
                    loss_vision_eigen_rank += self.wasserstein_1d_loss(stu_prob, tea_prob)
                    count_vision += 1

                    # --- Text (Multimedia case) ---
                    last_stu_text_hidden_state, _ = get_hidden_text_vision(
                        student_qry_hidden_states[-1][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    last_tea_text_hidden_state, _ = get_hidden_text_vision(
                        teacher_qry_hidden_states[-1][i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    stu_text_prob = self.compute_spectral_prob(last_stu_text_hidden_state)
                    tea_text_prob = self.compute_spectral_prob(last_tea_text_hidden_state)
                    
                    loss_text_eigen_rank += self.wasserstein_1d_loss(stu_text_prob, tea_text_prob)
                    count_text += 1

                    cur_idx_qry_img += 1
            else:
                # --- Text Only case ---
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

                stu_text_prob = self.compute_spectral_prob(last_stu_text_hidden_state)
                tea_text_prob = self.compute_spectral_prob(last_tea_text_hidden_state)
                
                loss_text_eigen_rank += self.wasserstein_1d_loss(stu_text_prob, tea_text_prob)
                count_text += 1

            # 2. POSITIVE Processing (Tương tự Query)
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    # --- Vision ---
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    stu_prob = self.compute_spectral_prob(stu_feat_pos)
                    tea_prob = self.compute_spectral_prob(tea_feat_pos)

                    loss_vision_eigen_rank += self.wasserstein_1d_loss(stu_prob, tea_prob)
                    count_vision += 1

                    # --- Text ---
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

                    stu_text_prob = self.compute_spectral_prob(last_stu_text_hidden_state)
                    tea_text_prob = self.compute_spectral_prob(last_tea_text_hidden_state)

                    loss_text_eigen_rank += self.wasserstein_1d_loss(stu_text_prob, tea_text_prob)
                    count_text += 1

                    cur_idx_pos_img += 1
            else:
                # --- Text Only ---
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

                stu_text_prob = self.compute_spectral_prob(last_stu_text_hidden_state)
                tea_text_prob = self.compute_spectral_prob(last_tea_text_hidden_state)

                loss_text_eigen_rank += self.wasserstein_1d_loss(stu_text_prob, tea_text_prob)
                count_text += 1

        if count_vision > 0:
            loss_vision_eigen_rank = loss_vision_eigen_rank / count_vision
        
        if count_text > 0:
            loss_text_eigen_rank = loss_text_eigen_rank / count_text
        
        loss_distill = 0.5 * loss_vision_eigen_rank + 0.5 * loss_text_eigen_rank

        loss = contrastive_loss + self.kd_loss_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }