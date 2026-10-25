import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_attn_text_vision

class PGA(nn.Module):
    def __init__(self, args):
        super(PGA, self).__init__()
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


    def mi_loss(
        self,
        student_features: torch.Tensor,  # [B, D]
        teacher_features: torch.Tensor,  # [B, D]
        temperature: float = 0.02,
    ) -> torch.Tensor:
        """Equation (13) của PGA-KD: student_i khớp teacher_i."""
        if student_features.ndim != 2 or student_features.shape != teacher_features.shape:
            raise ValueError("student_features và teacher_features phải cùng shape [B, D]")
        if temperature <= 0:
            raise ValueError("temperature phải > 0")

        student = F.normalize(student_features.float(), dim=-1)
        teacher = F.normalize(teacher_features.float(), dim=-1)

        # logits[i, j] = cosine(student_i, teacher_j) / temperature
        logits = student @ teacher.T / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)

    def extract_last_mean_attention(self, attentions, attention_mask):
        attn = attentions[-1].mean(dim=1)  # [B, L, L]

        batch_size = attn.size(0)
        max_length = attn.size(-1)

        left_padding = (
            attention_mask[:, -1].sum() == batch_size
        )

        if left_padding:
            eos_indices = torch.full(
                (batch_size,),
                max_length - 1,
                device=attn.device,
                dtype=torch.long
            )
        else:
            num_padding_tokens = (attention_mask == 0).sum(dim=1)
            eos_indices = (
                max_length - num_padding_tokens - 1
            ).to(attn.device)

        return attn[
            torch.arange(batch_size, device=attn.device),
            eos_indices,
            :
        ]

    @staticmethod
    def attention_pool(hidden_states, attention_scores, eps=1e-8):
        """
        hidden_states: [N, D]
        attention_scores: [N]
        """
        assert hidden_states.ndim == 2
        assert attention_scores.ndim == 1
        assert hidden_states.size(0) == attention_scores.size(0)
        assert hidden_states.size(0) > 0

        weights = attention_scores / attention_scores.sum().clamp_min(eps)

        return torch.sum(
            hidden_states * weights.unsqueeze(-1),
            dim=0
        )

    # def pga_loss(
    #     self,
    #     student_reps: torch.Tensor,  # [N, D_s]
    #     teacher_reps: torch.Tensor,  # [N, D_t]
    #     eta: float = 0.85,
    #     eps: float = 1e-8,
    # ) -> torch.Tensor:
    #     if student_reps.ndim != 2 or teacher_reps.ndim != 2:
    #         raise ValueError("student_reps và teacher_reps phải có shape [N, D]")
    #     if student_reps.size(0) != teacher_reps.size(0):
    #         raise ValueError("Student và teacher phải có cùng số mẫu")
    #     if student_reps.size(0) < 2:
    #         raise ValueError("PGA cần ít nhất 2 mẫu")
    #     if not 0 < eta <= 1:
    #         raise ValueError("eta phải nằm trong (0, 1]")

    #     # Eq. (7): Gram matrix bằng dot product, không normalize embedding.
    #     # Tính trong float32 để eigendecomposition ổn định hơn khi train bf16.
    #     hs = student_reps.float()
    #     gs = hs @ hs.T  # [N, N]

    #     with torch.no_grad():
    #         ht = teacher_reps.detach().float()
    #         gt = ht @ ht.T  # [N, N]

    #         # Eq. (8)-(9): phân rã phổ và giữ các thành phần chính.
    #         eigenvalues, eigenvectors = torch.linalg.eigh(gt)
    #         eigenvalues = eigenvalues.clamp_min(0)

    #         # eigh trả về thứ tự tăng dần -> đảo lại.
    #         eigenvalues = eigenvalues.flip(0)
    #         eigenvectors = eigenvectors.flip(1)

    #         total_energy = eigenvalues.sum()
    #         if total_energy <= eps:
    #             raise ValueError("Teacher Gram matrix có năng lượng gần bằng 0")

    #         cumulative = eigenvalues.cumsum(dim=0) / total_energy
    #         k = int(torch.searchsorted(cumulative.contiguous(), eta).item()) + 1

    #         top_values = eigenvalues[:k]
    #         top_vectors = eigenvectors[:, :k]
    #         gt_principal = (top_vectors * top_values.unsqueeze(0)) @ top_vectors.T

    #         # Center Gram matrix: C @ G @ C, không cần tạo C tường minh.
    #         gt_centered = (
    #             gt_principal
    #             - gt_principal.mean(dim=0, keepdim=True)
    #             - gt_principal.mean(dim=1, keepdim=True)
    #             + gt_principal.mean()
    #         )

    #     gs_centered = (
    #         gs
    #         - gs.mean(dim=0, keepdim=True)
    #         - gs.mean(dim=1, keepdim=True)
    #         + gs.mean()
    #     )

    #     # Eq. (10): 1 - centered kernel alignment.
    #     numerator = (gs_centered * gt_centered).sum()
    #     student_norm = torch.linalg.vector_norm(gs_centered)
    #     teacher_norm = torch.linalg.vector_norm(gt_centered)

    #     if teacher_norm <= eps:
    #         raise ValueError("Teacher principal Gram matrix sau centering gần bằng 0")

    #     cka = numerator / (student_norm * teacher_norm + eps)
    #     return 1.0 - cka

    
    def pga_loss(
        self, student_reps: torch.Tensor, teacher_reps: torch.Tensor, eps: float = 1e-8,
    ) -> torch.Tensor:

        hs = student_reps.float()
        ht = teacher_reps.detach().float()

        # Center features
        hs = hs - hs.mean(dim=0, keepdim=True)
        ht = ht - ht.mean(dim=0, keepdim=True)

        # Gram matrices
        gs = hs @ hs.T
        gt = ht @ ht.T

        # Linear CKA
        numerator = (gs * gt).sum()

        student_norm = torch.linalg.vector_norm(gs)
        teacher_norm = torch.linalg.vector_norm(gt)

        cka = numerator / (
            student_norm * teacher_norm + eps
        )

        return 1.0 - cka


    def forward(self, distiller, input_data):
        self.distiller = distiller
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
            if self.world_size > 1:
                all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
                all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
            else:
                all_teacher_qry_reps = teacher_qry_reps
                all_teacher_pos_reps = teacher_pos_reps
                        
        
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
        
        
        #------ KD MSE Loss Part ------
        proj_teacher_qry_reps = projectors['t2s'](all_teacher_qry_reps)
        proj_teacher_pos_reps = projectors['t2s'](all_teacher_pos_reps)

        # Tính MSE giữa các đại diện đã được projector
        mse_loss = nn.MSELoss()(torch.cat([all_student_qry_reps, all_student_pos_reps], dim=0), 
                                torch.cat([proj_teacher_qry_reps, proj_teacher_pos_reps], dim=0))
        
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

        # Extract last mean attention for both student and teacher

        stu_last_mean_attn_qry = self.extract_last_mean_attention(student_qry_attention, student_qry_input['attention_mask']) # [B, L]
        stu_last_mean_attn_pos = self.extract_last_mean_attention(student_pos_attention, student_pos_input['attention_mask'])

        tea_last_mean_attn_qry = self.extract_last_mean_attention(teacher_qry_attention, teacher_qry_input['attention_mask'])
        tea_last_mean_attn_pos = self.extract_last_mean_attention(teacher_pos_attention, teacher_pos_input['attention_mask'])

        # SCL loss

        stu_vision_pooled_reps = []
        tea_vision_pooled_reps = []
        stu_text_pooled_reps = []
        tea_text_pooled_reps = []

        for i in range(batch_size):
            # 1. QUERY Processing
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    # --- Vision ---
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                    cur_idx_qry_img += 1
                
                    last_stu_text_hidden_state, last_stu_vision_hidden_state = get_hidden_text_vision(
                        student_qry_hidden_states[-1][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat.size(0),
                        student_qry_input['attention_mask'][i]
                    )
                    

                    last_stu_text_attn, last_stu_vision_attn = get_attn_text_vision(
                        stu_last_mean_attn_qry[i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    last_tea_text_hidden_state, last_tea_vision_hidden_state = get_hidden_text_vision(
                        teacher_qry_hidden_states[-1][i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    last_tea_text_attn, last_tea_vision_attn = get_attn_text_vision(
                        tea_last_mean_attn_qry[i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    stu_text_pooled_rep = self.attention_pool(
                        last_stu_text_hidden_state, last_stu_text_attn
                    )
                    tea_text_pooled_rep = self.attention_pool(
                        last_tea_text_hidden_state, last_tea_text_attn
                    )
                    stu_vision_pooled_rep = self.attention_pool(
                        last_stu_vision_hidden_state, last_stu_vision_attn
                    )
                    tea_vision_pooled_rep = self.attention_pool(
                        last_tea_vision_hidden_state, last_tea_vision_attn
                    )

                    # print('============================')
                    # print('stu_text_pooled_rep.shape:', stu_text_pooled_rep.shape)
                    # print('tea_text_pooled_rep.shape:', tea_text_pooled_rep.shape)
                    # print('stu_vision_pooled_rep.shape:', stu_vision_pooled_rep.shape)
                    # print('tea_vision_pooled_rep.shape:', tea_vision_pooled_rep.shape)

                    stu_text_pooled_reps.append(stu_text_pooled_rep)  # [D]
                    tea_text_pooled_reps.append(tea_text_pooled_rep)  # [D]
                    stu_vision_pooled_reps.append(stu_vision_pooled_rep)  # [D]
                    tea_vision_pooled_reps.append(tea_vision_pooled_rep)  # [D]

            # 2. POSITIVE Processing (Tương tự Query)
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    # --- Vision ---
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]
                    cur_idx_pos_img += 1
                    last_stu_text_hidden_state, last_stu_vision_hidden_state = get_hidden_text_vision(
                        student_pos_hidden_states[-1][i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    last_stu_text_attn, last_stu_vision_attn = get_attn_text_vision(
                        stu_last_mean_attn_pos[i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    last_tea_text_hidden_state, last_tea_vision_hidden_state = get_hidden_text_vision(
                        teacher_pos_hidden_states[-1][i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    last_tea_text_attn, last_tea_vision_attn = get_attn_text_vision(
                        tea_last_mean_attn_pos[i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    stu_text_pooled_rep = self.attention_pool(
                        last_stu_text_hidden_state, last_stu_text_attn
                    )
                    tea_text_pooled_rep = self.attention_pool(
                        last_tea_text_hidden_state, last_tea_text_attn
                    )
                    stu_vision_pooled_rep = self.attention_pool(
                        last_stu_vision_hidden_state, last_stu_vision_attn
                    )
                    tea_vision_pooled_rep = self.attention_pool(
                        last_tea_vision_hidden_state, last_tea_vision_attn
                    )

                    # print('============================')
                    # print('stu_text_pooled_rep.shape:', stu_text_pooled_rep.shape)
                    # print('tea_text_pooled_rep.shape:', tea_text_pooled_rep.shape)
                    # print('stu_vision_pooled_rep.shape:', stu_vision_pooled_rep.shape)
                    # print('tea_vision_pooled_rep.shape:', tea_vision_pooled_rep.shape)

                    stu_text_pooled_reps.append(stu_text_pooled_rep)  # [D]
                    tea_text_pooled_reps.append(tea_text_pooled_rep)  # [D]
                    stu_vision_pooled_reps.append(stu_vision_pooled_rep)  # [D]
                    tea_vision_pooled_reps.append(tea_vision_pooled_rep)  # [D]

        stu_vision_pooled_reps = torch.stack(stu_vision_pooled_reps, dim=0)  # [B, D]
        tea_vision_pooled_reps = torch.stack(tea_vision_pooled_reps, dim=0)  # [B, D]
        stu_text_pooled_reps = torch.stack(stu_text_pooled_reps, dim=0)      # [B, D]
        tea_text_pooled_reps = torch.stack(tea_text_pooled_reps, dim=0)      # [B, D]

        proj_tea_vision_pooled_reps = projectors['t2s_img'](tea_vision_pooled_reps)
        proj_tea_text_pooled_reps = projectors['t2s_txt'](tea_text_pooled_reps)

        # Tính loss giữa các đại diện đã được projector
        intra_loss = self.mi_loss(stu_vision_pooled_reps, proj_tea_vision_pooled_reps) + self.mi_loss(stu_text_pooled_reps, proj_tea_text_pooled_reps)
        inter_loss = self.mi_loss(stu_vision_pooled_reps, proj_tea_text_pooled_reps) + self.mi_loss(stu_text_pooled_reps, proj_tea_vision_pooled_reps)

        scl_loss = intra_loss + inter_loss

        pga_loss = self.pga_loss(torch.cat([stu_vision_pooled_reps, stu_text_pooled_reps], dim=0), 
                                 torch.cat([proj_tea_vision_pooled_reps, proj_tea_text_pooled_reps], dim=0))

        kd_loss = (mse_loss + 0.5*scl_loss + pga_loss)
        
        loss = contrastive_loss + self.kd_loss_weight * kd_loss

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': kd_loss,
            "kd_rkd_loss": pga_loss,
            "kd_loss_mse": mse_loss,
            "cross_modal_loss": scl_loss,
        }