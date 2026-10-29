import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_attn_text_vision, pooling, extract_modality_tokens

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
        # proj_teacher_qry_reps = projectors['t2s'](all_teacher_qry_reps)
        # proj_teacher_pos_reps = projectors['t2s'](all_teacher_pos_reps)

        # Tính MSE giữa các đại diện đã được projector
        mse_loss = torch.zeros_like(contrastive_loss)
        # mse_loss = nn.MSELoss()(torch.cat([all_student_qry_reps, all_student_pos_reps], dim=0), 
        #                         torch.cat([proj_teacher_qry_reps, proj_teacher_pos_reps], dim=0))
        
        stu_qry = extract_modality_tokens(
            student_processor, student_qry_input, student_qry_image_features,
            student_qry_attention, student_qry_hidden_states, student_model.model_backbone
        )
        stu_pos = extract_modality_tokens(
            student_processor, student_pos_input, student_pos_image_features,
            student_pos_attention, student_pos_hidden_states, student_model.model_backbone
        )
        tea_qry = extract_modality_tokens(
            teacher_processor, teacher_qry_input, teacher_qry_image_features,
            teacher_qry_attention, teacher_qry_hidden_states, teacher_model.model_backbone
        )
        tea_pos = extract_modality_tokens(
            teacher_processor, teacher_pos_input, teacher_pos_image_features,
            teacher_pos_attention, teacher_pos_hidden_states, teacher_model.model_backbone
        )

        stu_img, tea_img, stu_txt, tea_txt = [], [], [], []

        for s_batch, t_batch in ((stu_qry, tea_qry), (stu_pos, tea_pos)):
            for s, t in zip(s_batch, t_batch):

                # Skip if either model has an empty modality
                if any(x[k].size(1) == 0 for x in (s, t) for k in ("image", "text")):
                    continue

                for sample, key, buffer in (
                    (s, "image", stu_img),
                    (t, "image", tea_img),
                    (s, "text", stu_txt),
                    (t, "text", tea_txt),
                ):
                    hidden = sample[key][-1]             # [N, D]
                    attn = sample[f"{key}_attention"]

                    pooled = (
                        self.attention_pool(hidden, attn[-1])
                        if attn is not None else hidden.mean(dim=0)
                    )
                    buffer.append(pooled)


        intra_loss = contrastive_loss.new_zeros(())
        inter_loss = contrastive_loss.new_zeros(())

        if len(stu_img) > 1:
            sv, tv, st, tt = [
                torch.stack(x) for x in (stu_img, tea_img, stu_txt, tea_txt)
            ]

            img_proj = projectors["t2s_img"]
            txt_proj = projectors["t2s_txt"]

            tv = img_proj(tv.to(next(img_proj.parameters()).dtype))
            tt = txt_proj(tt.to(next(txt_proj.parameters()).dtype))

            intra_loss = self.mi_loss(sv, tv) + self.mi_loss(st, tt)
            inter_loss = self.mi_loss(sv, tt) + self.mi_loss(st, tv)

        scl_loss = intra_loss + inter_loss

        # pga loss
        pga_loss = self.pga_loss(torch.cat([all_student_qry_reps, all_student_pos_reps], dim=0), 
                                 torch.cat([all_teacher_qry_reps, all_teacher_pos_reps], dim=0))

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