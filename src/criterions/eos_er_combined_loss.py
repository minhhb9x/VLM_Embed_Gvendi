import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text


class EOSERCombinedLoss(nn.Module):
    def __init__(self, args):
        super(EOSERCombinedLoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()

        self.eos_kd_weight = getattr(self.args, "eos_kd_weight", self.args.kd_weight)
        self.er_kd_weight = getattr(self.args, "er_kd_weight", self.args.kd_weight)
        self.projection_space = getattr(self.args, "eos_projection_space", "student")

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

    def _touch_all_projectors(self, reference: torch.Tensor):
        if not hasattr(self.distiller, "projectors") or self.distiller.projectors is None:
            return reference.new_zeros(())

        touched = reference.new_zeros(())
        for param in self.distiller.projectors.parameters():
            touched = touched + (param.reshape(-1)[:1].float().sum() * 0.0).to(reference.dtype)
        return touched

    def _select_projector(self, direction: str, modality: str):
        if not hasattr(self.distiller, "projectors") or self.distiller.projectors is None:
            return None

        projectors = self.distiller.projectors
        direction_to_keys = {
            "s2t": {
                "text": ["s2t_txt", "s2t", "proj_ST"],
                "vision": ["s2t_img", "s2t", "proj_SI", "proj_ST"],
            },
            "t2s": {
                "text": ["t2s_txt"],
                "vision": ["t2s_img"],
            },
        }
        preferred_keys = direction_to_keys[direction][modality]

        if isinstance(projectors, nn.ModuleDict):
            for key in preferred_keys:
                if key in projectors:
                    return projectors[key]
            return None

        if isinstance(projectors, nn.ModuleList) and len(projectors) > 0:
            return projectors[0]

        return None

    def _align_modal_reps(self, teacher_rep: torch.Tensor, student_rep: torch.Tensor, modality: str):
        if teacher_rep.size(-1) == student_rep.size(-1):
            return teacher_rep, student_rep

        if self.projection_space == "student":
            projector = self._select_projector("t2s", modality)
            if projector is None:
                raise ValueError(
                    "Requested teacher->student alignment but no t2s projector is available in distiller.projectors."
                )
            teacher_rep = projector(teacher_rep)
            if teacher_rep.size(-1) != student_rep.size(-1):
                raise ValueError(
                    f"Teacher projection did not match student dim ({teacher_rep.size(-1)} vs {student_rep.size(-1)})."
                )
            return teacher_rep, student_rep

        if self.projection_space == "teacher":
            projector = self._select_projector("s2t", modality)
            if projector is None:
                raise ValueError(
                    "Requested student->teacher alignment but no s2t projector is available in distiller.projectors."
                )
            student_rep = projector(student_rep)
            if teacher_rep.size(-1) != student_rep.size(-1):
                raise ValueError(
                    f"Student projection did not match teacher dim ({student_rep.size(-1)} vs {teacher_rep.size(-1)})."
                )
            return teacher_rep, student_rep

        raise ValueError(
            f"Invalid eos_projection_space='{self.projection_space}'. Use 'student' or 'teacher'."
        )


    def _attention_pool_with_eos(self, tokens: torch.Tensor, eos_rep: torch.Tensor):
        if tokens is None or tokens.size(0) == 0:
            return eos_rep
        scale = tokens.size(-1) ** -0.5
        scores = torch.matmul(tokens, eos_rep.unsqueeze(-1)).squeeze(-1) * scale
        weights = torch.softmax(scores, dim=0)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=0)

    def _sample_text_vision_pooled(self, hidden_state, eos_rep, num_text_tokens, attention_mask, num_vision_tokens):
        if num_vision_tokens > 0:
            text_hidden, vision_hidden = get_hidden_text_vision(
                hidden_state,
                num_text_tokens,
                num_vision_tokens,
                attention_mask,
            )
        else:
            text_hidden = get_hidden_text(hidden_state, num_text_tokens, attention_mask)
            vision_hidden = None

        pooled_text = self._attention_pool_with_eos(text_hidden, eos_rep)
        pooled_vision = self._attention_pool_with_eos(vision_hidden, eos_rep)
        return pooled_text, pooled_vision

    def _build_pooled_reps(self, reps, image_features, hidden_states, attention_mask, num_text_tokens):
        batch_size = reps.size(0)
        pooled_text_reps = []
        pooled_vision_reps = []

        cur_img_idx = 0
        for i in range(batch_size):
            has_image = (
                image_features is not None
                and cur_img_idx < len(image_features)
                and image_features[cur_img_idx] is not None
            )
            num_vision_tokens = image_features[cur_img_idx].size(0) if has_image else 0
            pooled_text, pooled_vision = self._sample_text_vision_pooled(
                hidden_states[-1][i],
                reps[i],
                num_text_tokens[i].item(),
                attention_mask[i],
                num_vision_tokens,
            )
            pooled_text_reps.append(pooled_text)
            pooled_vision_reps.append(pooled_vision)
            if has_image:
                cur_img_idx += 1

        return torch.stack(pooled_text_reps, dim=0), torch.stack(pooled_vision_reps, dim=0)

    def _kl_from_cosines(self, teacher_anchor: torch.Tensor, student_swap: torch.Tensor):
        teacher_prob = F.softmax(teacher_anchor / self.distiller.temperature, dim=0)
        student_prob = F.softmax(student_swap / self.distiller.temperature, dim=0)
        return F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")

    def _pair_eos_loss(self, teacher_text, teacher_vision, student_text, student_vision):
        aligned_teacher_text, aligned_student_text = self._align_modal_reps(teacher_text, student_text, modality="text")
        aligned_teacher_vision, aligned_student_vision = self._align_modal_reps(teacher_vision, student_vision, modality="vision")

        teacher_anchor = F.cosine_similarity(aligned_teacher_vision, aligned_teacher_text, dim=-1)
        student_change_vision = F.cosine_similarity(aligned_student_vision, aligned_teacher_text.detach(), dim=-1)
        student_change_text = F.cosine_similarity(aligned_teacher_vision.detach(), aligned_student_text, dim=-1)

        loss_change_vision = self._kl_from_cosines(teacher_anchor.detach(), student_change_vision)
        loss_change_text = self._kl_from_cosines(teacher_anchor.detach(), student_change_text)
        return 0.5 * (loss_change_vision + loss_change_text)

    def compute_effective_rank(self, hidden_state: torch.Tensor, eps: float = 1e-10):
        X = hidden_state.float()
        N = X.size(0)
        s = torch.linalg.svdvals(X) / torch.sqrt(torch.tensor(N, device=X.device, dtype=X.dtype))
        eigvals = s * s
        prob = eigvals.clamp(min=eps) / eigvals.sum()
        entropy = -(prob * torch.log(prob)).sum()
        return (torch.exp(entropy) / N).to(dtype=hidden_state.dtype)

    def _compute_er_loss(
        self,
        batch_size,
        student_image_features,
        teacher_image_features,
        student_hidden_states,
        teacher_hidden_states,
        student_attention_mask,
        teacher_attention_mask,
        num_student_text_tokens,
        num_teacher_text_tokens,
    ):
        cur_idx_img = 0
        loss_vision_er = 0.0
        loss_text_er = 0.0

        for i in range(batch_size):
            if student_image_features is not None and teacher_image_features is not None:
                has_image = (
                    cur_idx_img < len(student_image_features)
                    and cur_idx_img < len(teacher_image_features)
                    and student_image_features[cur_idx_img] is not None
                    and teacher_image_features[cur_idx_img] is not None
                )
            else:
                has_image = False

            if has_image:
                stu_feat = student_image_features[cur_idx_img]
                tea_feat = teacher_image_features[cur_idx_img]

                loss_vision_er += nn.SmoothL1Loss()(self.compute_effective_rank(stu_feat), self.compute_effective_rank(tea_feat))

                last_stu_text_hidden_state, _ = get_hidden_text_vision(
                    student_hidden_states[-1][i],
                    num_student_text_tokens[i].item(),
                    stu_feat.size(0),
                    student_attention_mask[i],
                )
                last_tea_text_hidden_state, _ = get_hidden_text_vision(
                    teacher_hidden_states[-1][i],
                    num_teacher_text_tokens[i].item(),
                    tea_feat.size(0),
                    teacher_attention_mask[i],
                )
                cur_idx_img += 1
            else:
                last_stu_text_hidden_state = get_hidden_text(
                    student_hidden_states[-1][i],
                    num_student_text_tokens[i].item(),
                    student_attention_mask[i],
                )
                last_tea_text_hidden_state = get_hidden_text(
                    teacher_hidden_states[-1][i],
                    num_teacher_text_tokens[i].item(),
                    teacher_attention_mask[i],
                )

            loss_text_er += nn.SmoothL1Loss()(
                self.compute_effective_rank(last_stu_text_hidden_state),
                self.compute_effective_rank(last_tea_text_hidden_state),
            )

        loss_vision_er = loss_vision_er / (cur_idx_img + 1e-8)
        loss_text_er = loss_text_er / (batch_size + 1e-8)
        return 0.5 * (loss_vision_er + loss_text_er)

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

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]
        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]

        batch_size = student_qry_input["input_ids"].size(0)

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, _, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, _, teacher_pos_hidden_states = teacher_pos_output

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, _, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, _, student_pos_hidden_states = student_pos_output

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
        contrastive_loss = self.loss_fn(scores / self.distiller.temperature, target)

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

        student_qry_text, student_qry_vision = self._build_pooled_reps(
            student_qry_reps, student_qry_image_features, student_qry_hidden_states, student_qry_input["attention_mask"], num_student_text_qry_tokens
        )
        # student_pos_text, student_pos_vision = self._build_pooled_reps(
        #     student_pos_reps, student_pos_image_features, student_pos_hidden_states, student_pos_input["attention_mask"], num_student_text_pos_tokens
        # )
        teacher_qry_text, teacher_qry_vision = self._build_pooled_reps(
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_hidden_states, teacher_qry_input["attention_mask"], num_teacher_text_qry_tokens
        )
        # teacher_pos_text, teacher_pos_vision = self._build_pooled_reps(
        #     teacher_pos_reps, teacher_pos_image_features, teacher_pos_hidden_states, teacher_pos_input["attention_mask"], num_teacher_text_pos_tokens
        # )

        eos_loss = 1.0 * (
            self._pair_eos_loss(teacher_qry_text, teacher_qry_vision, student_qry_text, student_qry_vision)
            #+ self._pair_eos_loss(teacher_pos_text, teacher_pos_vision, student_pos_text, student_pos_vision)
        )

        er_qry_loss = self._compute_er_loss(
            batch_size,
            student_qry_image_features,
            teacher_qry_image_features,
            student_qry_hidden_states,
            teacher_qry_hidden_states,
            student_qry_input["attention_mask"],
            teacher_qry_input["attention_mask"],
            num_student_text_qry_tokens,
            num_teacher_text_qry_tokens,
        )
        er_pos_loss = self._compute_er_loss(
            batch_size,
            student_pos_image_features,
            teacher_pos_image_features,
            student_pos_hidden_states,
            teacher_pos_hidden_states,
            student_pos_input["attention_mask"],
            teacher_pos_input["attention_mask"],
            num_student_text_pos_tokens,
            num_teacher_text_pos_tokens,
        )
        er_loss = 0.5 * (er_qry_loss + er_pos_loss)

        weighted_kd_loss = self.eos_kd_weight * eos_loss + self.er_kd_weight * er_loss

        projector_graph_anchor = self._touch_all_projectors(contrastive_loss)
        loss = contrastive_loss + weighted_kd_loss + projector_graph_anchor

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": weighted_kd_loss,
            "eos_kd_loss": eos_loss,
            "er_kd_loss": er_loss,
        }
