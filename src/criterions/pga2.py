import torch
import torch.nn.functional as F

from .pga import PGA
from .utils import extract_modality_tokens


class PGA2(PGA):
    """PGA with per-sample teacher representations loaded from disk."""

    @staticmethod
    def _teacher_reps(model_wrapper, batch, side, student_reps):
        caches = batch[f"teacher_{side}_caches"]
        encoded_dirs = batch["encoded_dir"]
        if caches is None:
            raise ValueError("PGA2 requires --caching_dir pointing to a teacher cache")
        if len(caches) != len(encoded_dirs) or len(caches) != len(student_reps):
            raise ValueError(f"PGA2: student and cached teacher {side} batch sizes differ")

        expected_dim = model_wrapper.teacher_hidden_dim
        reps = []
        for cache, encoded_dir in zip(caches, encoded_dirs):
            rep = cache.get("rep")
            if not torch.is_tensor(rep) or rep.shape != (expected_dim,):
                raise ValueError(
                    f"PGA2: {encoded_dir}/{side}.pt must contain a rep of shape ({expected_dim},)"
                )
            reps.append(rep.detach())
        reps = torch.stack(reps).to(device=student_reps.device, dtype=torch.float32)
        if model_wrapper.model_args.teacher_normalize:
            reps = F.normalize(reps, dim=-1)
        return reps

    def _scl_loss(self, model_wrapper, batch, student_outputs):
        student = model_wrapper.model
        processor = model_wrapper.get_processor()
        padding_side = getattr(student.encoder.config, "tokenizer_padding_side", "right")
        stu_img, tea_img, stu_txt, tea_txt = [], [], [], []

        for side, (_, image_features, attention, hidden_states) in student_outputs.items():
            samples = extract_modality_tokens(
                processor, batch[side], image_features, attention, hidden_states,
                student.model_backbone, padding_side=padding_side,
            )
            for sample, cache in zip(samples, batch[f"teacher_{side}_caches"]):
                teacher_img = cache.get("mean_last_img_token")
                teacher_txt = cache.get("mean_last_text_token")
                if teacher_img is None or teacher_txt is None:
                    continue
                if any(sample[modality].size(1) == 0 for modality in ("image", "text")):
                    continue

                for modality, buffer in (("image", stu_img), ("text", stu_txt)):
                    tokens = sample[modality][-1]
                    weights = sample[f"{modality}_attention"]
                    pooled = (
                        self.attention_pool(tokens, weights[-1])
                        if weights is not None else tokens.mean(dim=0)
                    )
                    buffer.append(pooled)
                tea_img.append(teacher_img)
                tea_txt.append(teacher_txt)

        if len(stu_img) < 2:
            return student_outputs["qry"][0].new_zeros(())

        sv, st = torch.stack(stu_img), torch.stack(stu_txt)
        img_proj = model_wrapper.projectors["t2s_img"]
        txt_proj = model_wrapper.projectors["t2s_txt"]
        tv = img_proj(torch.stack(tea_img).to(next(img_proj.parameters())))
        tt = txt_proj(torch.stack(tea_txt).to(next(txt_proj.parameters())))
        intra_loss = self.mi_loss(sv, tv) + self.mi_loss(st, tt)
        inter_loss = self.mi_loss(sv, tt) + self.mi_loss(st, tv)
        return intra_loss + inter_loss

    def forward(self, model_wrapper, batch):
        student = model_wrapper.model
        student_outputs = {side: student.encode_input(batch[side]) for side in ("qry", "pos")}
        student_qry, student_pos = (student_outputs[side][0] for side in ("qry", "pos"))
        teacher_qry = self._teacher_reps(model_wrapper, batch, "qry", student_qry)
        teacher_pos = self._teacher_reps(model_wrapper, batch, "pos", student_pos)

        if self.world_size > 1:
            student_qry = self._dist_gather_tensor(student_qry)
            student_pos = self._dist_gather_tensor(student_pos)
            teacher_qry = self._dist_gather_tensor(teacher_qry)
            teacher_pos = self._dist_gather_tensor(teacher_pos)

        scores = student.compute_similarity(student_qry, student_pos)
        scores = scores.view(student_qry.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device)
        contrastive_loss = self.loss_fn(scores / model_wrapper.temperature, target)

        mse_loss = contrastive_loss.new_zeros(())
        scl_loss = contrastive_loss.new_zeros(())
        # Enable t2s_img and t2s_txt in the projector config before uncommenting.
        # scl_loss = self._scl_loss(model_wrapper, batch, student_outputs)

        pga_loss = self.pga_loss(
            torch.cat((student_qry, student_pos), dim=0),
            torch.cat((teacher_qry, teacher_pos), dim=0),
        )
        kd_loss = mse_loss + 0.5 * scl_loss + pga_loss
        return {
            "loss": contrastive_loss + kd_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "kd_loss_rkd": pga_loss,
            "kd_mse_loss": mse_loss,
            "cross_modal_loss": scl_loss,
        }
