from src.criterions.contrastive_kd_loss import ContrastiveKDLoss
from src.criterions.contrastive_loss import ContrastiveLoss
from src.criterions.contrastive_er_loss import ContrastiveERLoss
from src.criterions.contrastive_loss_2 import ContrastiveLoss2
from src.criterions.eigen_rank_align import ERAlign
from src.criterions.vision_RKD import VisionRKDLoss
from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .span_propose import SpanProposeCriterion
from .span_propose_attn import SpanProposeCriterionWeighted
from .span_propose_attn_only_phrase import SpanProposeCriterionWeightedOnlyPhrase
from .penultimate_mse_loss import PenultimateMSELoss
from .vision_encoder_kd_loss import VisionEncoderLoss
from .kl_cosine_loss import KLCosineLoss
from .compute_effective_rank import EffectiveRankLoss
from .contrastive_pooling_loss import ContrastivePoolingLoss
from .grad_aggregation import GradPoolingLoss
from .talas import Talas
from .talas_jepa import TalasJepa
from .similarity_matrix_distillation import SimilarityMatrixDistillationLoss
from .simcse_cka_kd_loss import SimcseCKA
from .span_propose_attn_llava_ov import SpanProposeCriterionWeightedLLavaOV
from .mse_kd_loss import MSEKDLoss

from .mse_sigreg_kd_loss import MSESigRegLoss
from .rkd_sigreg_kd_loss import RKDSigRegLoss
from .ckd_sigreg_kd_loss import CKDSigRegLoss
from .emo_sigreg_kd_loss import EMOSigRegLoss
from .em_sigreg_kd_loss import EMSigRegKDLoss
from .span_attn_sigreg import SpanSigregCriterionWeighted

criterion_list = {
    "contrastive": ContrastiveLoss,
    "contrastive_2": ContrastiveLoss2,
    "contrastive_er": ContrastiveERLoss,
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
    "em_kd_llava_ov": EMKDLLavaLoss,
    "span_propose": SpanProposeCriterion,
    "span_propose_attn": SpanProposeCriterionWeighted,
    "span_propose_attn_only_phrase": SpanProposeCriterionWeightedOnlyPhrase,

    "vision_rkd": VisionRKDLoss,
    "penultimate_mse": PenultimateMSELoss,
    "contrastive_kd": ContrastiveKDLoss,
    "vision_encoder_kd": VisionEncoderLoss,
    "kl_cosine_loss": KLCosineLoss,
    "effective_rank_loss": EffectiveRankLoss,
    "eigen_rank_align_loss": ERAlign,
    "contrastive_pooling_loss": ContrastivePoolingLoss,
    "grad_pooling": GradPoolingLoss,
    "talas": Talas,
    "talas_jepa": TalasJepa,
    "similarity_matrix_distillation": SimilarityMatrixDistillationLoss,
    "simcse_cka_loss": SimcseCKA,
    "span_propose_attn_llava_ov": SpanProposeCriterionWeightedLLavaOV,

    "mse_kd": MSEKDLoss, 
    "mse_sigreg_kd": MSESigRegLoss,
    "rkd_sigreg_kd": RKDSigRegLoss,
    "ckd_sigreg_kd": CKDSigRegLoss,
    "emo_sigreg_kd": EMOSigRegLoss,
    "em_sigreg_kd": EMSigRegKDLoss,
    "span_attn_sigreg_kd": SpanSigregCriterionWeighted,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)
