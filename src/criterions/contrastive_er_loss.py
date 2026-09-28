import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

class ContrastiveERLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveERLoss, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight
    
    def compute_effective_rank(
        self,
        hidden_state: torch.Tensor, # [N, D]
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Tính toán Effective Rank chuẩn theo bài báo Diff-eRank.
        Sử dụng SVD trên dữ liệu đã được chuẩn hóa để ổn định hơn.
        """
        X = hidden_state.float() 

        N = X.size(0)
        s = torch.linalg.svdvals(X) / torch.sqrt(torch.tensor(N))

        eigvals = s * s 

        prob = eigvals.clamp(min=eps) / eigvals.sum()
        
        entropy = -(prob * torch.log(prob)).sum()
        
        effective_rank = torch.exp(entropy) / N

        return effective_rank.to(dtype=hidden_state.dtype)
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, model_wrapper, input_data):
        model = model_wrapper.model
        input_qry = input_data['qry']
        input_pos = input_data['pos']

        qry_reps, _, _, _ = model.encode_input(input_qry)
        pos_reps, _, _, _ = model.encode_input(input_pos)

        if self.world_size > 1:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_pos_reps = self._dist_gather_tensor(pos_reps)
        else:
            all_qry_reps = qry_reps
            all_pos_reps = pos_reps
            
        scores = model.compute_similarity(all_qry_reps, all_pos_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        er_qry = self.compute_effective_rank(all_qry_reps)
        er_pos = self.compute_effective_rank(all_pos_reps)

        ER_MIN = 0.7 # 0.7
        er_reg = (
            F.relu(ER_MIN - er_qry) +
            F.relu(ER_MIN - er_pos)
        ) / 2.0

        total_loss = contrastive_loss + self.kd_weight * er_reg

        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': er_reg
        }
        