import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
    
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

        return {
            'loss': contrastive_loss,
            'contrastive_loss': contrastive_loss,
        }
        