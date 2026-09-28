import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

from .utils import get_hidden_text_vision, get_hidden_text

class ContrastivePoolingLoss(nn.Module):
    def __init__(self, args):
        super(ContrastivePoolingLoss, self).__init__()
        self.args = args
        self.vision_weight = 0.6
        self.reg_weight = 1e-4
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, model_wrapper, input_data):
        model = model_wrapper.model
        tokenizer = model_wrapper.processor.tokenizer
        input_qry = input_data['qry']
        input_pos = input_data['pos']
        qry_output = model.encode_input(input_qry)
        pos_output = model.encode_input(input_pos)
        qry_reps, qry_image_features, _, qry_hidden_states = qry_output
        pos_reps, pos_image_features, _, pos_hidden_states = pos_output

        special_ids = torch.tensor(tokenizer.all_special_ids, device=input_qry['input_ids'].device)
        
        num_text_qry_tokens = (~torch.isin(input_qry['input_ids'], 
                                                   special_ids)).sum(dim=1)
        num_text_pos_tokens = (~torch.isin(input_pos['input_ids'], 
                                                   special_ids)).sum(dim=1)

        batch_size = qry_reps.size(0)

        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        z_qry_list = []
        z_pos_list = []
        for i in range(batch_size):
            z_qry = 0
            if qry_image_features is not None:
                num_vision_qry_tokens = qry_image_features[cur_idx_qry_img].size(0)
                last_text_state, last_vision_state = get_hidden_text_vision(
                    qry_hidden_states[-1][i], 
                    num_text_qry_tokens[i].item(), 
                    num_vision_qry_tokens, 
                    attention_mask=input_qry['attention_mask'][i]
                )
                cur_idx_qry_img += 1
                last_text_state = F.normalize(last_text_state, p=2, dim=-1)
                last_vision_state = F.normalize(last_vision_state, p=2, dim=-1)
                z_v_qry, _ = model.encoder.pool_v(last_vision_state.unsqueeze(0)) # [1, D]
                z_t_qry, _ = model.encoder.pool_t(last_text_state.unsqueeze(0)) # [1, D]
                z_qry = z_v_qry * self.vision_weight + z_t_qry * (1 - self.vision_weight)
            else:
                last_text_state = get_hidden_text(
                    qry_hidden_states[-1][i], 
                    num_text_qry_tokens[i].item(), 
                    attention_mask=input_qry['attention_mask'][i]
                )
                last_text_state = F.normalize(last_text_state, p=2, dim=-1)
                z_qry, _ = model.encoder.pool_t(last_text_state.unsqueeze(0)) # [1, D]

            z_qry_list.append(z_qry)

            if pos_image_features is not None:
                num_vision_pos_tokens = pos_image_features[cur_idx_pos_img].size(0)
                last_text_state, last_vision_state = get_hidden_text_vision(
                    pos_hidden_states[-1][i], 
                    num_text_pos_tokens[i].item(), 
                    num_vision_pos_tokens, 
                    attention_mask=input_pos['attention_mask'][i]
                )
                cur_idx_pos_img += 1
                last_text_state = F.normalize(last_text_state, p=2, dim=-1)
                last_vision_state = F.normalize(last_vision_state, p=2, dim=-1)
                z_v_pos, _ = model.encoder.pool_v(last_vision_state.unsqueeze(0)) # [1, D]
                z_t_pos, _ = model.encoder.pool_t(last_text_state.unsqueeze(0)) # [1, D]
                z_pos = z_v_pos * self.vision_weight + z_t_pos * (1 - self.vision_weight)
            else:
                last_text_state = get_hidden_text(
                    pos_hidden_states[-1][i], 
                    num_text_pos_tokens[i].item(), 
                    attention_mask=input_pos['attention_mask'][i]
                )
                last_text_state = F.normalize(last_text_state, p=2, dim=-1)
                z_pos, _ = model.encoder.pool_t(last_text_state.unsqueeze(0))# [1, D]
                
            z_pos_list.append(z_pos)

        z_qry = torch.cat(z_qry_list, dim=0)  # [B, D]
        z_pos = torch.cat(z_pos_list, dim=0)  # [B, D]

        z_qry = F.normalize(z_qry, p=2, dim=-1)
        z_pos = F.normalize(z_pos, p=2, dim=-1)

        scores = model.compute_similarity(z_qry, z_pos)
        scores = scores.view(z_qry.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (z_qry.size(0) // z_pos.size(0))

        gate_t = model.encoder.pool_t.gate
        gate_v = model.encoder.pool_v.gate

        reg = sum(w.abs().mean() for w in gate_t.parameters()) + \
                sum(w.abs().mean() for w in gate_v.parameters())

        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)
        
        total_loss = contrastive_loss + self.reg_weight * reg

        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
        }
        