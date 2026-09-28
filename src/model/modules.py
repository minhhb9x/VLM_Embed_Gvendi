import torch
import torch.nn as nn

class ModalityGatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, tokens, eps=1e-6):
        """
        tokens: [B, N, D]
        """
        g = torch.sigmoid(self.gate(tokens))  # [B, N, 1]
        pooled = (g * tokens).sum(dim=1) / (g.sum(dim=1) + eps)
        return pooled, g