import torch
import torch.nn as nn

class L2EmbedPenalty(nn.Module):
    def __init__(self, lamb):
        super(L2EmbedPenalty, self).__init__()
        self.lamb = lamb
        
    def forward(self, weight):
        penalty = torch.mean(torch.sum(torch.diff(weight[1:], dim = 0)**2, dim = -1))
        return self.lamb * penalty

