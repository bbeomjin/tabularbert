import torch
import torch.nn as nn
import math

class SquaredL2Penalty(nn.Module):
    def __init__(self, lamb):
        super(SquaredL2Penalty, self).__init__()
        self.lamb = lamb
    
    def forward(self, weight):
        diff = torch.diff(weight[1:], dim=0)
        penalty = (diff.pow(2).sum(dim=-1) / diff.size(-1)).mean()
        # penalty = diff.pow(2).sum(dim=-1).mean()
        return self.lamb * penalty

class L2Penalty(nn.Module):
    def __init__(self, lamb, tol: float = 1e-12):
        super(L2Penalty, self).__init__()
        self.lamb = lamb
        self.tol = tol
        
    def forward(self, weight):
        diff = torch.diff(weight[1:], dim=0)
        penalty = (torch.sqrt(diff.pow(2).sum(dim=-1) + self.tol) / math.sqrt(diff.size(-1))).mean()
        # penalty = (torch.sqrt(diff.pow(2).sum(dim=-1) + self.tol)).mean()
        return self.lamb * penalty

