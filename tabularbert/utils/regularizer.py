import torch
import torch.nn as nn

class SquaredL2Penalty(nn.Module):
    def __init__(self, lamb):
        super(SquaredL2Penalty, self).__init__()
        self.lamb = lamb
    
    def forward(self, weight):
        penalty = torch.mean(torch.sum(torch.diff(weight[1:], dim = 0)**2, dim = -1))
        return self.lamb * penalty

class L2Penalty(nn.Module):
    def __init__(self, lamb, tol: float = 1e-12):
        super(L2Penalty, self).__init__()
        self.lamb = lamb
        self.tol = tol
        
    def forward(self, weight):
        penalty = torch.mean(torch.sqrt(torch.sum(torch.diff(weight[1:], dim = 0)**2, dim = -1) + self.tol))
        return self.lamb * penalty

