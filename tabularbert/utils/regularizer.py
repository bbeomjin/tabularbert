import torch
import torch.nn as nn

class L2EmbedPenalty(nn.Module):
    def __init__(self, lamb):
        super(L2EmbedPenalty, self).__init__()
        self.lamb = lamb
        
    def forward(self, weight):
        # penalty = torch.mean(torch.sqrt(torch.sum(torch.diff(weight[1:], dim = 0)**2, dim = -1)))
        penalty = torch.mean(torch.sum(torch.diff(weight[1:], dim = 0)**2, dim = -1))
        return self.lamb * penalty



class ProximalL2(nn.Module):
    def __init__(self, lamb):
        super(ProximalL2, self).__init__()
        self.lamb = lamb
    
    def forward(self, weight, grad, eta: float):
        weight = weight.detach().clone()
        updated = weight.add(grad, alpha = -eta)
        weight = torch.concat([weight[2:], weight[-2:-1]], dim = 0)
        diff = updated[1:] - weight
        W2norm = torch.norm(diff, 2, dim = -1)
        shrinkage = torch.clamp(1 - (self.lamb * eta) / W2norm, min = 0)
        updated[1:] = shrinkage.view(-1, 1) * diff + weight
        return updated
