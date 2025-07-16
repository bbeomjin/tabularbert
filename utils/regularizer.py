import torch
import torch.nn as nn

class L2EmbedPenalty(nn.Module):
    def __init__(self, lamb):
        super(L2EmbedPenalty, self).__init__()
        self.lamb = lamb
        
    def forward(self, weight):
        pen = torch.sum(torch.diff(weight, dim = 0)[1:]**2)
        return self.lamb * pen
        

class L2EmbedPenaltyStop(nn.Module):
    def __init__(self, lamb):
        super(L2EmbedPenaltyStop, self).__init__()
        self.lamb = lamb
        
    def forward(self, weight):
        pen = torch.sum((weight[1:-1] - weight[2:].detach())**2)
        return self.lamb * pen

        
# labels = torch.tensor([[1, 4, 2],
#                       [2, 0, 3]])

# embeddings = nn.Parameter(torch.randn(2, 3, 10))
# idx = torch.argsort(labels)

# embeddings[idx.unsqueeze(0)]
# idx.unsqueeze(0).shape
# torch.gather(embeddings, 2, idx.unsqueeze(-1))

# idx
# embeddings[torch.arange(embeddings.size(0)).unsqueeze(-1), idx]
