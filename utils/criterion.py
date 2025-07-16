import torch
import torch.nn as nn
import torch.nn.functional as F
class GMTCrossEntropy(nn.Module):
    def __init__(self, encoded_info, ignore_index = -100):
        super(GMTCrossEntropy, self).__init__()
        self.encoded_info = encoded_info
        self.ignore_index = ignore_index
        self.p = len(encoded_info)
        self.ce = nn.ModuleList()
        for _ in encoded_info:
            # ignore_index = info['L'] * (self.ignore_index - 1) + self.ignore_index - 1
            self.ce.append(nn.CrossEntropyLoss(ignore_index = ignore_index))
        
        
    def forward(self, preds, targets):
        # bin_target = targets[0]
        # subbin_target = targets[1]
        loss = .0
        for j, _ in enumerate(self.encoded_info):    
            # Labels start with 1
            # loss += self.ce[j](preds[j], info['L'] * (bin_target[:, j] - 1) + (subbin_target[:, j] - 1))
            loss += self.ce[j](preds[j], targets[:, j])
        return loss / (j + 1)

# class GMTWasserstein(nn.Module):
#     def __init__(self, encoded_info, ignore_index = -100):
#         super(GMTWasserstein, self).__init__()
#         self.encoded_info = encoded_info
#         self.ignore_index = ignore_index
#         self.p = len(encoded_info)   

#     def _wasserstein(self, pred, target, encoded_info):
#         softmax = F.softmax(pred, dim = -1)
#         valid_idx = target != self.ignore_index    
#         fill_value = encoded_info['K'] * encoded_info['L']
#         target = target.masked_fill(target == self.ignore_index, fill_value)
#         target = F.one_hot(target, fill_value + 1)[:, :-1]
#         # return torch.mean((torch.abs(softmax.cumsum(dim = -1) - target.cumsum(dim = -1))**2).sum(dim = -1)[valid_idx])
#         return torch.mean(((softmax.cumsum(dim = -1) - target.cumsum(dim = -1))**2).sum(dim = -1)[valid_idx]**(1/2))

#     def forward(self, preds, targets):
#         loss = .0
#         for j, k in enumerate(self.encoded_info):
#             loss += self._wasserstein(preds[j], targets[:, j], self.encoded_info[k])
#         return loss / (j + 1)
        
        
class GMTMSE(nn.Module):
    def __init__(self, encoded_info, ignore_index = -100):
        super(GMTMSE, self).__init__()
        self.encoded_info = encoded_info
        self.ignore_index = ignore_index
        self.p = len(encoded_info)   
        
    def _mse(self, pred, target, encoded_info):
        softmax = F.softmax(pred, dim = -1)
        valid_idx = target != self.ignore_index
        yhat = ((torch.arange(encoded_info['K'] * encoded_info['L'], device = target.device).float() + 1) * softmax).sum(dim = -1)
        return F.mse_loss(yhat[valid_idx], target[valid_idx].float() + 1)
        
    def forward(self, preds, targets):
        loss = .0
        for j, k in enumerate(self.encoded_info):
            loss += self._mse(pred = preds[j], target = targets[:, j], encoded_info = self.encoded_info[k])
        return loss / (j + 1)


class GMTWAS(nn.Module):
    def __init__(self, encoded_info, ignore_index = -100):
        super(GMTWAS, self).__init__()
        self.encoded_info = encoded_info
        self.ignore_index = ignore_index
        self.p = len(encoded_info)   
        
    def _was(self, pred, target, encoded_info):
        softmax = F.softmax(pred, dim = -1)
        valid_idx = target != self.ignore_index
        labels = torch.arange(encoded_info['K'] * encoded_info['L'], device = target.device).float()
        # return (torch.abs(target.unsqueeze(-1).float() - labels) * softmax).sum(dim = -1)[valid_idx].mean()
        return ((torch.abs(target.unsqueeze(-1).float() - labels) * softmax)**2).sum(dim = -1)[valid_idx].mean()
        
    def forward(self, preds, targets):
        loss = .0
        for j, k in enumerate(self.encoded_info):
            loss += self._was(pred = preds[j], target = targets[:, j], encoded_info = self.encoded_info[k])
        return loss / (j + 1)
    

