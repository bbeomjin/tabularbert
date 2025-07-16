import torch

class MLMAccuracy:
    def __init__(self, ignore_index = -100):
        self.ignore_index = ignore_index
        
    def __call__(self, preds, targets):        
        n = 0
        n_correct = 0
        for j, pred in enumerate(preds):
            mask_idx = targets[:, j] == self.ignore_index
            n_correct += (pred.argmax(dim = -1) == targets[:, j])[~mask_idx].sum()
            n += torch.sum(~mask_idx)
        return n_correct / n    
    
class Accuracy:
    def __init__(self, ignore_index = -100):
        self.ignore_index = ignore_index
        
    def __call__(self, preds, targets):
        return (preds.argmax(dim = -1) == targets)[targets != self.ignore_index].float().mean()