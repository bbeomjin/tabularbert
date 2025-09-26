import torch.nn as nn
    
class Accuracy(nn.Module):
    def __init__(self, ignore_index: int=-100):
        super(Accuracy, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, preds, targets):
        return (preds.argmax(dim = -1) == targets)[targets != self.ignore_index].float().mean()
    
       
       
class ClassificationError(nn.Module):
    def __init__(self, ignore_index: int=-100):
        super(ClassificationError, self).__init__()
        self.ignore_index = ignore_index
        self.accuracy = Accuracy(ignore_index)
        
    def forward(self, preds, targets):
        return 1 - self.accuracy(preds, targets)

    

class RMSE(nn.Module):
    def __init__(self, weight: float=1.0):
        super(RMSE, self).__init__()
        self.weight = weight
        
    def forward(self, preds, targets):
        return self.weight * (preds - targets).pow(2).mean().sqrt()



class MAE(nn.Module):
    def __init__(self, weight: float=1.0):
        super(MAE, self).__init__()
        self.weight = weight
        
    def forward(self, preds, targets):
        return self.weight * (preds - targets).abs().mean()
