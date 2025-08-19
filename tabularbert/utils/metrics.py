import torch.nn as nn
    
class Accuracy(nn.Module):
    def __init__(self, ignore_index = -100):
        super(Accuracy, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, preds, targets):
        return (preds.argmax(dim = -1) == targets)[targets != self.ignore_index].float().mean()
    
    
    
class ClassificationError(nn.Module):
    def __init__(self, ignore_index = -100):
        super(ClassificationError, self).__init__()
        self.ignore_index = ignore_index
        self.accuracy = Accuracy(ignore_index)
        
    def forward(self, preds, targets):
        return 1 - self.accuracy(preds, targets)