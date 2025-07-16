import torch
import os
class CheckPoint:
    def __init__(self, save_path: str, max: bool):
        
        self.max = max
        self.loss = None
        save_path = os.path.expanduser(save_path)
        if os.path.exists(save_path) is not True:
            os.mkdir(save_path)
        
        self.save_path = os.path.join(save_path, 'model_checkpoint.pt')
        
    def __call__(self, x, model, epoch):
        if self.loss is None:
            self.loss = x
            torch.save(model, self.save_path.format(epoch = epoch))
        else:
            if self.max:
                if x > self.loss:
                    self.loss = x
                    torch.save(model, self.save_path.format(epoch = epoch))
            else:
                if x < self.loss:
                    self.loss = x
                    torch.save(model, self.save_path.format(epoch = epoch))

def make_save_dir(dir, type):
    folder_name = os.path.join(type, 'version')
    v = 0
    while True:
        path = os.path.join(dir, folder_name + str(v))
        if os.path.exists(path) is not True:
            os.mkdir(path)
            break
        else:
            v += 1
    return path
