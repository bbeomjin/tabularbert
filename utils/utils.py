import torch
import os
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Install with: pip install wandb")

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



class DualLogger:
    """
    Unified logger that supports both TensorBoard and Weights & Biases (WandB).
    
    This logger automatically logs metrics to both platforms when available,
    providing comprehensive experiment tracking and visualization capabilities.
    
    Args:
        log_dir (str): Directory for TensorBoard logs
        project_name (str, optional): WandB project name. Default: "tabular-bert"
        experiment_name (str, optional): Experiment name for WandB run
        config (dict, optional): Configuration dictionary to log
        use_wandb (bool): Whether to use WandB logging. Default: True
    """
    
    def __init__(self, 
                 log_dir: str,
                 project_name: str="tabular-bert",
                 experiment_name: str=None,
                 config: dict=None,
                 use_wandb: bool=True):
        # Initialize TensorBoard logger
        self.tb_logger = SummaryWriter(log_dir)
        
        # Initialize WandB logger if available and requested
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                dir=os.path.dirname(log_dir)
            )
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value to both TensorBoard and WandB."""
        # Log to TensorBoard
        self.tb_logger.add_scalar(tag, value, step)
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars to both platforms."""
        # Log to TensorBoard
        self.tb_logger.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Log to WandB (flatten the nested structure)
        if self.use_wandb:
            wandb_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            wandb.log(wandb_dict, step=step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard (WandB histograms handled differently)."""
        self.tb_logger.add_histogram(tag, values, step)
        
        if self.use_wandb:
            # WandB handles histograms differently - log as histogram object
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_model_graph(self, model, input_to_model):
        """Log model graph to TensorBoard."""
        try:
            self.tb_logger.add_graph(model, input_to_model)
        except Exception as e:
            warnings.warn(f"Could not log model graph: {e}")
    
    def close(self):
        """Close both loggers."""
        self.tb_logger.close()
        if self.use_wandb:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()