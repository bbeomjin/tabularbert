import functools
import json
import os
import warnings
import pathlib
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from model.embedding import NumEmbedding
from model.BERT import BERT, Classifier, Regressor
from utils.type import ArrayLike
from utils.utils import DualLogger, CheckPoint, make_save_dir
from utils.data import SSLDataset, QuantileDiscretize
from utils.criterion import TabularMSE, TabularWasserstein
from utils.regularizer import L2EmbedPenalty
from utils.scheduler import WarmupCosineLR


class TabularBERT(nn.Module):
    """
    TabularBERT: A BERT-based model for tabular data analysis.
    
    This model applies BERT transformer architecture to tabular data by discretizing
    continuous values into bins and treating them as tokens. 
    
    Architecture:
    1. NumEmbedding: Converts bin indices to embeddings with positional information
    2. BERT: Transformer encoder for learning contextual representations
    3. Classifier: Multi-task classification head for sequence-to-sequence predictions
    4. Regressor: Multi-task regression head for sequence-to-value predictions
    
    Key Features:
    - Handles mixed-type tabular data (categorical and continuous)
    - Multi-task learning with both classification and regression outputs
    - Position-aware embeddings for tabular structure
    - Self-attention mechanism for feature interactions
    
    Args:
        encoding_info (Dict[str, Dict[str, int]]): Nested dictionary containing encoding
                                                   information for each variable/column
        max_len (int): Maximum number of bins across all variables
        max_position (int): Maximum number of variables/columns in the dataset
        embedding_dim (int): Dimension of embedding vectors. Default: 256
        n_layers (int): Number of transformer encoder layers. Default: 4
        n_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout probability for regularization. Default: 0.1
    
    Example:
        >>> encoding_info = {'var1': 10, 'var2': 5}
        >>> model = TabularBERT(encoding_info, max_len=10, max_position=2)
        >>> bin_ids = torch.randint(1, 11, (32, 2))  # batch_size=32, 2 variables
        >>> reg_outputs, cls_outputs = model(bin_ids)
    """
    
    def __init__(self,
                 encoding_info: Dict[str, int],
                 max_len: int,
                 max_position: int,
                 embedding_dim: int=1024,
                 n_layers: int=3,
                 n_heads: int=8,
                 dropout: float=0.3) -> None:
        super(TabularBERT, self).__init__()
        
        # Store configuration
        self.encoding_info = encoding_info
        self.max_len = max_len
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Validate inputs
        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})")
        
        # Initialize model components
        self.embedding = NumEmbedding(
            max_len=max_len,
            max_position=max_position,
            embedding_dim=embedding_dim,
            mask_idx=0
        )
        
        self.bert = BERT(
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.classifier = Classifier(embedding_dim=embedding_dim,
                                     encoding_info=encoding_info)
        self.regressor = Regressor(embedding_dim=embedding_dim,
                                   encoding_info=encoding_info)
        
    def forward(self, bin_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the TabularBERT model.
        
        Args:
            bin_ids (torch.Tensor): Input tensor of shape (batch_size, num_variables)
                                   containing bin indices for each variable
        
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
                - reg_outputs: List of regression outputs (one tensor per variable)
                - cls_outputs: List of classification outputs (one tensor per variable)
        """
        # Embedding layer: bin_ids -> embeddings with positional information
        embeddings = self.embedding(bin_ids)
        
        # BERT encoder: learn contextual representations
        contextualized_embeddings = self.bert(embeddings)
        
        # Prediction heads: generate task-specific outputs
        cls_outputs = self.classifier(contextualized_embeddings)
        reg_outputs = self.regressor(contextualized_embeddings)
        
        return cls_outputs, reg_outputs
    
    def get_embeddings(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract contextualized embeddings without prediction heads.
        
        Useful for feature extraction or transfer learning scenarios.
        
        Args:
            bin_ids (torch.Tensor): Input tensor of shape (batch_size, num_variables)
        
        Returns:
            torch.Tensor: Contextualized embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        embeddings = self.embedding(bin_ids)
        return self.bert(embeddings)
    

class TabularBERTTrainer(nn.Module):
    """
    TabularBERTTrainer: A comprehensive training framework for TabularBERT models.
    
    This class manages the complete training workflow for tabular data analysis,
    including both self-supervised pretraining and supervised fine-tuning phases.
    It handles data preprocessing, model initialization, and training orchestration.
    
    Training Phases:
    1. Self-Supervised Pretraining: Uses masked language modeling on tabular data
       to learn meaningful representations without labels
    2. Supervised Fine-tuning: Adapts the pretrained model for specific downstream
       tasks (classification/regression) using labeled data
    
    Key Features:
    - Automated data preprocessing and binning
    - Self-supervised pretraining with masking strategies
    - Multi-task fine-tuning for classification and regression
    - Flexible encoding schemes for different data types
    
    Args:
        x (ArrayLike): Training data array/matrix
        num_bins (int): Number of bins for discretization
        encoding_info Dict[str, int]: 
                      Encoding configuration for variables. Can be:
                      - Dict mapping variable names to number of bins
                      - Omitted variables are discretized into the default number of bins (num_bins)
                      - None: All variables are discretized into the default number of bins (num_bins)
        device (torch.device): Device for computation. Default: CPU
        valid_x (ArrayLike, optional): Validation data for monitoring training
    
    Example:
        >>> import torch
        >>> import numpy as np
        >>> 
        >>> # Prepare data
        >>> x_train = np.random.randn(1000, 5)  # 1000 samples, 5 features
        >>> encoding_info = {'var1': 20, 'var5': 35}
        >>> 
        >>> # Initialize trainer
        >>> trainer = TabularBERTTrainer(
        ...     x=x_train,
        ...     num_bins=50,
        ...     encoding_info=encoding_info,
        ...     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ... )
        >>> 
        >>> # Run pretraining
        >>> trainer.pretrain(epochs=100)
        >>> 
        >>> # Fine-tune for downstream task
        >>> trainer.finetune(y_train, epochs=50)
    """
    
    def __init__(
        self,
        x: ArrayLike,
        num_bins: int=50,
        encoding_info: Dict[str, int]=None,
        device: torch.device=torch.device('cpu'),
        valid_x: ArrayLike=None
    ) -> None:
        super(TabularBERTTrainer, self).__init__()
        
        # Store input data and configuration
        self.x = x
        self.valid_x = valid_x
        self.device = device
        
        # Handle encoding_info: ensure it's not empty dict
        self.encoding_info = encoding_info if encoding_info else None
        
        # Initialize discretizer and process data
        self.discretizer = QuantileDiscretize(
            num_bins=num_bins, 
            encoding_info=self.encoding_info
        )
        self.discretizer.fit(x)
        
        # Discretize training data
        self.bin_ids = self.discretizer.discretize(x)
        
        # Discretize validation data if provided
        self.valid_bin_ids = (
            self.discretizer.discretize(valid_x) 
            if valid_x is not None else None
        )
        
        # Initialize model components (will be set later)
        self.model = None
        self.optimizer = None
    
    def setup_directories_and_logging(self, 
                                      project_name: str="tabular-bert",
                                      experiment_name: str=None,
                                      use_wandb: bool=True) -> None:
        """
        Setup save directories and logging infrastructure.
        
        Creates necessary directories for model checkpoints and initializes
        dual logging (TensorBoard + WandB) for comprehensive experiment tracking.
        
        Args:
            project_name (str): WandB project name. Default: "tabular-bert"
            experiment_name (str, optional): Experiment name for WandB run
            use_wandb (bool): Whether to use WandB logging. Default: True
        """
        # Get project root directory
        project_root = os.path.dirname(pathlib.Path(__file__).parent)
        
        # Create save directory for pretraining
        self.save_dir = make_save_dir(project_root, 'pretraining')
        
        # Initialize configuration dictionary for comprehensive tracking
        self.config = {
            # Data configuration
            'data': {
                'num_bins': self.discretizer.num_bins,
                'encoding_info': self.discretizer.encoding_info,
                'data_shape': self.x.shape if hasattr(self.x, 'shape') else None,
                'has_validation': self.valid_x is not None,
                'validation_shape': self.valid_x.shape if self.valid_x is not None else None
            },
            # System configuration
            'system': {
                'device': str(self.device)
            },
            # Model configuration (will be updated in set_mlm)
            'model': {},
            # Optimizer configuration (will be updated in set_optimizer)
            'optimizer': {},
            # Training configuration (will be updated in pretrain)
            'training': {}
        }
        
        # Save initial configuration as JSON
        self._save_config()
        
        # Initialize dual logger (TensorBoard + WandB)
        log_dir = os.path.join(self.save_dir, 'log')
        self.logger = DualLogger(
            log_dir=log_dir,
            project_name=project_name,
            experiment_name=experiment_name,
            config=self.config,
            use_wandb=use_wandb
        )
    
    def _save_config(self) -> None:
        """
        Save the current configuration to a JSON file.
        
        Creates a comprehensive configuration file that includes all model,
        optimizer, and training parameters for reproducibility.
        """
        config_path = os.path.join(self.save_dir, 'config.json')
        
        # Create a serializable copy of the config
        serializable_config = self._make_serializable(self.config.copy())
        
        try:
            with open(config_path, 'w') as f:
                json.dump(serializable_config, f, indent=4, sort_keys=True)
        except Exception as e:
            warnings.warn(f"Could not save config to {config_path}: {e}")
    
    def _make_serializable(self, obj):
        """
        Convert objects to JSON-serializable format.
        
        Handles numpy arrays, torch tensors, and other non-serializable objects.
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'shape'):  # numpy arrays, torch tensors
            return list(obj) if obj.size <= 100 else f"<{type(obj).__name__} shape={obj.shape}>"
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        else:
            return obj

    def set_mlm(self, 
                embedding_dim: int=1024,
                n_layers: int=3,
                n_heads: int=8,
                dropout: float=0.3) -> None:
        """
        Initialize the TabularBERT model for masked language modeling.
        
        Args:
            embedding_dim (int): Dimension of embedding vectors. Default: 1024
            n_layers (int): Number of transformer encoder layers. Default: 3
            n_heads (int): Number of attention heads. Default: 8
            dropout (float): Dropout probability for regularization. Default: 0.3
        """
        max_len = max([v for v in self.discretizer.encoding_info.values()])
        max_position = len(self.discretizer.encoding_info)
        
        # Update model configuration
        self.config['model'] = {
            'architecture': 'TabularBERT',
            'embedding_dim': embedding_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'dropout': dropout,
            'max_len': max_len,
            'max_position': max_position,
            'total_parameters': None  # Will be updated after model creation
        }
        
        # Create model
        self.model = TabularBERT(
            encoding_info=self.discretizer.encoding_info,
            max_len=max_len,
            max_position=max_position,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        ).to(self.device)
        
        # Update total parameters count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config['model']['total_parameters'] = total_params
        self.config['model']['trainable_parameters'] = trainable_params
        
        # Save updated configuration
        self._save_config()
    
    def set_optimizer(self,
                      lr: float=1e-4,
                      weight_decay: float=0.001,
                      betas: Tuple[float, float]=(0.9, 0.999)
                      ) -> None:
        """
        Configure the optimizer for training.
        
        Args:
            lr (float): Learning rate. Default: 1e-4
            weight_decay (float): Weight decay for regularization. Default: 0.001
            betas (Tuple[float, float]): Adam beta parameters. Default: (0.9, 0.999)
        """
        # Update optimizer configuration
        self.config['optimizer'] = {
            'type': 'AdamW',
            'lr': lr,
            'weight_decay': weight_decay,
            'betas': betas
        }
        
        # Create optimizer partial function
        self.optimizer = functools.partial(
            torch.optim.AdamW, 
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
        
        # Save updated configuration
        self._save_config()
    
    def pretrain(self, 
              epochs: int=1000,
              batch_size: int=256,
              lamb: float=0.5,
              mask_token_prob: float=0.1,
              random_token_prob: float=0.1,
              unchanged_token_prob: float=0.1,
              ignore_index: int=-100,
              num_workers: int=0,
              ) -> None:
        """
        Run self-supervised pretraining using masked language modeling.
        
        Args:
            epochs (int): Number of training epochs. Default: 1000
            batch_size (int): Batch size for training. Default: 256
            lamb (float): Regularization parameter. Default: 0.5
            mask_token_prob (float): Probability of replacing tokens with [MASK]. Default: 0.1
            random_token_prob (float): Probability of replacing tokens with random values. Default: 0.1
            unchanged_token_prob (float): Probability of keeping original tokens unchanged. Default: 0.1
            ignore_index (int): Index to ignore in loss calculation. Default: -100
        """
        # Update training configuration
        self.config['training'] = {
            'phase': 'pretraining',
            'epochs': epochs,
            'batch_size': batch_size,
            'regularization_lambda': lamb,
            'masking': {
                'mask_token_prob': mask_token_prob,
                'random_token_prob': random_token_prob,
                'unchanged_token_prob': unchanged_token_prob
            }
        }
        
        # Save updated configuration before training starts
        self._save_config()
        
        train_dataset = SSLDataset(
            x = self.x,
            bin_ids=self.bin_ids,
            encoding_info=self.discretizer.encoding_info,
            mask_token_prob=mask_token_prob,
            random_token_prob=random_token_prob,
            unchanged_token_prob=unchanged_token_prob,
            ignore_index=ignore_index,
            num_workers=num_workers
        )
        trainloader = DataLoader(train_dataset, 
                                 batch_size = batch_size,
                                 shuffle = True, 
                                 drop_last = True)

        if self.valid_x is not None:
            valid_dataset = SSLDataset(
                x = self.valid_x,
                bin_ids=self.valid_bin_ids,
                encoding_info=self.discretizer.encoding_info,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                unchanged_token_prob=unchanged_token_prob,
                ignore_index=ignore_index,
                num_workers=num_workers
            )

            validloader = DataLoader(valid_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     drop_last=False)
            
        
        if self.model is None:
            warnings.warn(
                """
                TabularBERT Model Auto-Configuration\n
                ======================================================================\n
                No model configuration detected. Initializing with optimized defaults:\n\n
                Architecture: TabularBERT Transformer\n
                Embedding Dimension: 1024\n
                Transformer Layers: 3\n
                Attention Heads: 8\n
                Dropout Rate: 0.3\n\n
                Tip: Use trainer.set_mlm() to customize architecture before training.\n
                ======================================================================
                """,
                UserWarning
            )
            self.set_mlm()
        
        # Define loss functions
        mse_loss = TabularMSE(self.discretizer.encoding_info)
        wasserstein_loss = TabularWasserstein(self.discretizer.encoding_info, ignore_index=ignore_index)
        
        # Define regularizer
        embed_penalty = L2EmbedPenalty(lamb)
        
        # Define metric
        checkpoint = CheckPoint(self.save_dir, max=False)
        
        # Define optimizer
        if self.optimizer is None:
            warnings.warn(
                """
                TabularBERT Optimizer Auto-Configuration\n
                ======================================================================\n
                No optimizer configuration detected. Initializing with optimized defaults:\n\n
                Optimizer: AdamW\n
                Learning Rate: 1e-4\n
                Weight Decay: 0.01\n
                Beta Parameters: (0.9, 0.999)\n\n
                Tip: Use trainer.set_optimizer() to customize optimizer before training.\n
                ======================================================================
                """,
                UserWarning
            )
            self.set_optimizer()
            
        self.optimizer = self.optimizer(params=self.model.parameters())
        total_steps = epochs * len(trainloader)
        self.scheduler = WarmupCosineLR(self.optimizer, 
                                   warmup_epochs=1,
                                   max_epochs=total_steps, 
                                   eta_min=1e-5,
                                   warmup_start_lr=1e-5)
        
        # Training loop with progress tracking
        global_step = 0
        best_loss = float('inf')
        
        print(f"\n Starting TabularBERT Pretraining")
        print(f"{'='*60}")
        print(f"Dataset: {len(trainloader)} batches ({len(trainloader.dataset)} samples)")
        print(f"Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"{'='*60}\n")
        
        for epoch in tqdm.tqdm(range(epochs), desc='Pretraining Progress'):
            # Training phase
            train_metrics = self._run_training_epoch(
                trainloader, mse_loss, wasserstein_loss, embed_penalty, global_step
            )
            global_step = train_metrics['global_step']
            
            # Validation phase (if validation data available)
            if self.valid_x is not None:
                valid_metrics = self._run_validation_epoch(
                    validloader, mse_loss, wasserstein_loss, embed_penalty, epoch
                )
                
                # Model checkpointing based on validation loss
                current_loss = valid_metrics['avg_loss']
                checkpoint(current_loss, self.model, epoch)
                
                # Elegant progress reporting
                self._log_epoch_progress(train_metrics['avg_loss'], valid_metrics['avg_loss'])
            else:
                # No validation data - checkpoint on training loss
                current_loss = train_metrics['avg_loss']
                checkpoint(current_loss, self.model, epoch)
                
                # Training-only progress reporting
                self._log_epoch_progress(train_metrics['avg_loss'])
        
        print(f"\n Pretraining completed!")
        print(f"Model saved to: {self.save_dir}")
    
    def _run_training_epoch(self, trainloader, mse_loss, wasserstein_loss, 
                           embed_penalty, global_step):
        """
        Execute one training epoch with efficient batch processing.
        
        Returns:
            dict: Training metrics including total loss and global step
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(trainloader)
        
        for batch_idx, (bin_ids, labels, tabular_x) in enumerate(trainloader):
            bin_ids = bin_ids.to(self.device)
            labels = labels.to(self.device)
            tabular_x = tabular_x.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            cls_predictions, reg_predictions = self.model(bin_ids)
            
            # Compute losses
            wasserstein_loss_val = wasserstein_loss(cls_predictions, labels)
            mse_loss_val = mse_loss(reg_predictions, tabular_x)
            regularization_loss = embed_penalty(self.model.embedding.bin_embedding.weight)
            
            # Combined loss
            total_batch_loss = wasserstein_loss_val + mse_loss_val + regularization_loss
            
            # Backward pass and optimization
            total_batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics tracking
            total_loss += total_batch_loss.item()
            
            # Log detailed metrics
            self.logger.log_scalar('Loss/Train/Total', total_batch_loss.item(), global_step)
            self.logger.log_scalar('Loss/Train/Wasserstein', wasserstein_loss_val.item(), global_step)
            self.logger.log_scalar('Loss/Train/MSE', mse_loss_val.item(), global_step)
            self.logger.log_scalar('Loss/Train/Regularization', regularization_loss.item(), global_step)
            
            global_step += 1
        
        return {
            'avg_loss': total_loss / num_batches,
            'total_loss': total_loss,
            'global_step': global_step
        }
    
    def _run_validation_epoch(self, validloader, mse_loss, wasserstein_loss, 
                             embed_penalty, epoch):
        """
        Execute one validation epoch with no gradient computation.
        
        Returns:
            dict: Validation metrics including average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(validloader)
        
        with torch.no_grad():
            for batch_idx, (bin_ids, labels, tabular_x) in enumerate(validloader):
                bin_ids = bin_ids.to(self.device)
                labels = labels.to(self.device)
                tabular_x = tabular_x.to(self.device)
                
                # Forward pass only
                cls_predictions, reg_predictions = self.model(bin_ids)
                
                # Compute losses
                wasserstein_loss_val = wasserstein_loss(cls_predictions, labels)
                mse_loss_val = mse_loss(reg_predictions, tabular_x)
                regularization_loss = embed_penalty(self.model.embedding.bin_embedding.weight)
                
                # Combined loss
                total_batch_loss = wasserstein_loss_val + mse_loss_val + regularization_loss
                total_loss += total_batch_loss.item()
                
                # Log validation metrics
                self.logger.log_scalar('Loss/Valid/Total', total_batch_loss.item(), epoch)
                self.logger.log_scalar('Loss/Valid/Wasserstein', wasserstein_loss_val.item(), epoch)
                self.logger.log_scalar('Loss/Valid/MSE', mse_loss_val.item(), epoch)
        
        return {
            'avg_loss': total_loss / num_batches,
            'total_loss': total_loss
        }
    
    def _log_epoch_progress(self, train_loss, valid_loss=None):
        """
        Log epoch loss information (progress bar handled by tqdm).
        """
        if valid_loss is not None:
            loss_info = f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
        else:
            loss_info = f"Train Loss: {train_loss:.6f}"
        
        print(f"  {loss_info}")    
                    
                    
