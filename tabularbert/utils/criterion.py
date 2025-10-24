"""
Loss functions for TabularBERT training.

This module provides various loss functions optimized for tabular data learning,
including cross-entropy, MSE, and Wasserstein losses with proper handling of
masked tokens and multi-task learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class TabularCrossEntropy(nn.Module):
    """
    Multi-task cross-entropy loss for tabular data classification.
    
    Handles multiple features with different vocabulary sizes and supports
    masked token ignoring for self-supervised learning.
    
    Args:
        encoding_info (Dict[str, int]): Mapping of feature names to vocabulary sizes
        ignore_index (int): Index to ignore in loss calculation. Default: -100
    """
    
    def __init__(self, encoding_info: Dict[str, Dict[str, int]], ignore_index: int=-100) -> None:
        super(TabularCrossEntropy, self).__init__()
        self.encoding_info = encoding_info
        self.ignore_index = ignore_index
        self.compute_ids = [i for i, (_, v) in enumerate(encoding_info.items()) if 'num_categories' in v.keys()]

    def _compute_ce(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for a single feature.
        
        Args:
            prediction (torch.Tensor): Logits for the feature
            target (torch.Tensor): Target class indices
            
        Returns:
            torch.Tensor: Cross-entropy loss for valid (non-ignored) targets
        """
        return F.cross_entropy(prediction, target, ignore_index=self.ignore_index)
        
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-task cross-entropy loss.
        
        Args:
            predictions (List[torch.Tensor]): List of prediction tensors for each feature
            targets (torch.Tensor): Target tensor of shape (batch_size, num_features)
            
        Returns:
            torch.Tensor: Average cross-entropy loss across all features
        """
        
        if len(self.compute_ids) == 0:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        total_loss = 0.0
        for feature_idx in self.compute_ids:
            feature_loss = self._compute_ce(predictions[feature_idx], targets[:, feature_idx])
            total_loss += feature_loss
            
        return total_loss / len(self.compute_ids)


class TabularMSE(nn.Module):
    """
    Mean Squared Error loss for tabular data regression.
    
    Handles NaN values in targets for masked token scenarios and supports
    multi-task regression learning.
    
    Args:
        encoding_info (Dict[str, int]): Mapping of feature names to dimensions
    """
    
    def __init__(self, encoding_info: Dict[str, Dict[str, int]]) -> None:
        super(TabularMSE, self).__init__()
        self.encoding_info = encoding_info
        self.compute_ids = [i for i, (_, v) in enumerate(encoding_info.items()) if 'num_bins' in v.keys()]
        
    def _compute_mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE for a single feature, handling NaN values.
        
        Args:
            prediction (torch.Tensor): Predicted values
            target (torch.Tensor): Target values (may contain NaN)
            
        Returns:
            torch.Tensor: MSE loss for valid (non-NaN) targets
        """
        valid_mask = ~torch.isnan(target)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)
            
        return F.mse_loss(prediction[valid_mask], target[valid_mask])
    
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-task MSE loss.
        
        Args:
            predictions (List[torch.Tensor]): List of prediction tensors for each feature
            targets (torch.Tensor): Target tensor of shape (batch_size, num_features)
            
        Returns:
            torch.Tensor: Average MSE loss across all features
        """
        
        if len(self.compute_ids) == 0:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        total_loss = 0.0
        for feature_idx in self.compute_ids:
            feature_loss = self._compute_mse(
                predictions[feature_idx], 
                targets[:, feature_idx]
            )
            total_loss += feature_loss
            
        return total_loss / len(self.compute_ids)


class TabularWasserstein(nn.Module):
    """
    Wasserstein distance loss for tabular data with ordinal relationships.
    
    Computes Earth Mover's Distance between predicted and target distributions,
    useful for ordinal classification tasks where class ordering matters.
    
    Args:
        encoding_info (Dict[str, int]): Mapping of feature names to vocabulary sizes
        ignore_index (int): Index to ignore in loss calculation. Default: -100
    """
    
    def __init__(self, encoding_info: Dict[str, Dict[str, int]], ignore_index: int=-100) -> None:
        super(TabularWasserstein, self).__init__()
        self.encoding_info = encoding_info
        self.ignore_index = ignore_index
        self.compute_ids = [i for i, (_, v) in enumerate(encoding_info.items())]
        self.vocab_sizes = [v.get('num_bins', v.get('num_categories')) for _, v in encoding_info.items()]
        
    def _compute_wasserstein(self, prediction: torch.Tensor, target: torch.Tensor, 
                           vocab_size: int) -> torch.Tensor:
        """
        Compute Wasserstein distance for a single feature.
        
        Args:
            prediction (torch.Tensor): Logits for the feature
            target (torch.Tensor): Target class indices
            vocab_size (int): Vocabulary size for this feature
            
        Returns:
            torch.Tensor: Wasserstein distance loss
        """
        
        # Create mask for valid targets
        valid_mask = target != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)
        
        # Convert logits to probabilities
        pred_probs = F.softmax(prediction[valid_mask], dim=-1)
        
        # Convert targets to one-hot encoding (excluding the placeholder)
        target_onehot = F.one_hot(target[valid_mask], vocab_size + 1).float()
        
        # Compute cumulative distributions
        pred_cumsum = pred_probs.cumsum(dim=-1)
        target_cumsum = target_onehot.cumsum(dim=-1)
        
        # Compute Wasserstein distance (L2 norm of cumulative differences)
        # wasserstein_dist = ((pred_cumsum - target_cumsum)**2).sum(dim=-1)
        wasserstein_dist = ((pred_cumsum - target_cumsum)**2).mean(dim=-1)
        
        return wasserstein_dist.mean()

    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-task Wasserstein loss.
        
        Args:
            predictions (List[torch.Tensor]): List of prediction tensors for each feature
            targets (torch.Tensor): Target tensor of shape (batch_size, num_features)
            
        Returns:
            torch.Tensor: Average Wasserstein loss across all features
        """
        
        if len(self.compute_ids) == 0:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        total_loss = 0.0
        for feature_idx in self.compute_ids:
            feature_loss = self._compute_wasserstein(
                predictions[feature_idx], 
                targets[:, feature_idx],
                self.vocab_sizes[feature_idx]
            )
            total_loss += feature_loss
            
        return total_loss / len(self.compute_ids)

