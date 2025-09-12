"""
This module provides a neural network model that combines a pretrained TabularBERT model
with a feedforward multilayer perceptron (MLP) for downstream tasks such as classification
and regression on tabular data.
"""

import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    PyTorch implementation of a multilayer perceptron.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_layers (List[int]): List of hidden layer dimensions
        activation (str): Activation function class (default: ReLU)
        dropouts (List[float]): List of dropout probabilities (default: 0.1)
        batch_norm (bool): Whether to use batch normalization (default: False)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str='ReLU',
        dropouts: List[float]=0.1,
        batch_norm: bool=False
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropouts = dropouts
        self.batch_norm = batch_norm
        
        layers = []
        prev_dim = input_dim
        
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(hidden_layers)
        
        # Hidden layers
        for hidden_dim, dropout in zip(hidden_layers, dropouts):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(getattr(nn, activation)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
