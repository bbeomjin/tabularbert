import torch
import torch.nn as nn
from typing import Dict, List


class BERT(nn.Module):
    """
    BERT (Bidirectional Encoder Representations from Transformers) model implementation.
    
    This class implements a BERT-style transformer encoder that processes embedded sequences
    and applies bidirectional attention to learn contextual representations.
    
    The model consists of multiple transformer encoder layers that can process sequences
    with attention mechanisms to capture long-range dependencies.
    
    Args:
        embedding_dim (int): Dimension of input embeddings. Default: 256
        n_layers (int): Number of transformer encoder layers. Default: 12
        n_heads (int): Number of attention heads in each layer. Default: 12
        dropout (float): Dropout probability for regularization. Default: 0.1
    """

    def __init__(self, 
                 embedding_dim: int=256,
                 n_layers: int=12, 
                 n_heads: int=12,
                 dropout: float=0.1) -> None:
        super(BERT, self).__init__()
        
        # Validate that embedding_dim is divisible by n_heads
        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})")
        
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # Create transformer encoder layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BERT model.
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            torch.Tensor: Contextualized representations of shape (batch_size, seq_len, embedding_dim)
        """
        # Apply transformer layers with optional padding mask
        x = self.transformer(embeddings)
        return x



class Classifier(nn.Module):
    """
    Classification head for sequence-to-sequence prediction.
    
    This classifier applies different linear transformations to each position in the sequence
    (excluding the CLS token) to predict class labels for multiple tasks simultaneously.
    Each position can have a different number of output classes as specified in encoding_info.
    
    Args:
        embedding_dim (int): Dimension of input embeddings from the BERT model
        encoding_info (Dict[str, int]): Dictionary of binning information
                                       e.g., {'var1': 10, 'var2': 5} for 10 and 5 classes respectively
    """
    
    def __init__(self, 
                 embedding_dim: int, 
                 encoding_info: Dict[str, Dict[str, int]]) -> None:
        super(Classifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.encoding_info = encoding_info
        
        # Create linear layers for each task
        self.fc = nn.ModuleList([
            nn.Linear(embedding_dim, v.get('num_bins', v.get('num_categories')) + 1) 
            for _, v in encoding_info.items()
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using normal distribution."""
        for layer in self.fc:
            if layer is not None:
                nn.init.normal_(layer.weight, mean=0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
                             where seq_len includes the CLS token at position 0
        
        Returns:
            List[torch.Tensor]: List of classification logits for each position (excluding CLS token)
                               Each tensor has shape (batch_size, num_classes_for_variable_i)
        """
        
        if x.shape[1] != len(self.encoding_info):
            raise ValueError(f"Expected {len(self.encoding_info)} sequence length, got {x.shape[1]}")
        
        outputs = []
        # The following indexing may occur mismatched outputs with the encoding_info when the encoding_info is not sorted by the variable name
        # TODO: define variable indicies matching the variable name and j set to the variable index
        for j in range(len(self.encoding_info)):
            # Apply corresponding linear layer to position j
            logits = self.fc[j](x[:, j])
            outputs.append(logits)
            
        return outputs



class Regressor(nn.Module):
    """
    Regression head for sequence-to-value prediction.
    
    This regressor applies linear transformations to BERT embeddings from each sequence position
    (excluding the CLS token) to predict continuous real values for multiple regression tasks.
    Each position's embedding is independently mapped to a single scalar value.
    
    Args:
        embedding_dim (int): Dimension of input embeddings from the BERT model
        encoding_info (Dict[str, int]): Dictionary of binning information
                                      The values are not used for regression (always output 1 value per task)
                                      but kept for consistency with Classifier interface
    """
    
    def __init__(self,
                 embedding_dim: int,
                 encoding_info: Dict[str, Dict[str, int]]) -> None:
        super(Regressor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.encoding_info = encoding_info
        
        # Create linear layers for each regression task (each outputs 1 value)
        self.fc = nn.ModuleList([
            nn.Linear(embedding_dim, 1) 
            if 'num_bins' in v.keys() else None for k, v in encoding_info.items()
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using normal distribution."""
        for layer in self.fc:
            if layer is not None:
                nn.init.normal_(layer.weight, mean=0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the regressor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
                             where seq_len includes the CLS token at position 0
        
        Returns:
            List[torch.Tensor]: List of regression outputs for each position (excluding CLS token)
                               Each tensor has shape (batch_size,) containing scalar predictions
        """
        
        if x.shape[1] != len(self.encoding_info):
            raise ValueError(f"Expected {len(self.encoding_info)} sequence length, got {x.shape[1]}")
        
        outputs = []
        
        # The following indexing may occur mismatched outputs with the encoding_info when the encoding_info is not sorted by the variable name
        # TODO: define variable indicies matching the variable name and j set to the variable index
        for j, (k, v) in enumerate(self.encoding_info.items()):
            if 'num_bins' in v.keys():
                # Apply corresponding linear layer to position j
                prediction = self.fc[j](x[:, j]).flatten()
                outputs.append(prediction)
            else:
                outputs.append(None)
            
        return outputs
