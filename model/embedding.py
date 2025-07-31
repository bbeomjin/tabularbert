import torch.nn as nn
import torch
import math

class NumEmbedding(nn.Module):
    """
    Numerical embedding layer that combines bin embeddings with positional embeddings.
    
    This module creates embeddings for numerical sequences by:
    1. Converting bin IDs to embeddings
    2. Adding positional information
    3. Prepending a learnable CLS token
    
    Args:
        max_len (int): Maximum length of the input sequence (number of bin indices)
        max_position (int): Maximum position for positional embeddings (number of columns in the input)
        embedding_dim (int): Dimension of the embedding vectors
        mask_idx (int, optional): Index used for masking. 
    """
    
    def __init__(self, 
                 max_len: int,
                 max_position: int,
                 embedding_dim: int,
                 mask_idx: int = None) -> None:
        super(NumEmbedding, self).__init__()
        self.max_len = max_len
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        self.mask_idx = mask_idx
        
        if mask_idx is not None:
            max_len += 1
            
        self.positional_embedding = PositionalEmbedding(max_position, embedding_dim)
        self.bin_embedding = nn.Embedding(max_len, embedding_dim)
        self.cls_embedding = nn.Embedding(1, embedding_dim)
        
        # Pre-register [CLS] token as a buffer for efficiency
        # This avoids creating the tensor in every forward pass
        self.register_buffer('cls_token', torch.zeros(1, 1, dtype = torch.long))
        
    def forward(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NumEmbedding layer.
        
        Args:
            bin_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing bin indices
        
        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, sequence_length + 1, embedding_dim)
                         with CLS token prepended
        """
        batch_size = bin_ids.size(0)
        
        # Efficiently create CLS tokens for the batch
        cls_token = self.cls_token.expand(batch_size, -1)
        cls_embedded = self.cls_embedding(cls_token)
        
        # Get positional and bin embeddings
        positional_embedded = self.positional_embedding(bin_ids)
        bin_embedded = self.bin_embedding(bin_ids)
        
        # Combine bin and positional embeddings
        embedded = bin_embedded + positional_embedded
        
        # Prepend CLS token
        embedded = torch.cat([cls_embedded, embedded], dim = 1)
        
        return embedded
        
class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding layer that adds position information to sequences.
    
    This module creates position-dependent embeddings that are added to input embeddings
    to provide the model with information about the position (column) of each element in the sequence.
    
    Args:
        max_len (int): Maximum sequence length that can be processed (number of columns)
        embedding_dim (int): Dimension of the embedding vectors
    """
    
    def __init__(self,
                 max_len: int,
                 embedding_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_len, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEmbedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
                             Used only to determine sequence length and device
        
        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        seq_len = x.size(1)
        # Create position indices for the sequence length
        positions = torch.arange(seq_len, device = x.device, dtype = torch.long)
        # Expand to match batch size
        positions = positions.unsqueeze(0).expand(x.size(0), -1)
        return self.embedding(positions)


    

if __name__ == '__main__':
    embedding = NumEmbedding(10, 3, 16)
    x = torch.randint(10, (256, 3))
    embedded = embedding(x)
    print(embedded.shape)
    