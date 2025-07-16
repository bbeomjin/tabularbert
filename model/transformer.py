import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerBlock(nn.Module):
    """
    """

    def __init__(self,
                 d_model: int, 
                 n_heads: int, 
                 d_feed_forward: int,
                 dropout: float):
        """
        """

        super().__init__()
        self.attention = MultiHeadedAttention(n_heads = n_heads, d_model = d_model)
        self.feed_forward = PositionwiseFeedForward(d_model = d_model, 
                                                    d_hidden = d_feed_forward,
                                                    dropout = dropout)
        self.attn_resid_connect = ResidualConnection(d_model = d_model, dropout = dropout)
        self.ff_resid_connect = ResidualConnection(d_model = d_model, dropout = dropout)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, mask):
        x = self.attn_resid_connect(x, lambda _x: self.attention(_x, _x, _x, mask = mask))
        x = self.ff_resid_connect(x, self.feed_forward)
        return self.dropout(x)
    
    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.f(x)

class ResidualConnection(nn.Module):
    
    def __init__(self, 
                 d_model: int,
                 dropout: float):
        super(ResidualConnection, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, layer):
        out = layer(self.layer_norm(x))
        return x + self.dropout(out)
    

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self):
        super(Attention, self).__init__()
    
    def forward(self, query, key, value, mask = None, dropout = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self,
                 n_heads: int,
                 d_model: int,
                 dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        
        query = self.query_linear(query).view(batch_size, -1, self.n_heads, self.d_k)
        key = self.key_linear(key).view(batch_size, -1, self.n_heads, self.d_k)
        value = self.value_linear(value).view(batch_size, -1, self.n_heads, self.d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query = query.transpose(1, 2),
                                 key = key.transpose(1, 2),
                                 value = key.transpose(1, 2),
                                 mask = mask,
                                 dropout = self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).flatten(2)

        return self.output_linear(x)
    

# For test  
# encoder = TransformerBlock(10, 2, 10, 0.1)
# X = torch.randn(10, 5, 10)
# encoder(X, mask = None)