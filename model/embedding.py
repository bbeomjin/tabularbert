import torch.nn as nn
import torch
import math


# class NumEmbedding(nn.Module):
#     def __init__(self, 
#                  max_len,
#                  embedding_dim,
#                  padding_idx = None):
#         super(NumEmbedding, self).__init__()
#         self.max_len = max_len
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.embed = nn.Embedding(max_len, embedding_dim)
        
#     def forward(self, ids, level_ids):
#         embed_out = self.embed(ids)
#         lev_embed = embed_out[level_ids == 1].view(ids.size(0), -1, self.embedding_dim)
#         sublevel_embed = embed_out[level_ids == 2].view(ids.size(0), -1, self.embedding_dim)
#         return lev_embed + sublevel_embed


class MulticolNumEmbedding(nn.Module):
    def __init__(self, 
                 max_len,
                 max_position,
                 embedding_dim,
                 mask_idx = None):
        super(MulticolNumEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.max_position = max_position
        self.mask_idx = mask_idx
        
        if mask_idx is not None:
            # For [MASK] token
            max_len += 1
        
        self.bin_embeddings = nn.ModuleList()
        self.subbin_embeddings = nn.ModuleList()
        for _ in range(max_position):
            self.bin_embeddings.append(nn.Embedding(max_len, embedding_dim))
            self.subbin_embeddings.append(nn.Embedding(max_len, embedding_dim))
        
    def forward(self, bin_ids, subbin_ids):
        bin_embedded = torch.stack([self.bin_embeddings[i](bin_ids[:, i]) for i in range(self.max_position)], dim = 1)
        subbin_embedded = torch.stack([self.subbin_embeddings[i](subbin_ids[:, i]) for i in range(self.max_position)], dim = 1)
        return bin_embedded + subbin_embedded

class NumEmbedding(nn.Module):
    def __init__(self, 
                 max_len,
                 max_position,
                 embedding_dim,
                 mask_idx = None):
        super(NumEmbedding, self).__init__()
        self.max_len = max_len
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        self.mask_idx = mask_idx
        
        if mask_idx is not None:
            # max_position += 1
            max_len += 1
            
        self.positional_embedding = PositionalEmbedding(max_position, embedding_dim)
        self.bin_embedding = nn.Embedding(max_len, embedding_dim)
        self.subbin_embedding = nn.Embedding(max_len, embedding_dim)
        self.cls_embedding = nn.Embedding(1, embedding_dim)
        # self.bin_embedding = PLEmbedding(max_position, embedding_dim)
        # self.linear = EmbeddingLinear(max_len, embedding_dim)
        # self.subbin_embedding = PLEmbedding(max_position, embedding_dim)
        # self.lev_embed = nn.Sequential(
        #     PLEmbedding(max_len, embedding_dim),
        #      EmbeddingLinear(embedding_dim))
        # self.sublev_embed = nn.Sequential(
        #     PLEmbedding(max_len, embedding_dim),
        #      EmbeddingLinear(embedding_dim))
        # self.mask_w = nn.Parameter(torch.randn(1, embedding_dim))
        
    def forward(self, bin_ids, subbin_ids):
        cls_token = torch.zeros(bin_ids.size(0), 1, dtype = torch.long, device = bin_ids.device)
        cls_embedded = self.cls_embedding(cls_token)
        
        positional_embedded = self.positional_embedding(bin_ids)
        bin_embedded = self.bin_embedding(bin_ids)
        subbin_embedded = self.subbin_embedding(subbin_ids)
        embedded = bin_embedded + subbin_embedded + positional_embedded
        # lev_embed_out[level_ids == self.mask_idx] = self.mask_w
        # lev_linear_out = self.linear(level_ids)
        # lev_embed_out = lev_embed_out * lev_linear_out
        # sublev_embed_out[sublevel_ids == self.mask_idx] = self.mask_w
        embedded = torch.concat([cls_embedded, embedded], dim = 1)
        return embedded
        # return bin_embedded + subbin_embedded
        # return bin_embedded + positional_embedded
        # return bin_embedded


# class PositionalEmbedding(nn.Module):

#     def __init__(self, 
#                  max_len: int,
#                  embedding_dim: int,
#                  ):
#         super(PositionalEmbedding, self).__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, embedding_dim).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_len: int,
                 embedding_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_len, embedding_dim)
        # self.weight = nn.Parameter(torch.randn(max_len, embedding_dim) * 0.1)
    def forward(self, x):
        return self.embedding(torch.arange(x.size(1), device = x.device))
        # return self.weight[:x.size(1)]


# class NumEmbedding(nn.Module):
#     def __init__(self, 
#                  lev_num_embeddings,
#                  sublev_num_embeddings,
#                  embedding_dim,
#                  padding_idx = None):
#         super(NumEmbedding, self).__init__()
#         self.lev_num_embeddings = lev_num_embeddings
#         self.sublev_num_embeddings = sublev_num_embeddings
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         # self.lev_embed = nn.Embedding(lev_num_embeddings, embedding_dim)
#         self.lev_embed = ScaleEmbedding(embedding_dim)
#         # self.sublev_embed = nn.Embedding(sublev_num_embeddings, embedding_dim)
#         self.sublev_embed = ScaleEmbedding(embedding_dim)
        
#     def forward(self, x):
#         lev, sublev = x
#         lev_embed_out = self.lev_embed(lev)
#         sublev_embed_out = self.sublev_embed(sublev)
#         # return lev_embed_out + sublev_embed_out
#         return sublev_embed_out


class PLEmbedding(nn.Module):
    def __init__(self,
                 max_len: int,
                 embedding_dim: int):
        super(PLEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        
        bins = torch.linspace(0, 1, embedding_dim + 1)
        bins[0] = -1e-8
        ple = torch.ones(max_len, embedding_dim).float()
        ple.requires_grad = False
        
        self.register_buffer('bins', bins)
        self.register_buffer('ple', ple)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        ple = self.ple.repeat(x.size(0), 1, 1)
        idx_t_1 = x >= self.bins[:-1]
        idx_t = x < self.bins[1:]
        idx = idx_t_1 & idx_t
        ple[:, :x.size(1)][idx_t] = 0.0
        ple[:, :x.size(1)][idx] = (((x - self.bins[:-1]) * idx) / ((self.bins[1:] - self.bins[:-1]) * idx))[idx]
        return ple[:, :x.size(1)]


class EmbeddingLinear(nn.Module):
    def __init__(self,
                 max_len: int,
                 embedding_dim: int):
        super(EmbeddingLinear, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.randn(1, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
    def forward(self, x):
        x = torch.log(1 + self.W.exp()) + self.bias
        return x



# class EmbeddingLinear(nn.Module):
#     def __init__(self,
#                  embedding_dim: int):
#         super(EmbeddingLinear, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
#         self.bias = nn.Parameter(torch.zeros(embedding_dim))
#     def forward(self, x):
#         x = torch.matmul(x, torch.log(1 + self.W.exp())) 
#         return x

   
# class PositionalEmbedding(nn.Module):

#     def __init__(self, 
#                  max_len: int,
#                  d_model: int,
#                  ):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]

# class ScaleEmbedding(nn.Module):

#     def __init__(self, 
#                  d_model: int):
#         super().__init__()
#         self.d_model = d_model
#         # Compute the numerical encodings.
#         mul_term = (math.log10(2 * math.pi) - torch.arange(0, d_model / 2, 1).float()).pow(10)
#         self.register_buffer('mul_term' , mul_term.view(1, 1, -1))

#     def forward(self, x):
#         N, L = x.shape
#         x = x.unsqueeze(-1) * self.mul_term
#         ne = torch.zeros(N, L, self.d_model, device = x.device).float()
#         ne[:, :, 0::2] = torch.sin(x)
#         ne[:, :, 1::2] = torch.cos(x)
#         return ne



# embed = PositionalEmbedding(100, 10)
# embed(torch.arange(3).reshape(1, -1))


# se = ScaleEmbedding(6)
# res = se(torch.tensor([[1, 2, 3, 4]], dtype = torch.float32))

# (res[0, 0] - res[0, 0].mean()) / res[0, 0].std()
# (res[0, 1] - res[0, 1].mean()) / res[0, 1].std()






# class BERTEmbedding(nn.Module):
#     """
#     BERT Embedding which is consisted with under features
#         1. TokenEmbedding : normal embedding matrix
#         2. PositionalEmbedding : adding positional information using sin, cos
#         2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

#         sum of all these features are output of BERTEmbedding
#     """

#     def __init__(self, vocab_size, embed_size, padding_idx, num_idx, dropout=0.1):
#         """
#         :param vocab_size: total vocab size
#         :param embed_size: embedding size of token embedding
#         :param dropout: dropout rate
#         """
#         super(BERTEmbedding, self).__init__()
#         # self.token = TokenEmbedding(vocab_size = vocab_size, embed_size = embed_size)
#         self.token = NumEmbedding(vocab_size, embed_size, padding_idx = padding_idx, num_idx = num_idx)
#         self.position = PositionalEmbedding(d_model = self.token.embedding_dim, max_len = 2048)
#         self.scale = ScaleEmbedding(d_model = self.token.embedding_dim)
#         self.dropout = nn.Dropout(p = dropout)
#         self.embed_size = embed_size

#     def forward(self, x):
#         x = self.token(x) + self.position(x[0])
#         return self.dropout(x)


# class NumEmbedding(nn.Module):
#     def __init__(self, 
#                  num_embeddings,
#                  embedding_dim,
#                  padding_idx = None,
#                  num_idx = None):
#         super(NumEmbedding, self).__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.num_idx = num_idx
#         self.embed = nn.Embedding(num_embeddings, embedding_dim)
#         self.lin_embed = nn.Linear(1, embedding_dim, bias = False)
#         self.scale_embed = ScaleEmbedding(embedding_dim)
        
#     def forward(self, x):
#         tokens, nums = x
#         N, L = nums.shape
#         embed_out = self.embed(tokens)
#         num_embed_out = self.lin_embed(nums.unsqueeze(-1))
#         # num_embed_out += self.scale_embed(nums)
#         embed_out[tokens == self.num_idx] = num_embed_out.view(N * L, -1)
#         return embed_out
        

# class TokenEmbedding(nn.Embedding):
#     def __init__(self, 
#                  vocab_size: int,
#                  embed_size: int = 512,
#                  padding_idx: int = 0,
#                  ) -> None:
#         super().__init__(vocab_size, embed_size, padding_idx = padding_idx)
        
# class SegmentEmbedding(nn.Embedding):
#     def __init__(self, 
#                  embed_size: int = 512,
#                  padding_idx: int = 0,
#                  ) -> None:
#         super().__init__(3, embed_size, padding_idx = padding_idx)
        
# class PositionalEmbedding(nn.Module):

#     def __init__(self, 
#                  d_model: int, 
#                  max_len: int = 512):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]
    

    

