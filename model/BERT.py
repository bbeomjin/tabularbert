import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock
from .embedding import NumEmbedding


# class BERT(nn.Module):
#     """
#     BERT model : Bidirectional Encoder Representations from Transformers.
#     """

#     def __init__(self, 
#                  embedding,
#                  n_layers: int = 12, 
#                  n_heads: int = 12,
#                  dropout: float = 0.1):
#         """
#         :param vocab_size: vocab_size of total words
#         :param hidden: BERT model hidden size
#         :param n_layers: numbers of Transformer blocks(layers)
#         :param attn_heads: number of attention heads
#         :param dropout: dropout rate
#         """

#         super(BERT, self).__init__()
#         self.d_model = embedding.embedding_dim
#         self.n_layers = n_layers
#         self.n_heads = n_heads

#         # paper noted they used 4*hidden_size for ff_network_hidden_size
#         # self.d_feed_forward = self.d_model * 4
#         self.d_feed_forward = self.d_model * 1

#         # embedding for BERT, sum of positional, segment, token embeddings
#         self.embedding = embedding

#         # multi-layers transformer blocks, deep network
#         self.transformer_blocks = nn.ModuleList(
#             [TransformerBlock(self.d_model, n_heads, self.d_feed_forward, dropout) for _ in range(n_layers)])
    
#     def forward(self, level_ids, sublevel_ids, attn_mask):
#         # attention masking for padded token
#         # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
#         mask = (attn_mask > 0).unsqueeze(1).repeat(1, attn_mask.size(1), 1).unsqueeze(1)

#         # embedding the indexed sequence to sequence of vectors
#         x = self.embedding(level_ids, sublevel_ids)

#         # running over multiple transformer blocks
#         for transformer in self.transformer_blocks:
#             x = transformer.forward(x, mask)

#         return x
    

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, 
                 embedding,
                 n_layers: int = 12, 
                 n_heads: int = 12,
                 dropout: float = 0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(BERT, self).__init__()
        self.embedding_dim = embedding.embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = embedding

        # multi-layers transformer blocks, deep network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = self.n_heads,
            dropout = dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, n_layers
        )
    
    def forward(self, bin_ids, subbin_ids):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (attn_mask > 0).unsqueeze(1).repeat(1, attn_mask.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(bin_ids, subbin_ids)

        # running over multiple transformer blocks
        x = self.transformer(x)

        return x


class GMTMaskedLanguageModel(nn.Module):
    """
    """

    def __init__(self, bert: BERT, encoded_info):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super(GMTMaskedLanguageModel, self).__init__()
        self.bert = bert
        self.encoded_info = encoded_info
        self.fc = nn.ModuleList()
        for k in encoded_info.keys():
            self.fc.append(nn.Linear(self.bert.embedding_dim, encoded_info[k]['K'] * encoded_info[k]['L']))
        # self.fc = nn.Linear(self.bert.embedding_dim, encoded_info[0]['K'] * encoded_info[0]['L'])
        
    def forward(self, bin_ids, subbin_ids):
        bert_out = self.bert(bin_ids, subbin_ids)
        out = list()
        for j in range(bert_out.size(1) - 1):
            # out.append(self.fc[j](bert_out[:, j]))
            out.append(self.fc[j](bert_out[:, j + 1]))
            # out.append(self.fc(bert_out[:, j]))
        return out


# class GMTMaskedLanguageModel2(nn.Module):
#     """
#     """

#     def __init__(self, bert: BERT, encoded_info):
#         """
#         :param bert: BERT model which should be trained
#         :param vocab_size: total vocab size for masked_lm
#         """

#         super(GMTMaskedLanguageModel2, self).__init__()
#         self.bert = bert
#         self.encoded_info = encoded_info
#         self.fc = nn.Linear(self.bert.embedding_dim, 100)
        
#     def forward(self, bin_ids, subbin_ids):
#         bert_out = self.bert(bin_ids, subbin_ids)
#         out = self.fc(bert_out)
#         return out


class HierarchyCassification(nn.Module):
    """
    """

    def __init__(self, bert: BERT, vocab_size, n_levels = 100):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super(HierarchyCassification, self).__init__()
        self.bert = bert
        self.level_lm = MaskedLanguageModel(self.bert.embedding_dim, vocab_size)
        self.W = nn.Parameter(torch.randn(n_levels, self.bert.embedding_dim, vocab_size))
        self.bias = nn.Parameter(torch.zeros(n_levels, vocab_size))
        
    def forward(self, level_ids, sublevel_ids):
        attn_mask = torch.ones((level_ids.size(0), level_ids.size(1)), dtype = level_ids.dtype, device = sublevel_ids.device)
        x = self.bert(level_ids, sublevel_ids, attn_mask)
        out = self.level_lm(x)
        ix = torch.argmax(out, dim = -1)
        sublevel_out = torch.matmul(x.unsqueeze(-1).transpose(-1, -2), self.W[ix]).squeeze(-2) + self.bias[ix]
        return out, sublevel_out


    
class RegBERT(nn.Module):
    """
    BERT for Selfies
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super(RegBERT, self).__init__()
        self.bert = bert
        # self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.lm = MaskedLanguageModel(self.bert.embedding_dim, vocab_size)

    def forward(self, x, attn_mask, segment_label):
        x = self.bert(x, attn_mask, segment_label)
        return self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.linear(x)