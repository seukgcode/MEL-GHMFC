"""
Multi-headed Attention
"""
import torch.nn as nn
from utils.helper import clones, attention


class MultiHeadedAttenton(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttenton, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Set up a mask for each head
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Mapping all features in a batch：d_model => h * d_k
        Q, K, V = [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for linear, x in zip(self.linears, (query, key, value))]

        # 2) Calculate attention
        x, self.attn = attention(Q, K, V, mask, self.dropout)

        # 3) Using view to realize concat
        x_all = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x_all)
