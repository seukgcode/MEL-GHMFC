
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    res_scores = dropout(scores) if dropout is not None else scores

    return torch.matmul(res_scores, value), 0
