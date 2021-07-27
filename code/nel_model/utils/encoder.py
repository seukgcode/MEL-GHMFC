import torch.nn as nn
from utils.helper import clones
from utils.norm import LayerNorm
from utils.residual import SublayerConnection


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.layer_attns = []

    def forward(self, query, key, mask=None):
        for layer in self.layers:
            query = layer(query, key, mask)
            self.layer_attns.append(layer.attn)
        return self.norm(query)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.layer_attns = []

    def forward(self, query, key, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            query = layer(query, key, src_mask, tgt_mask)
            self.layer_attns.append(layer.attn)
        return self.norm(query)


class EncoderLayer(nn.Module):
    """
        encoder consisits of two layers：
        multi-head self-attention + feed forward
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.attn = None

    def forward(self, query, key, mask):
        query_ = self.sublayer[0](query, lambda feat: self.self_attn(query, key, key, mask))
        self.attn = self.self_attn.attn
        return self.sublayer[1](query_, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size
        self.attn = None

    def forward(self, query, key, src_mask, tgt_mask):
        query_1 = self.sublayer[0](query, lambda feat: self.self_attn(query, query, query, src_mask))

        query_2 = self.sublayer[1](query, lambda feat: self.cross_attn(query_1, key, key, tgt_mask))

        self.attn = self.cross_attn.attn
        return self.sublayer[2](query_2, self.feed_forward)
