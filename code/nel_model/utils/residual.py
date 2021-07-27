"""
    Define residual connection
"""
import torch.nn as nn
from utils.norm import LayerNorm


class SublayerConnection(nn.Module):
    """
        Residual link layer
Note that standardization is the first one
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
            ------------------------------------------
            Args:
                x: Input characteristics
                Sublayer: the level of running
            Returns:
        """
        norm_x = self.norm(x)
        sub_x = self.dropout(sublayer(norm_x))
        out = x + sub_x
        return out
