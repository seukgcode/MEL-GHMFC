"""
    -----------------------------------
    word level features
"""
import torch.nn as nn
from utils import clones
from utils import MultiHeadedAttenton
from utils import EncoderLayer, Encoder, DecoderLayer, Decoder
from utils import PositionwiseFeedForward


class WordLevel(nn.Module):
    def __init__(self, args):
        super(WordLevel, self).__init__()
        self.hidden_size = args.hidden_size
        self.nheaders = args.nheaders
        self.nlayers = args.num_attn_layers
        self.dropout = args.dropout
        self.ff_size = args.ff_size
        self.seq_len = args.max_sent_length
        self.img_len = args.img_len



        self.decodes = clones(Decoder(
            DecoderLayer(self.hidden_size,
                         MultiHeadedAttenton(self.nheaders,
                                             self.hidden_size,
                                             self.dropout),
                         MultiHeadedAttenton(self.nheaders,
                                             self.hidden_size,
                                             self.dropout),
                         PositionwiseFeedForward(
                             self.hidden_size,
                             self.ff_size,
                             self.dropout
                         ),
                         self.dropout),
            self.nlayers
            ),
            2
        )

    def forward(self, seq, img, mask):
        """
            Word level features
            ------------------------------------------
            Args:
                seq: tensor, (b, seq_len, hidden_size),
                img: tensor, (b, img_len, hidden_size),
                mask: tensor, (b, 1, seq_len),
            Returns:
        """


        img_feat = self.decodes[0](img, seq, None, mask)
        seq_feat = self.decodes[1](seq, img_feat, mask, None)
        layer_attns_img = self.decodes[0].layer_attns
        layer_attns_seq = self.decodes[1].layer_attns

        img_feat_lin = img_feat.max(dim=1)[0]
        seq_feat_lin = seq_feat.max(dim=1)[0]

        return seq_feat_lin, img_feat_lin, (layer_attns_img, layer_attns_seq)
