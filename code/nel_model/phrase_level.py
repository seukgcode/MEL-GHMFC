"""
    -----------------------------------
    phrase level features
"""

import torch
import torch.nn as nn
from utils import clones
from utils import MultiHeadedAttenton
from utils import EncoderLayer, Encoder, DecoderLayer, Decoder
from utils import PositionwiseFeedForward


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class PhraseLevel(nn.Module):
    def __init__(self, args):
        super(PhraseLevel, self).__init__()
        self.hidden_size = args.hidden_size
        self.nheaders = args.nheaders
        self.dropout = args.dropout
        self.ff_size = args.ff_size
        self.seq_len = args.max_sent_length
        self.img_len = args.img_len
        self.nlayers = args.num_attn_layers

        self.conv_unigram = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size,
                      out_channels=self.hidden_size,
                      kernel_size=1,
                      stride=1),
            Permute(0, 2, 1),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )
        self.conv_bigram = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size,
                      out_channels=self.hidden_size,
                      kernel_size=2,
                      padding=1),
            Permute(0, 2, 1),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )
        self.conv_trigram = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size,
                      out_channels=self.hidden_size,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            Permute(0, 2, 1),
            nn.LayerNorm(self.hidden_size)
        )

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
            Phrase-level features
            ------------------------------------------
            Args:
                seq: tensor, (b, seq_len, hidden_size), Sentence features
                img: tensor, (b, img_len, hidden_size), Image features
                mask: tensor, (b, 1, seq_len), Sentence mask
            Returns:
        """

        # phrase feat
        seq_p = seq.permute(0, 2, 1)
        unigram = self.conv_unigram(seq_p)

        bigram = self.conv_bigram(seq_p)
        bigram = bigram.narrow(1, 0, self.seq_len)

        trigram = self.conv_trigram(seq_p)

        unigram = unigram.unsqueeze(-1)
        bigram = bigram.unsqueeze(-1)
        trigram = trigram.unsqueeze(-1)

        context_feat = torch.max(torch.cat((unigram, bigram, trigram), dim=-1), -1)[0]



        img_feat = self.decodes[0](img, context_feat, None, mask)
        seq_feat = self.decodes[1](context_feat, img_feat, mask, None)

        layer_attns_img = self.decodes[0].layer_attns
        layer_attns_seq = self.decodes[1].layer_attns

        img_feat_lin = img_feat.max(dim=1)[0]
        seq_feat_lin = seq_feat.max(dim=1)[0]

        return seq_feat_lin, img_feat_lin, context_feat, (layer_attns_img, layer_attns_seq)

