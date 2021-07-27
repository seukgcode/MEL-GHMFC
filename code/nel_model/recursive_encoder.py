import torch
import torch.nn as nn


class RecursiveEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, embedding_dim_last, output_dim):
        super(RecursiveEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_dim_last = embedding_dim_last
        self.output_dim = output_dim

        self.dense_hw = nn.Linear(self.input_dim, self.embedding_dim)
        self.dense_hp = nn.Linear(self.input_dim + self.embedding_dim, self.embedding_dim)
        self.dense_hs = nn.Linear(self.input_dim + self.embedding_dim, self.embedding_dim_last)
        self.dense_out = nn.Linear(self.embedding_dim_last, self.output_dim)

    def forward(self, ques_w, img_w, ques_p, img_p, ques_q, img_q):
        hw = nn.Dropout(0.5)(ques_w + img_w)
        hw = nn.Tanh()(self.dense_hw(hw))

        hp = nn.Dropout(0.5)(torch.cat((ques_p + img_p, hw), dim=1))
        hp = nn.Tanh()(self.dense_hp(hp))

        hs = nn.Dropout(0.5)(torch.cat((ques_q + img_q, hp), dim=1))
        hs = nn.Tanh()(self.dense_hs(hs))

        out = self.dense_out(nn.Dropout(0.5)(hs))

        return out
