
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ConvFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ConvFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv2(self.dropout(F.relu(self.conv1(x.permute(0, 2, 1))))).permute(0, 2, 1)
        return x


