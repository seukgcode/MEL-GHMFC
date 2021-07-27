"""
    -----------------------------------
    Achieve triplet loss
"""
import torch
from torch import nn
import torch.nn.functional as F


def cos_sim_batch(input_1, input_2, eps=1e-5):
    """
        batch cos similarity
        ------------------------------------------
        Args:
            input_1: (batch_size, hidden_size)
            input_2: (batch_size, hidden_size)
        Returns:
    """
    # inner = lambda x, y: torch.matmul(x.unsqueeze(1), y.unsqueeze(-1)).squeeze()
    inner = lambda x, y: (x * y).sum(dim=-1)

    dot = inner(input_1, input_2)
    m_1 = inner(input_1, input_1) ** 0.5
    m_2 = inner(input_2, input_2) ** 0.5

    return dot / (m_1 * m_2 + eps)


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.25, sim='cos', p=2):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        """
            call
            ------------------------------------------
            Args:
                anchor: (batch_size, hidden_size)
                pos: (batch_size, hidden_size)
                neg: (batch_size, hidden_size)
            Returns:
        """
        sim_p = cos_sim_batch(anchor, pos)
        sim_n = cos_sim_batch(anchor, neg)

        loss = sim_n - sim_p + self.margin
        hinge_loss = F.relu(loss)
        return hinge_loss.mean()


if __name__ == '__main__':
    x = torch.randn(64, 512)
    y = torch.randn(64, 512)
    print(cos_sim_batch(x, y).size())