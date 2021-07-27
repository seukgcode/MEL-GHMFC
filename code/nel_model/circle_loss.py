"""
    Realization of circle loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_similarity(query, feats):
    """
        dot similarity with batch
        ------------------------------------------
        Args:
            query: (batch_size, hidden_size)
            feats: (batch_size, n, hidden_size)
        Returns:
            sim: (batch_size, n)
    """
    return feats.matmul(query.unsqueeze(-1)).squeeze()


def cosine_similarity(query, feats):
    """
        cosine similarity with batch
        ------------------------------------------
        Args:
            query: (batch_size, hidden_size)
            feats: (batch_size, n, hidden_size)
        Returns:
            sim: (batch_size, n)
    """
    up = feats.matmul(query.unsqueeze(-1))  # (batch, n, 1)
    up = up.squeeze(2)

    abs_query = torch.sqrt(torch.sum(query ** 2, dim=-1, keepdim=True))  # (batch, 1)
    abs_feats = torch.sqrt(torch.sum(feats ** 2, dim=-1))  # (batch, n)
    abs_ = abs_query * abs_feats  # (batch, n)

    res = up / abs_  # batch_n

    return res


class CircleLoss(nn.Module):
    def __init__(self, scale=None, margin=None, similarity=None):
        super(CircleLoss, self).__init__()
        self.scale = scale if scale else 32
        self.margin = margin if margin else 0.25
        self.similarity = similarity if similarity else 'cos'

    def forward(self, q, p, n):
        """
            circle loss
            ------------------------------------------
            Args:
            Returns:
        """

        if self.similarity.lower() == 'dot':
            sim_p = dot_similarity(q, p)
            sim_n = dot_similarity(q, n)
        elif self.similarity == 'cos':
            sim_p = cosine_similarity(q, p)
            sim_n = cosine_similarity(q, n)
        else:
            raise ValueError('This similarity is not implemented.')

        alpha_p = F.relu(-sim_p + 1 + self.margin)  # batch size, n_pos
        alpha_n = F.relu(sim_n + self.margin)  # batch size, n_neg

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -self.scale * alpha_p * (sim_p - delta_p)
        logit_n = self.scale * alpha_n * (sim_n - delta_n)

        loss = F.softplus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

        return loss.mean()


if __name__ == '__main__':
    x = torch.randn(64, 512)
    feats = torch.randn(64, 1, 512)
    print(cosine_similarity(x, feats).size())

    x = torch.tensor([3., 4., 5.])
    y = torch.tensor([6., 8., 10.])
    y.unsqueeze_(dim=1)
    print((torch.randn((64)) / torch.randn((64, 1))).size())
