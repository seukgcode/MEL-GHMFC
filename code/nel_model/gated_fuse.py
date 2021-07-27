import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, args):
        super(GatedFusion, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

        self.lin_seq_att = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.lin_img_att = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(dim=0),
        )

        self.filtration_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )

    def forward(self, ques_w, img_w, ques_p=0, img_p=0, ques_q=0, img_q=0):
        seq_att = self.lin_seq_att(ques_w + ques_p + ques_q)
        img_att = self.lin_img_att(img_w + img_p + img_q)

        attn_gate = self.gate(torch.cat([seq_att.unsqueeze(0), img_att.unsqueeze(0)], dim=0)).squeeze()
        fusion = (seq_att - img_att) * attn_gate[0].unsqueeze(-1) + img_att
        out = self.filtration_gate(torch.cat([seq_att, fusion], dim=-1))

        return out


if __name__ == '__main__':
    from args import Args
    a = Args()

    x = torch.randn(a.batch_size, a.hidden_size)
    y = torch.randn(a.batch_size, a.hidden_size)
    gf = GatedFusion(args=a)
    res = gf(x,y)
    print(res.size())