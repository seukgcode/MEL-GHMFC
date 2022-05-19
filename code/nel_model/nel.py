import torch
import torch.nn as nn
# from torch.nn import TripletMarginLoss
from word_level import WordLevel
from phrase_level import PhraseLevel
from sent_level import SentLevel
from gated_fuse import GatedFusion
from recursive_encoder import RecursiveEncoder
from circle_loss import CircleLoss
from triplet_loss import TripletMarginLoss
def Contrastive_loss(out_1, out_2, batch_size, temperature=0.5):
    out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)  # [2*B, 2*B]
    '''
    torch.mm是矩阵乘法，a*b是对应位置上的数相除，维度和a，b一样
    '''
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    '''
    torch.eye生成对角线上为1，其他为0的矩阵
    torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    '''
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1)-pos_sim))).mean()
    return loss

class NELModel(nn.Module):
    def __init__(self, args):
        super(NELModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.output_size = args.output_size
        self.seq_len = args.max_sent_length
        self.text_feat_size = args.text_feat_size
        self.img_feat_size = args.img_feat_size
        self.feat_cate = args.feat_cate.lower()

        self.img_trans = nn.Sequential(
            nn.Linear(self.img_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )
        if 'w' in self.feat_cate:
            self.word_level = WordLevel(args)
        if 'p' in self.feat_cate:
            self.phrase_level = PhraseLevel(args)
        if 's' in self.feat_cate:
            self.sent_level = SentLevel(args)

        self.gated_fuse = GatedFusion(args)

        # Dimension reduction
        self.out_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # circle loss
        self.loss_function = args.loss_function
        self.loss_margin = args.loss_margin
        self.sim = args.similarity
        if self.loss_function == 'circle':
            self.loss_scale = args.loss_scale
            self.loss = CircleLoss(self.loss_scale, self.loss_margin, self.sim)
        else:
            self.loss_p = args.loss_p
            self.loss = TripletMarginLoss(margin=self.loss_margin, p=self.loss_p)

    def forward(self, bert_feat, img, bert_mask=None, pos_feats=None, neg_feats=None):
        """

            ------------------------------------------
            Args:
                bert_feat: tensor: (batch_size, max_seq_len, text_feat_size), the output of bert hidden size
                img: float tensor: (batch_size, ..., img_feat_size), image features - resnet
                bert_mask: tensor: (batch_size, max_seq_len)
                pos_feats(optional): (batch_size, n_pos, output_size)
                neg_feats(optional): (batch_size, n_neg, output_size)
            Returns:
        """
        batch_size = img.size(0)
        img = img.view((batch_size, -1, self.img_feat_size))
        img_trans = self.img_trans(img)

        bert_trans = self.text_trans(bert_feat)
        # sent_feat = bert_trans[:, 0]  # Cut seq_feat: the 0th position is the sentence vector, and the 1-2 position is the word vector
        seq_feat = bert_trans.narrow(dim=1, start=1, length=self.seq_len)

        mask = None
        if bert_mask is not None:
            mask = bert_mask.narrow(dim=1, start=1, length=self.seq_len)
            mask.unsqueeze_(-2)

        seq_att_w, img_att_w, attn_w, seq_att_p, img_att_p, seq_att_s, img_att_s, attn_p = 0, 0, 0, 0, 0, 0, 0, 0

        if 'w' in self.feat_cate:
            seq_att_w, img_att_w, attn_w = self.word_level(seq_feat, img_trans, mask)

        if 'p' in self.feat_cate:
            seq_att_p, img_att_p, context_feat, attn_p = self.phrase_level(seq_feat, img_trans, mask)

            if 's' in self.feat_cate:
                seq_att_s, img_att_s, attn_s = self.sent_level(context_feat, img_trans)
        fusion = self.gated_fuse(seq_att_w, img_att_w, seq_att_p, img_att_p, seq_att_s,img_att_s)

        query = self.out_trans(fusion)

        if pos_feats is None or neg_feats is None:
            return query, \
                   (attn_w, attn_p)

        pos_feats_trans = self.trans(pos_feats)
        neg_feats_trans = self.trans(neg_feats)
        
        ct_text_feats=torch.mean(bert_trans,dim=1)
        ct_text_feats=nn.functional.normalize(cl_text_feats,dim=-1)
        ct_img_feats=torch.mean(img_trans,dim=1)
        ct_img_feats=nn.functional.normalize(cl_img_feats,dim=-1)
        ct_loss=Contrastive_loss(ct_text_feats,ct_img_feats,batch_size,0.6)
 
        if self.loss_function == 'circle':
            loss = self.loss(query, pos_feats_trans, neg_feats_trans)+ct_loss
        else:
            loss = self.loss(query, pos_feats_trans.squeeze(), neg_feats_trans.squeeze())+ct_loss

        return loss, query, \
               (attn_w, attn_p)

    def trans(self, x):
        return x

