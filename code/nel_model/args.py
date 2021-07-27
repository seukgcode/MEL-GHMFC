class Args:
    def __init__(self):
        self.hidden_size = 512
        self.dropout = 0.2
        self.batch_size = 32
        self.text_feat_size = 768
        self.img_feat_size = 2048
        self.nheaders = 8
        self.num_attn_layers = 1
        self.ff_size = 2048
        self.max_seq_length = 128
        self.img_len = 196
        self.output_size = 768

        self.loss_scale = 32
        self.loss_margin = 0.25
        self.similarity = 'cos'
