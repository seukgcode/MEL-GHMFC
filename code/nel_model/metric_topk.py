"""
    top-k metric
"""
from circle_loss import cosine_similarity, dot_similarity


def cal_top_k(args, query, pos_feats, search_feats):
    """
        Input query, positive sample features, negative sample features
        return the ranking of positive samples
        ------------------------------------------
        Args:
        Returns:
    """

    if args.similarity == 'cos':
        ans = similarity_rank(query, pos_feats, search_feats, cosine_similarity)
    elif args.similarity == 'dot':
        ans = similarity_rank(query, pos_feats, search_feats, dot_similarity)
    else:
        ans = lp_rank(query, pos_feats, search_feats, args.loss_p)

    return ans


def similarity_rank(query, pos_feats, search_feats, cal_sim):
    """
        Sample ranking based on similarity
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    sim_p = cal_sim(query, pos_feats).detach().cpu().numpy()  # batch_size, 1
    sim_s = cal_sim(query, search_feats).detach().cpu().numpy()  # batch_size, n_search

    sim_mat = sim_s - sim_p
    ranks = (sim_mat > 0).sum(-1) + 1


    return ranks, sim_p, sim_s


def lp_distance(x, dim, p):
    return (x ** p).sum(dim=dim) ** (1 / p)


def lp_rank(query, pos_feats, search_feats, p=2):
    """
        Using LP distance to calculate the rank of positive examples
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    dis_p = lp_distance(query - pos_feats.squeeze(), dim=-1, p=p).detach().cpu().numpy()
    dis_sf = lp_distance(query.unsqueeze(1) - search_feats, dim=-1, p=p).detach().cpu().numpy()

    batch_size = dis_p.size(0)
    for i in range(batch_size):
        rank = 0
        for dis in dis_sf[i]:
            if dis < dis_p[i]:
                rank += 1
        rank_list.append(rank)

    return rank_list, dis_p, dis_sf