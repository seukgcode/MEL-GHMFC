"""
    -----------------------------------
    dataset of nel
"""
import torch
from torch.utils.data import Dataset
import h5py
import json
import re
import random
from os.path import join

INAME_PATTERN = re.compile("/(\d+)\.")


def neg_sample_online(neg_id, neg_iid, tfidf_neg, negid2qid, max_sample_num=3, threshold=0.95):
    """
        Online negative sampling algorithm
        ------------------------------------------
        Args:
        Returns:
    """
    N = len(tfidf_neg)
    cands = set()

    while len(cands) < max_sample_num:
        rand = random.random()
        if not tfidf_neg[neg_id] or rand > threshold:
            cand = random.randint(0, N - 1)
        else:
            rand_word = random.choice(tfidf_neg[neg_id])
            cand = random.choice(neg_iid[rand_word])

        if cand != neg_id:
            cands.add(cand)

    return [negid2qid[c] for c in cands]


class NELDataset(Dataset):
    def __init__(self, args,
                 all_input_ids,
                 all_input_mask,
                 all_segment_ids,
                 all_answer_id,
                 all_img_id,
                 all_mentions,
                 answer_list,
                 contain_search_res=False):
        # text info
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_answer_id = all_answer_id
        self.all_img_id = all_img_id
        self.all_mentions = all_mentions

        # answer
        self.answer_list = answer_list  # id2ansStr
        self.answer_mapping = {answer: i for i, answer in enumerate(self.answer_list)}  # ansStr2id



        # Online negative sampling
        self.max_sample_num = args.neg_sample_num
        neg_config = json.load(open(args.path_neg_config))
        self.neg_iid = neg_config["neg_iid"]
        self.tfidf_neg = neg_config["tfidf_neg"]
        self.negid2qid = neg_config["keys_ordered"]
        self.qid2negid = {qid: i for i, qid in enumerate(neg_config["keys_ordered"])}

        # Sample features of negative sampling
        self.neg_list = json.load(open(join(args.dir_neg_feat, "neg_list.json")))
        self.neg_mapping = {sample: i for i, sample in enumerate(self.neg_list)}
        self.ansid2negid = {i: self.neg_mapping[ans] for i, ans in enumerate(self.answer_list)}

        neg_feat_h5 = h5py.File(join(args.dir_neg_feat, "neg_feats.h5"), 'r')
        self.neg_features = neg_feat_h5.get("features")

        # image features
        img_list = json.load(open(join(args.dir_img_feat, "img_list.json")))
        img_list = [INAME_PATTERN.findall(iname)[0] for iname in img_list]
        self.img_mapping = {iname: i for i, iname in enumerate(img_list)}

        img_feat_h5 = h5py.File(join(args.dir_img_feat, "img_features.h5"), 'r')
        self.img_features = img_feat_h5.get("features")

        # search candidates
        self.contain_search_res = contain_search_res
        if self.contain_search_res:
            self.search_res = json.load(open(args.path_candidates, "r"))  # mention: [qid0, qid1, ..., qidn]

    def __len__(self):
        return len(self.all_answer_id)

    def __getitem__(self, idx):
        sample = dict()
        sample["input_ids"] = self.all_input_ids[idx]  # torch.tensor (hidden_size, )
        sample["input_mask"] = self.all_input_mask[idx]
        sample["segment_ids"] = self.all_segment_ids[idx]
        sample["answer_id"] = self.all_answer_id[idx]
        # sample["mentions"] = self.all_mentions[idx]

        # image
        img_id = self.img_mapping[self.all_img_id[idx]]
        sample["img_feat"] = torch.from_numpy(self.img_features[img_id])

        ans_id = int(self.all_answer_id[idx])
        ans_str = self.answer_list[ans_id]
        # pos + neg samples
        pos_sample_id = self.ansid2negid[ans_id]
        sample["pos_sample"] = torch.tensor([self.neg_features[pos_sample_id]])

        # Negative example：list
        neg_ids = neg_sample_online(self.qid2negid[ans_str], self.neg_iid, self.tfidf_neg, self.negid2qid, self.max_sample_num)
        neg_ids_map = [self.neg_mapping[nid] for nid in neg_ids]  # Convert negative example str id into id of sample feature
        sample["neg_sample"] = torch.tensor([self.neg_features[nim] for nim in neg_ids_map])

        # return search results
        if self.contain_search_res:
            qids_searched = self.search_res[self.all_mentions[idx]]
            qids_searched_map = [self.neg_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor([self.neg_features[qsm] for qsm in qids_searched_map])

        return sample
