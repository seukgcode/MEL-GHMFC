"""
    -----------------------------------
   Pretreatment operation of negative sampling on kg：
    1. Calculate TF-IDF for each document word
    2. Find the words that best represent the document
    3. Add according to inverted index
"""
import argparse
import os
import json
import pickle
from nltk import word_tokenize
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import random
import logging
logger = logging.getLogger(__name__)


STOP_WORDS = {",", ".", "!", ":", "'", "\"", ";", "/", "\\", "Sex", "Name", "Nick", "Occupation", "Birth", "Languages",
              "Religion", "Alma", "mater", ""}


def parse_args():
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument("--path_qid2abs", default="data/prepro_data/dm/qid2abs.json", help="Path to id2abs.json")
    parser.add_argument("--path_keys_ordered", default="data/prepro_data/dm/qids_ordered.json", help="Path to qids_ordered.json")
    parser.add_argument("--dir_cache", default="data/prepro_data/dm/neg_cache", help="Directory to data model prepro")

    parser.add_argument("--overwrite_cache", action="store_true", help="Should overwrite?")
    parser.add_argument("--max_sample_num",
                        default=10,
                        type=int,
                        help="Directory to data model prepro")

    parser.add_argument("--rebuild_dict",
                        action="store_true",
                        help="Should rebuild the dict if it exists?")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    return parser.parse_args()


def build_dict(sent_list):
    """
        -----------------------------------
        build tf-idf dict
    """
    toks_list = []
    for sent in sent_list:
        toks = word_tokenize(sent)
        toks_list.append([tok for tok in toks if tok not in STOP_WORDS])
    dct = Dictionary(toks_list)
    return dct, toks_list


def build_inverted_index(nwords, corpus):
    """
        Building inverted index of word sample ID
1. Firstly, a real inverted index is constructed, in which the total number of words (iid) in each sentence is recorded;
2. Replace the contents of inverted index into NEG_iid)：
        scale  0 1 2 3 4 5 ... N
        cor_id 0 0 0 1 2 2 ... M
        N >> M
        ------------------------------------------
        Args:
        Returns:
    """
    inverted_index = [[] for _ in range(nwords)]
    for i, cor in enumerate(corpus):
        for tok_freq in cor:
            tok, freq = tok_freq
            inverted_index[tok].append([i, freq])

    neg_iid = []
    for words_iid in inverted_index:
        neg_list = []
        for id_freq in words_iid:
            idx, freq = id_freq
            neg_list.extend([idx] * freq)
        neg_iid.append(neg_list)

    return inverted_index, neg_iid


def build_tfidf(corpus, base=100):
    """
        Constructing TF-IDF list of id2words
        scale  0 1 2 3 4 5 ... N
        word   0 0 0 1 2 2 ... M
        N >> M
        ------------------------------------------
        Args:
        Returns:
    """
    model = TfidfModel(corpus)
    tfidf = [model[cor] for cor in corpus]
    tfidf_neg = []
    for toks_list in tfidf:
        neg_list = []
        for tok_val in toks_list:
            tok_id, val = tok_val
            neg_list.extend([tok_id] * int(val * base))
        tfidf_neg.append(neg_list)
    return tfidf, tfidf_neg


def negative_sample(max_sample_num, neg_iid, tfidf_neg):
    """
        negative sample
        ------------------------------------------
        Args:
        Returns:
    """
    N = len(tfidf_neg)
    samples = []
    all_cand_ids = list(range(N))
    for i in range(N):  # the i-th example
        cands = set()
        if not tfidf_neg[i]:
            while len(cands) < max_sample_num:
                cand = random.choice(all_cand_ids)
                if cand != i:
                    cands.add(cand)
        else:
            while len(cands) < max_sample_num:
                rand_word = random.choice(tfidf_neg[i])
                cand = random.choice(neg_iid[rand_word])
                if cand != i:
                    cands.add(cand)
        samples.append(list(cands))

    return samples


def main():
    args = parse_args()
    random.seed(args.seed)

    qid2abs = json.load(open(args.path_qid2abs))
    keys_ordered = list(qid2abs.keys())

    path_dict = os.path.join(args.dir_cache, "tfidf_dict.pkl")
    path_neg = os.path.join(args.dir_cache, "neg.json")
    path_samples = os.path.join(args.dir_cache, f"samples_{args.max_sample_num}.json")

    if os.path.exists(args.dir_cache) and not args.overwrite_cache:
        # dictionary = pickle.load(open(path_dict))
        config = pickle.load(open(path_neg))
        neg_iid = config["neg_iid"]
        tfidf_neg = config["tfidf_neg"]
        keys_ordered = config["keys_ordered"]
    else:
        os.makedirs(args.dir_cache, exist_ok=True)
        # construct dicts
        dictionary, toksList_ordered = build_dict([qid2abs[k] for k in keys_ordered])
        nwords = len(dictionary)

        # Turn each sentence into the form of ID-freq
        corpus = [dictionary.doc2bow(toks_list) for toks_list in toksList_ordered]  # sent 2 ids

        _, neg_iid = build_inverted_index(nwords, corpus)
        _, tfidf_neg = build_tfidf(corpus)

        pickle.dump(dictionary, open(path_dict, "wb"))
        json.dump({
            "keys_ordered": keys_ordered,
            "neg_iid": neg_iid,
            "tfidf_neg": tfidf_neg,
            "corpus": corpus,
        }, open(path_neg, 'w'), indent=2)

    logger.info(f"Now, negative sampling, max_sample_num: {args.max_sample_num}")
    samples_ordered = negative_sample(args.max_sample_num, neg_iid, tfidf_neg)
    samples_ordered = [[keys_ordered[idx] for idx in sample] for sample in samples_ordered]
    samples = dict(zip(keys_ordered, samples_ordered))
    json.dump(keys_ordered, open(args.path_keys_ordered, 'w'), indent=2)
    json.dump(samples, open(path_samples, "w"), indent=2)
    logger.info(f"Sampling done!")


if __name__ == '__main__':
    main()
