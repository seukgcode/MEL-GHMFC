"""
    KVQA data split
"""
import json
import os
import argparse
import random


def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument("--dir_prepro",
                        default="data/prepro_data/nel",
                        help="It will be used as output path.")
    parser.add_argument("--path_dataset",
                        default="data/input_data/KVQA_sup/dataset_clear.json",
                        help="Path to dataset KVQA.")
    parser.add_argument("--path_splits",
                        default="data/prepro_data/nel/splits.json",
                        help="Path to splits file")
    parser.add_argument("--split_mode", default='721', help="The mode of splitting")
    parser.add_argument("--shuffle", action="store_true", help="Whether shuffle the data or not")
    parser.add_argument("--seed", default=42, help="shuffle seeds")

    return parser.parse_args()


def split(ndata, mode='721'):
    N = len(ndata)

    pro_tr = int(mode[:1]) if mode[:2] is not '10' else 10
    pro_te = int(mode[1:2]) if pro_tr is not 0 and mode[1:3] is not '10' else 10
    pro_de = int(mode[2:]) if pro_tr + pro_te != 10 else 0

    assert pro_tr + pro_te + pro_de == 10

    n_tr = N * pro_tr // 10
    n_te = N * pro_te // 10

    return {
        'train': ndata[: n_tr],
        'test': ndata[n_tr: n_tr + n_te],
        'dev': ndata[n_tr + n_te:]
    }


def main():
    args = parse_args()
    print(json.dumps(args.__dict__, indent=2))
    random.seed = args.seed

    dataset = json.load(open(args.path_dataset))
    keys = list(dataset.keys())
    if args.shuffle:
        random.shuffle(keys)

    ndata = split(keys, mode=args.split_mode)
    json.dump(ndata, open(args.path_splits, 'w'), indent=2)
    print(len(ndata["train"]), len(ndata["test"]), len(ndata["dev"]))


if __name__ == '__main__':
    main()
