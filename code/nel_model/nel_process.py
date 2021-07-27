"""
    -----------------------------------
    KVQA data split
    -----------------------------------
    Increase cross-validation split

"""
import json
import os
import argparse
import random
from os.path import join


def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument("--dir_prepro",
                        default="data/prepro_data/nel",
                        help="It will be used as output path.")
    parser.add_argument("--path_dataset",
                        required=True,
                        default=None,
                        help="Path to dataset KVQA.")
    parser.add_argument("--split_mode", default='721', help="The mode of splitting")
    parser.add_argument("--shuffle", action="store_true", help="Whether shuffle the data or not")
    parser.add_argument("--seed", default=42, type=int, help="shuffle seeds")
    parser.add_argument("--mode", default="normal", help="normal | cross")
    parser.add_argument("--folds", default=5, type=int, help="fold num of cross validation.")

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


def cross_split(ndata, folds):
    N = len(ndata)
    nfold = N // folds

    data_folds = [[] for _ in range(folds)]
    for f in range(folds):
        if f != folds - 1:
            fdata = ndata[f * nfold: (f+1)*nfold]
        else:
            fdata = ndata[f * nfold:]

        data_folds[f] = fdata
    return data_folds


def main():
    args = parse_args()
    print(json.dumps(args.__dict__, indent=2))
    random.seed = args.seed

    dataset = json.load(open(args.path_dataset))
    keys = sorted(list(dataset.keys()))
    if args.shuffle:
        random.shuffle(keys)

    if args.mode == "normal":
        ndata = split(keys, mode=args.split_mode)

        for cls in ['train', 'test', 'dev']:
            path_file = join(args.dir_prepro, f"{cls}.json")
            part_data = {k: dataset[k] for k in ndata[cls]}

            json.dump(part_data, open(path_file, 'w'), indent=2)
        print(len(ndata["train"]), len(ndata["test"]), len(ndata["dev"]))

    if args.mode == "cross":
        data_folds = cross_split(keys, folds=args.folds)
        dir_cross = join(args.dir_prepro, "cross")
        os.makedirs(dir_cross, exist_ok=True)
        for f in range(args.folds):
            path_train = join(dir_cross, f"train{f}.json")
            path_dev = join(dir_cross, f"dev{f}.json")

            dev_data = {k: dataset[k] for k in data_folds[f]}
            json.dump(dev_data, open(path_dev, 'w'), indent=2)

            train_data = []
            for i, data in enumerate(data_folds):
                if i == f:
                    continue
                train_data += data_folds[i]
            train_data = {k: dataset[k] for k in train_data}
            json.dump(train_data, open(path_train, 'w'), indent=2)


if __name__ == '__main__':
    main()
