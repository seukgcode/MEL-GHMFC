"""
    search kg for candidates by name
"""
import argparse
import json
import os
from os.path import join, exists
from fuzzywuzzy import process
from multiprocessing.pool import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_dm",
                        default="data/prepro_data/dm",
                        help="Dir to dm")
    parser.add_argument("--dir_output",
                        default="data/output_data/ner",
                        help="Dir to output")
    parser.add_argument("--path_input",
                        default=None,
                        required=True,
                        help="Dir to output")
    parser.add_argument("--num_search",
                        default=100,
                        type=int,
                        help="Number of search results")
    return parser.parse_args()


def run(m, ne_list, num_search):
    return process.extract(m, ne_list, limit=num_search)


def run_list(list_m, ne_list, num_search):
    res = []
    for m in list_m:
        res.append(run(m, ne_list, num_search))
    return res


def main():
    args = parse_args()
    print(f"Search candidates with fuzzywuzzy (num: {args.num_search}).")
    ne2qid = json.load(open(os.path.join(args.dir_dm, "ne2qid.json")))
    res_ner = json.load(open(args.path_input))

    path_search_tmp = join(args.dir_output, f"search_ner{args.num_search}.json")

    ne_list = list(ne2qid.keys())

    pool = Pool(40)
    search_res = [pool.apply_async(run_list, (m_list, ne_list, args.num_search,)) for m_list in res_ner]
    search_tmp = [sr.get() for sr in search_res]
    json.dump(search_tmp, open(path_search_tmp, 'w'), indent=2)


if __name__ == '__main__':
    main()