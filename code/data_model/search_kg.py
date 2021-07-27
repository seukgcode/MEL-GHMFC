"""
    search kg for candidates by name
"""
import argparse
import json
import os
from elasticsearch import Elasticsearch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_dm",
                        default="data/prepro_data/dm",
                        help="Dir to dm")
    parser.add_argument("--dir_prepro",
                        default="data/prepro_data/nel",
                        help="Dir to NEL prepro")
    parser.add_argument("--server_es",
                        default="localhost:9200",
                        help="Server ip of elastic search.")
    parser.add_argument("--num_search",
                        default=50,
                        type=int,
                        help="Number of search results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.dir_prepro, exist_ok=True)

    es = Elasticsearch(args.server_es, ignore=400)

    # build index
    index = "kvqa_nes"
    if not es.indices.exists(index):
        ne2qid = json.load(open(os.path.join(args.dir_dm, "ne2qid.json")))
        for ne, qid in ne2qid.items():
            es.index(index=index, body={'name': ne, 'qid': qid})

    # search
    dsl = lambda x: {
        "query": {
            "match": {
                "name": {
                    "fuzziness": "AUTO",
                    "query": x
                }
            }
        },
        "size": args.num_search
    }

    nel_one2one = json.load(open(os.path.join(args.dir_dm, "nel_121.json")))
    men2qids = {}

    for data in nel_one2one.values():
        result = es.search(index='names', body=dsl(data['mentions']))
        hits = result['hits']['hits']
        hits_qids = [h['_source']['qid'] for h in hits]
        men2qids[data['mentions']] = hits_qids
    json.dump(men2qids, open(os.path.join(args.dir_prepro, f"search_top{args.num_search}.json"), 'w'), indent=2)


if __name__ == '__main__':
    main()
