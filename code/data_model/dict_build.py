"""
    construct all dictionaries for EDA
"""
import os
import json
import argparse
import pandas as pd
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_nes",
                        default="data/input_data/KVQA/KGfacts/Qid-NamedEntityMapping.csv",
                        help="Path to file 'Qid-NamedEntityMapping.csv'")
    parser.add_argument("--path_kgfacts",
                        default="data/input_data/KVQA/KGfacts/KGfacts-CloseWorld.csv",
                        help="Path to file 'Qid-NamedEntityMapping.csv")
    parser.add_argument("--path_dataset",
                        default="data/input_data/KVQA_sup/dataset_clear.json",
                        help="Path to file 'dataset.json'")
    parser.add_argument("--path_prepro",
                        default="data/prepro_data/dm",
                        help="Path to prepro dm")

    return parser.parse_args()


# ne2id and id2ne
def mapping_ne_qid(args):
    if os.path.exists(os.path.join(args.path_prepro, "ne2qid.json")):
        ne2qid = json.load(open(os.path.join(args.path_prepro, "ne2qid.json")))
    else:
        with open(args.path_nes, 'r+') as f:
            nes_text = f.read()
            if nes_text[:3] != 'Qid':
                f.seek(0, 0)
                f.write("Qid	Name\n" + nes_text)
        ne2qid = pd.read_csv(args.path_nes, sep='	')
        ne2qid["Name"] = [eval(x).decode(encoding="utf-8") for x in ne2qid["Name"]]
        ne2qid = dict(zip(ne2qid["Name"], ne2qid['Qid']))
        json.dump(ne2qid, open(os.path.join(args.path_prepro, "ne2qid.json"), 'w'), indent=4)

    qid2ne = {v: k for k, v in ne2qid.items()}
    json.dump(qid2ne, open(os.path.join(args.path_prepro, "qid2ne.json"), 'w'), indent=4)
    return ne2qid, qid2ne


def _fact_split(line):
    sps = line.split(',')
    if len(sps) == 3:
        return sps
    new_sps = []
    i = 0
    while i < len(sps):
        tmp_str = sps[i]
        while i + 1 < len(sps) and sps[i+1][0] == ' ':
            tmp_str += sps[i+1]
            i += 1
        new_sps.append(tmp_str)
        i += 1
    return new_sps


def mapping_facts_qid(args, qid2ne):
    """
        The facts related to entities are extracted and mapped
        ------------------------------------------
        Args:
        Returns:
    """
    heads = []
    relations = []
    tails = []
    with open(args.path_kgfacts) as f:
        for line in f.readlines():
            sps = _fact_split(line)
            h, r, t = sps
            heads.append(h)
            relations.append(r)
            tails.append(t)

    qid2facts = defaultdict(dict)  # id : relation: value

    for i, head in enumerate(heads):
        if head in qid2ne:
            r, t = relations[i], tails[i]
            qid2facts[head][r] = qid2facts[head].get(r, [])
            qid2facts[head][r].append(t.strip())

    path_store = os.path.join(args.path_prepro, "qid2facts.json")
    json.dump(qid2facts, open(path_store, 'w'), indent=4)

    return qid2facts


def mapping_abs_qid(args, qid2facts, qid2ne):
    """
        Turn facts into a summary
        ------------------------------------------
        Args:
        Returns:
    """
    qid2abs = dict()
    for qid, facts in qid2facts.items():
        abst = []

        # sex
        if 'sex' in facts:
            abst.append("Sex: " + facts['sex'][0])

        # birth info
        birth_date = facts.get('date of birth', [''])[0][:4]
        birth_place = facts.get('place of birth', [''])[0]
        death_date = facts.get('date of death', [''])[0][:4]
        death_place = facts.get('place of death', [''])[0]
        if birth_date:
            birth_info = "Birth: " + birth_date
            if birth_place:
                birth_info = f"{birth_info}, {birth_place}"
            if death_date:
                birth_info = f"{birth_info}. Death: {death_date}"
                if death_place:
                    birth_info = f"{birth_info}, {death_place}"
            abst.append(birth_info)
        if "religion" in facts:
            abst.append("Religion: " + facts['religion'][0])

        abst.append('Occupation: ' + ", ".join(facts['occupation']))
        if "work started" in facts:
            abst.append("Work starts: " + facts['work started'][0][:4])

        if "spouse" in facts:
            spouse = []
            for s in facts['spouse']:
                if s in qid2ne:
                    spouse.append(qid2ne[s])
                else:
                    if s[0] not in ['Q', 't'] or len(set(s) & set("0123456789")) < 1:
                        spouse.append(s)
            if len(spouse) > 0:
                abst.append('Spouse: ' + ', '.join(spouse))

        # language
        if 'knows language' in facts or 'native language' in facts:
            langs = facts.get('native language', []) + facts.get('knows language', [])
            langs = "Languages: " + ', '.join(list(set(langs)))
            abst.append(langs)

        if "alma mater" in facts:
            abst.append("Alma mater: " + ', '.join(facts['alma mater']))

        if "nick name" in facts:
            abst.append("Nick name: " + ', '.join(facts['nick name']))

        qid2abs[qid] = ". ".join(abst) + '.'

    path_store = os.path.join(args.path_prepro, "qid2abs.json")
    json.dump(qid2abs, open(path_store, 'w'), indent=4)
    return qid2abs


def main():
    args = parse_args()
    os.makedirs(args.path_prepro, exist_ok=True)

    # Handle the mapping of named entities and database IDs
    ne2qid, qid2ne = mapping_ne_qid(args)

    # Handle the mapping of database ID and knowledge
    qid2facts = mapping_facts_qid(args, qid2ne)

    # Handle the mapping of database ID and digest
    mapping_abs_qid(args, qid2facts, qid2ne)

    # path_ordered_qids = os.path.join(args.path_prepro, "qids_ordered.json")
    # json.dump(list(qid2ne.keys()), open(path_ordered_qids, "w"), indent=2)


if __name__ == "__main__":
    main()
