"""
    -----------------------------------
    MNEL predict
"""
 import json
import logging
import argparse
import torch
import pickle
import h5py
import re

from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from os.path import join, exists
from fuzzywuzzy import process


from transformers.modeling_auto import MODEL_MAPPING
from transformers import (
    ALL_PRETRAINED_MODEL_ARCHIVE_MAP,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from nel_utils import read_examples_from_input, convert_examples_to_features
from nel import NELModel

from circle_loss import cosine_similarity, dot_similarity
from metric_topk import lp_distance
from nel_train import recover_nel_args


logger = logging.getLogger(__name__)

ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)
MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

INAME_PATTERN = re.compile("/(\d+)\.")


def parse_arg():
    parser = argparse.ArgumentParser()
    # Type of Inputs
    parser.add_argument("--input_type", default="console", help="Type of input: console|file")

    # Path parameters
    parser.add_argument(
        "--path_candidates",
        default=None,
        type=str,
        help="Path to search results.",
    )
    parser.add_argument(
        "--path_ne2qid",
        default=None,
        type=str,
        help="Path to json file 'ne2qid'"
    )
    parser.add_argument(
        "--dir_neg_feat",
        default=None,
        type=str,
        help="Path to negative samples' features."
    )
    parser.add_argument(
        "--dir_img_feat",
        default=None,
        type=str,
        help="Path to image features."
    )
    parser.add_argument(
        "--dir_predict",
        default="data/predict_data/nel"
    )
    # model configs
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--max_sent_length",
        default=32,
        type=int,
        help="The maximum total input sequence length in nel model.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    # negative sample
    parser.add_argument(
        "--img_len",
        default=196,
        type=int,
        help="The number of image regions.",
    )
    parser.add_argument(
        "--dropout",
        default=0.2,
        type=float,
        help="Dropout",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hidden_size", default=512, type=int, help="Hidden size."
    )
    parser.add_argument(
        "--ff_size", default=2048, type=int, help="Feed forward size."
    )
    parser.add_argument(
        "--text_feat_size", default=768, type=int, help="Text feature's size."
    )
    parser.add_argument(
        "--img_feat_size", default=2048, type=int, help="Image feature's size."
    )
    parser.add_argument(
        "--nheaders", default=8, type=int, help="Num of attention headers."
    )
    parser.add_argument(
        "--num_attn_layers", default=1, type=int, help="Num of attention layers"
    )
    parser.add_argument(
        "--output_size", default=768, type=int, help="Output size."
    )

    # loss
    parser.add_argument(
        "--loss_function", default="circle", type=str, help="Loss: triplet | circle"
    )

    parser.add_argument(
        "--loss_margin", default=0.25, type=float, help="margin of circle loss."
    )
    parser.add_argument(
        "--similarity", default='cos', type=str, help="Similarity"
    )
    # triplet loss
    parser.add_argument(
        "--loss_p", default=2, type=int, help="The norm degree for pairwise distance."
    )
    # circle loss
    parser.add_argument(
        "--loss_scale", default=32, type=int, help="Scale of circle loss."
    )
    parser.add_argument(
        "--feat_cate", default='wp', help="word (w), phrase(p) and sentence(s), default: wp"
    )

    # cross validation
    parser.add_argument("--do_cross", action="store_true", help="Whether to run cross validation.")
    parser.add_argument("--folds", default=5, type=int, help="Num of folds in cross validation.")
    parser.add_argument("--cross_arg", default=None, type=str, help="Arg to valid in cross validation.")
    parser.add_argument("--mode_pace", default='add', help="Pace mode of arg in validation: add|multiple")
    parser.add_argument("--pace", default=0.1, type=float, help="Pace of arg in cross validation.")
    parser.add_argument("--ub_arg", default=1, type=float, help="The upper bound of arg in cross validation.")

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--multi_gpus", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="id of gpu")
    parser.add_argument("--single_gpu", action="store_true", help="is single gpu?")

    return parser.parse_args()


def _get_input_from_console():
    """
        Get input from the console
        ------------------------------------------
        Args:
        Returns:
    """
    texts = []
    print("Please input sentences, one sentence a time: ")
    inp = input()
    while inp.strip() != '':
        texts.append(inp.strip())
        inp = input()

    mentions = []
    print("Please input mentions, one mention a time: ")
    for i in range(len(texts)):
        inp = input()
        mentions.append(inp.strip())

    imgs = []
    print("Please input image ids, one id a time: ")
    for i in range(len(texts)):
        inp = input()
        imgs.append(inp.strip())

    return texts, mentions, imgs


def set_device(args):
    # Setup CUDA, GPU & distributed training
    if args.multi_gpus:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda", args.gpu_id)
        else:
            device = torch.device("cpu")
        args.n_gpu = 1
    return device


def _get_input_from_file():
    return [], [], []


def get_inputs(args):
    """
        Receive input
        ------------------------------------------
        Args:
        Returns:
    """
    if args.input_type.lower() == 'console':
        return _get_input_from_console()
    else:
        return _get_input_from_file()


def similarity_link(args, query, candidates_list):
    """
        Use query to sort candidates and return a list of candidates sorted by similarity
        ------------------------------------------
        Args:
        Returns:
    """
    ranks = []
    similarities = []
    cal_sim = cosine_similarity if args.similarity == 'cos' else dot_similarity

    for i, candidates in enumerate(candidates_list):
        query_input = query[i].view(1, *query[i].size())
        cand_input = candidates.view(1, *candidates.size())

        sim = cal_sim(query_input, cand_input)  # 1, n_cands
        sim.squeeze_()

        sim_score, sim_rank = sim.sort(dim=-1, descending=True)  # The greater the similarity, the higher the ranking

        ranks.append(sim_rank)

        # Calculate similarity
        # sim_approx = F.softmax(sim_score, dim=-1)
        sim_approx = sim_score
        similarities.append(sim_approx)

    return ranks, similarities


def main():
    args = parse_arg()
    batch_size = 8
    args.device = set_device(args)
    os.makedirs(args.dir_predict, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    args.model_type = args.model_type.lower()
    # logger.info("Training/evaluation parameters %s", args)
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=None,
        **tokenizer_args,
    )

    # prepare candidates
    # Obtain input data and process it into features
    # texts, mentions, imgs = get_inputs(args)
    texts = [
        "Osorio during his time at New York Red Bulls",
        "Jinnah announcing the creation of Pakistan over All India Radio on 3 June 1947.",
        "Hendricks in March 2006 at the Los Angeles Wizard World Comic Con",
        "Bendix as Riley with Sterling Holloway, 1957",
        "Coleman at the 72nd Annual Peabody Awards Luncheon",
    ]
    mentions = [
        "Osorio",
        "Jinnah",
        "Hendricks",
        "Bendix",
        "Coleman"
    ]
    imgs = [
        "31364",
        "20210",
        "18902",
        "46541",
        "14653"
    ]
    examples = read_examples_from_input(texts=texts, imgs=imgs)
    features = convert_examples_to_features(examples,  None, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    # Use mentions to extract candidates
    ne2qid = json.load(open(args.path_ne2qid))
    ne_list = list(ne2qid.keys())  # Read all entity names in the knowledge base
    # qid2ne = {qid: ne for qid, ne in ne2qid.items()}

    candidate_qids_list = []  # record id of candidates
    for mention in mentions:
        matches = process.extract(mention, ne_list)
        matches_qids = [ne2qid[m[0]] for m in matches]
        candidate_qids_list.append(matches_qids)

    # Convert the qid of candidates into features
    neg_list = json.load(open(join(args.dir_neg_feat, "neg_list.json")))
    qid2negID = {qid: i for i, qid in enumerate(neg_list)}  # Read the order of candidates characteristics

    neg_feat_h5 = h5py.File(join(args.dir_neg_feat, "neg_feats.h5"), 'r')
    neg_features = neg_feat_h5.get("features")  # read features of candidates

    candidate_tensors_list = []  # Each qid corresponds to several candidates
    for cql in candidate_qids_list:  # Convert each group of candiates into feature form
        negIDs_map = [qid2negID[nid] for nid in cql]  # Convert qid to candidates id
        candidate_tensors_list.append(torch.tensor([neg_features[nim] for nim in negIDs_map]))

    # prepare image features
    img_list = json.load(open(join(args.dir_img_feat, "img_list.json")))
    img_list = [INAME_PATTERN.findall(iname)[0] for iname in img_list]
    id2img_feat_id = {iname: i for i, iname in enumerate(img_list)}

    img_feat_h5 = h5py.File(join(args.dir_img_feat, "img_features.h5"), 'r')
    img_features = img_feat_h5.get("features")

    img_feats_tensor = torch.tensor([img_features[id2img_feat_id[imgID]] for imgID in imgs])  # obtain features

    # Organize the data required by the model into dataloader
    pred_dateset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, img_feats_tensor)
    pred_dataloader = DataLoader(pred_dateset, batch_size=batch_size)

    # Start to predict
    #      # First, load all models
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=None,
    )

    # Restore history args
    path_args_history = join(args.model_name_or_path, "training_args.bin")
    args_his = torch.load(path_args_history)
    recover_nel_args(args, args_his)

    nel_model = NELModel(args)
    path_nel_state = join(args.model_name_or_path, "nel_model.pkl")
    logger.info(f"Load nel model state dict from {path_nel_state}")
    nel_model.load_state_dict(torch.load(path_nel_state))

    model.to(args.device)
    nel_model.to(args.device)

    linked_ranks = []
    linked_similarities = []
    attn_list = []
    with torch.no_grad():
        for step, batch in enumerate(pred_dataloader):
            for k in range(len(batch)):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            bert_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]
            }

            bert_out = model(**bert_inputs)

            nel_inputs = {
                "bert_feat": bert_out[0],
                "img": batch[3],
                "bert_mask": batch[1],
            }
            query, attn = nel_model(**nel_inputs)

            query = query.to(torch.device("cpu"))

            #attn value, including (w, p, s) attention value
            
            attn_list.extend(attn)  # extend is to connect each batch

            candidate_tensors = candidate_tensors_list[step * batch_size: (step + 1) * batch_size]

            # Sort candidates by similarity, ranks: the index sorted by similarity from largest to smallest; similarities: sort by similarity from largest to smallest
            ranks, similarities = similarity_link(args, query, candidate_tensors)
            linked_ranks.extend(ranks)
            linked_similarities.extend(similarities)

    rankedQids_list = []
    for i in range(len(linked_ranks)):  # for each input
        candidate_qids = candidate_qids_list[i]  # Find the previously saved candidates
        ranked_qids = []
        for r in linked_ranks[i]:
            ranked_qids.append(candidate_qids[r])

        rankedQids_list.append(list(zip(ranked_qids, linked_similarities[i])))

    predict_res = {
        "rank": rankedQids_list,
        "attention": attn_list,
        "query": query,
        "candidate_tensor": candidate_tensors_list,
        "candidate_qid": candidate_qids_list
    }

    path_out = join(args.dir_predict, f"res{len(rankedQids_list)}.pkl")
    pickle.dump(predict_res, open(path_out, "wb"))



if __name__ == "__main__":
    main()
