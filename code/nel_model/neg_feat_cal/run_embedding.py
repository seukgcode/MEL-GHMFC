from transformers import BertTokenizer,BertModel
import json
import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import h5py
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
        """
        self.guid = guid
        self.words = words


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def read_examples_from_file(args):
    file_path = args.path_qid2abs
    examples = []
    data = json.load(open(file_path, encoding="utf-8"))
    keys_ordered = list(data.keys())
    json.dump(keys_ordered, open(os.path.join(args.dir_output, "neg_list.json"), 'w'), indent=2)
    for key in keys_ordered:
       examples.append(InputExample(guid=key, words=data[key]))
    return examples


def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        sequence_dict = tokenizer.encode_plus(example.words, max_length=max_seq_length, pad_to_max_length=True)
        input_ids = sequence_dict['input_ids']
        input_mask = sequence_dict['attention_mask']
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask)
        )
    return features


def load_and_cache_examples(args, tokenizer, mode):

    # logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args)
    features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer,
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask)
    return dataset


def evaluate(args, model, tokenizer, mode):
    eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 1)
    
    model.eval()
    sen_embeddings = torch.FloatTensor(len(eval_dataset), 768)
    device = torch.device("cpu")
    count = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states.to(device)
            last_hidden_states = last_hidden_states.squeeze()[0]
            sen_embeddings[count] = last_hidden_states
            count += 1
    return sen_embeddings


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--path_qid2abs",
        default=None,
        type=str,
        required=True,
        help="The input data.",
    )
    parser.add_argument(
        "--dir_output",
        default=None,
        type=str,
        required=True,
        help="The output dir.",
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
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
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
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained('bert-base-cased')
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    model.to(device)

    logger.info("Training/evaluation parameters %s", args)

    sen_embeddings = evaluate(args, model, tokenizer, mode='eval')
    h5_file = h5py.File(os.path.join(args.dir_output, "neg_feats.h5"), 'w')
    h5_file.create_dataset("features", data=sen_embeddings.numpy())

    logger.info("Done")


if __name__ == "__main__":
    main()
