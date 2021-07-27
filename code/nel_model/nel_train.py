import os
import json
import logging
import argparse
import numpy as np
import torch
import random
import pickle
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm, trange
from os.path import join, exists
from glob import glob

from transformers.modeling_auto import MODEL_MAPPING
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from nel_utils import read_examples_from_file, convert_examples_to_features
from nel_dataset import NELDataset
from nel import NELModel
from metric_topk import cal_top_k
from time import time


logger = logging.getLogger(__name__)

#ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)

MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def parse_arg():
    parser = argparse.ArgumentParser()

    # Path parameters

    parser.add_argument(
        "--dir_prepro",
        default=None,
        type=str,
        required=True,
        help="The prepro data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--path_ans_list",
        default=None,
        type=str,
        help="Path to ordered answer list.",
    )
    parser.add_argument(
        "--path_candidates",
        default=None,
        type=str,
        help="Path to search results.",
    )
    parser.add_argument(
        "--path_neg_config",
        default="data/prepro_data/dm/neg_cache/neg.json",
        type=str,
        help="Path to neg.json",
    )
    parser.add_argument(
        "--dir_img_feat",
        default=None,
        type=str,
        help="Path to image features."
    )
    parser.add_argument(
        "--dir_neg_feat",
        default=None,
        type=str,
        help="Path to negative samples' features."
    )
    parser.add_argument(
        "--dir_eval",
        default=None,
        type=str,
        help="Path to eval model",
    )
    parser.add_argument(
        "--dir_output",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
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
        default="bert-base-cased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_sent_length",
        default=32,
        type=int,
        help="The maximum total input sequence length in nel model.",
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
        "--neg_sample_num",
        default=3,
        type=int,
        help="The num of negatives sample.",
    )

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
    parser.add_argument(
        "--feat_cate", default='wp', help="word (w), phrase(p) and sentence(s), default: wp"
    )
    parser.add_argument(
        "--rnn_layers", default=2, type=int, help="rnn_layers."
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

    # cross validation
    parser.add_argument("--do_cross", action="store_true", help="Whether to run cross validation.")
    parser.add_argument("--folds", default=5, type=int, help="Num of folds in cross validation.")
    parser.add_argument("--cross_arg", default=None, type=str, help="Arg to valid in cross validation.")
    parser.add_argument("--mode_pace", default='add', help="Pace mode of arg in validation: add|multiple")
    parser.add_argument("--pace", default=0.1, type=float, help="Pace of arg in cross validation.")
    parser.add_argument("--ub_arg", default=1, type=float, help="The upper bound of arg in cross validation.")

    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--do_cross_eval", action="store_true", help="Whether to run predictions on the cross set.")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
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

    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--multi_gpus", action="store_true", help="Using multiple CUDAs when available")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--test_model_dir",
        default=None,
        type=str,
        help="The test model dir.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--gpu_id", type=int, default=0, help="id of gpu")
    parser.add_argument("--single_gpu", action="store_true", help="is single gpu?")

    return parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, nel_model, answer_list, tokenizer, fold=""):
    """

    :param args:
    :param train_dataset:
    :param model:
    :param tokenizer:
    :param labels:
    :param pad_token_label_id:
    :return:
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)  # 1 iter

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)] + \
        [p for n, p in nel_model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)] + \
        [p for n, p in nel_model.named_parameters() if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {
            "params": params_decay,
            "weight_decay": args.weight_decay,
        },
        {"params": params_nodecay, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        nel_model = torch.nn.DataParallel(nel_model)
        print("train", type(model), model.device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    nel_model.zero_grad()

    set_seed(args)  # Added here for reproductibility

    epoch_start_time = time()
    step_start_time = None
    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        if epoch == epochs_trained:
            logger.info(f"  Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin.")
        else:
            logger.info(f"  Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin ({(time() - epoch_start_time) / (epoch - epochs_trained):2f}s/epoch).")
        epoch_iterator = train_dataloader
        num_steps = len(train_dataloader)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            nel_model.train()

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            bert_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["input_mask"]}
            bert_out = model(**bert_inputs)

            nel_inputs = {"bert_feat": bert_out[0],
                          "img": batch["img_feat"],
                          "bert_mask": batch["input_mask"],
                          "pos_feats": batch["pos_sample"],
                          "neg_feats": batch["neg_sample"]}

            outputs = nel_model(**nel_inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(nel_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                nel_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    if step_start_time is None:
                        step_start_time = time()
                        logger.info(f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch+1}: {step + 1}/{num_steps}")
                    else:
                        log_tim = (time() - step_start_time)
                        logger.info(
                            f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch+1}: {step + 1}/{num_steps} ({log_tim:.2f}s/50step)")
                        step_start_time = time()
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.dir_output, f"checkpoint{fold}-{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(nel_model.state_dict(), join(output_dir, f'nel_model.pkl'))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):
                        logger.info(f"********Global step {global_step}, DEV Evaluation********", )
                        results, _ = evaluate(args, model, nel_model, answer_list, tokenizer, mode=f"dev{fold}")[:2]
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                        json.dump(results, open(os.path.join(output_dir, "dev_results.json"), 'w'), indent=2)

            if 0 < args.max_steps < global_step:
                break

        if 0 < args.max_steps < global_step:
            # train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, nel_model, answer_list, tokenizer, mode, prefix=""):
    time_eval_beg = time()

    eval_dataset, guks = load_and_cache_examples(args, tokenizer, answer_list, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and type(model) not in {"torch.nn.parallel.data_parallel.DataParallel"}:
        model = torch.nn.DataParallel(model)
        nel_model = torch.nn.DataParallel(nel_model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    nel_model.eval()

    all_ranks = []
    all_sim_p = None
    all_sim_n = None

    time_eval_rcd = time()
    nsteps = len(eval_dataloader)
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)
            bert_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["input_mask"]}
            bert_out = model(**bert_inputs)

            nel_inputs = {"bert_feat": bert_out[0],
                          "img": batch["img_feat"],
                          "bert_mask": batch["input_mask"],
                          "pos_feats": batch["pos_sample"],
                          "neg_feats": batch["neg_sample"]
                          }

            outputs = nel_model(**nel_inputs)
            tmp_eval_loss, query = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss

            pos_feat_trans = nel_model.trans(batch["pos_sample"])
            neg_feat_trans = nel_model.trans(batch["search_res"])

            rank_list, sim_p, sim_n = cal_top_k(args, query, pos_feat_trans, neg_feat_trans)

            if all_sim_p is None:
                all_sim_p = sim_p
            else:
                all_sim_p = np.append(all_sim_p, sim_p, axis=0)

            if all_sim_n is None:
                all_sim_n = sim_n
            else:
                all_sim_n = np.append(all_sim_n, sim_n, axis=0)

            all_ranks.extend(rank_list)

            nb_eval_steps += 1

            if (i + 1) % 100 == 0:
                print(f"{mode}: {i + 1}/{nsteps}, loss: {tmp_eval_loss}, {time()-time_eval_rcd:.2f}s/100steps")
                time_eval_rcd = time()

    eval_loss = eval_loss.item() / nb_eval_steps

    all_ranks = np.array(all_ranks)
    results = {
        "mean_rank": sum(all_ranks) / len(all_ranks) + 1,
        "top1": int(sum(all_ranks <= 1)),
        "top3": int(sum(all_ranks <= 3)),
        "top5": int(sum(all_ranks <= 5)),
        "top10": int(sum(all_ranks <= 10)),
        "top20": int(sum(all_ranks <= 20)),
        "top30": int(sum(all_ranks <= 30)),
        "top40": int(sum(all_ranks <= 40)),
        "top50": int(sum(all_ranks <= 50)),
        "all": len(all_ranks),
        "loss": float(eval_loss)
    }

    logger.info("***** Eval results %s *****", prefix)
    logger.info(f"  eval loss: {eval_loss}")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    logger.info(f"Eval time: {time() - time_eval_beg:2f}")

    return results, eval_loss, all_ranks, all_sim_p, all_sim_n, guks


def load_and_cache_examples(args, tokenizer, answer_list, mode):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.dir_prepro,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    guks = []
    if mode != 'test' and os.path.exists(cached_features_file) and not args.overwrite_cache and not args.do_cross:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.dir_prepro)

        examples = read_examples_from_file(args.dir_prepro, mode)
        guks = [ex.guk for ex in examples]

        features = convert_examples_to_features(
            examples,
            answer_list,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            # pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0] and not args.do_cross:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_answer_id = [f.answer_id for f in features]
    all_img_id = [f.img_id for f in features]
    all_mentions = [f.mentions for f in features]

    contain_search_res = False if mode is "train" else True

    dataset = NELDataset(args,
                         all_input_ids,
                         all_input_mask,
                         all_segment_ids,
                         all_answer_id,
                         all_img_id,
                         all_mentions,
                         answer_list,
                         contain_search_res)
    return dataset, guks


def recover_nel_args(args, args_his):

    args.neg_sample_num = args_his.neg_sample_num
    args.dropout = args_his.dropout

    args.hidden_size = args_his.hidden_size
    args.ff_size = args_his.ff_size
    args.nheaders = args_his.nheaders
    args.num_attn_layers = args_his.num_attn_layers
    args.output_size = args_his.output_size
    args.text_feat_size = args_his.text_feat_size
    args.img_feat_size = args_his.img_feat_size
    args.feat_cate = args_his.feat_cate
    args.rnn_layers = args_his.rnn_layers

    args.loss_scale = args_his.loss_scale
    args.loss_margin = args_his.loss_margin
    args.loss_function = args_his.loss_function
    args.similarity = args_his.similarity


def main():
    args = parse_arg()
    if (
        os.path.exists(args.dir_output)
        and os.listdir(args.dir_output)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.dir_output
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.multi_gpus:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        if torch.cuda.is_available() and not args.no_cuda:
            device = torch.device("cuda", args.gpu_id)
        else:
            device = torch.device("cpu")
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    answer_list = json.load(open(args.path_ans_list))
    args.model_type = args.model_type.lower()

    # logger.info("Training/evaluation parameters %s", args)
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)

    if args.do_cross and (args.do_train or args.do_eval or args.do_predict):
        raise ValueError("You shouldn't do eval or predict or train When do cross")

    # Training
    if args.do_train:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
            **tokenizer_args,
        )
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # Initialize the nel model
        nel_model = NELModel(args)
        path_nel_state = join(args.model_name_or_path, "nel_model.pkl")
        if exists(path_nel_state):
            logger.info(f"Load nel model state dict from {path_nel_state}")
            nel_model.load_state_dict(torch.load(path_nel_state, map_location=lambda storage, loc: storage))


            path_args_history = join(args.model_name_or_path, "training_args.bin")
            args_his = torch.load(path_args_history)
            recover_nel_args(args, args_his)
        else:
            for p in nel_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        logger.info("Device: %s", args.device)
        model.to(args.device)
        nel_model.to(args.device)

        train_dataset, _ = load_and_cache_examples(args, tokenizer, answer_list,  mode="train")
        global_step, tr_loss = train(args, train_dataset, model, nel_model, answer_list, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        # Use dir_eval as the model path for do_eval verification
        if args.dir_eval is None:
            args.dir_eval = args.dir_output

        eval_dirs_raw = []
        if "checkpoint" in args.dir_eval:
            eval_dirs_raw.append(args.dir_eval)
        else:
            eval_dirs_raw = glob(join(args.dir_eval, "checkpoint*"))

        eval_dirs = []
        for d in eval_dirs_raw:
            if exists(join(d, "test_results.json")):
                continue
            eval_dirs.append(d)

        for dir_ckpt in eval_dirs:
            config = AutoConfig.from_pretrained(
                dir_ckpt,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                dir_ckpt,
                cache_dir=args.cache_dir if args.cache_dir else None,
                **tokenizer_args,
            )
            model = AutoModel.from_pretrained(
                dir_ckpt,
                from_tf=bool(".ckpt" in dir_ckpt),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )


            path_args_history = join(dir_ckpt, "training_args.bin")
            args_his = torch.load(path_args_history)
            recover_nel_args(args, args_his)

            nel_model = NELModel(args)
            path_nel_state = join(dir_ckpt, "nel_model.pkl")
            logger.info(f"Load nel model state dict from {path_nel_state}")
            nel_model.load_state_dict(torch.load(path_nel_state, map_location=lambda storage, loc: storage))
            model.to(args.device)
            nel_model.to(args.device)
            results, _ = evaluate(args, model, nel_model, answer_list, tokenizer, mode="dev")[:2]
            json.dump(results, open(os.path.join(dir_ckpt, "dev_results.json"), 'w'), indent=2)

    if args.do_cross:
        logger.info(f"  ****************Cross Validation, Folds: {args.folds}****************")
        cross_arg = args.cross_arg
        args.dir_prepro = join(args.dir_prepro, "cross")
        dir_output_raw = args.dir_output
        value_cross_arg = getattr(args, cross_arg)
        while value_cross_arg <= args.ub_arg:
            logger.info(f"  Arg: {args.cross_arg}, "
                        f"now: {value_cross_arg}/{args.ub_arg} ({args.pace} {args.mode_pace})")
            args.dir_output = join(dir_output_raw, cross_arg, f"{value_cross_arg}")


            for f in range(args.folds):
                logger.info(
                    f"  Arg: {args.cross_arg}, now: {value_cross_arg}/{args.ub_arg} "
                    f"({args.pace} {args.mode_pace}), fold: {f+1}/{args.folds}")

                config = AutoConfig.from_pretrained(
                    args.config_name if args.config_name else args.model_name_or_path,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                    **tokenizer_args,
                )
                model = AutoModel.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                )
                print(f"dropout: {args.dropout}, weight_decay: {args.weight_decay}")

                nel_model = NELModel(args)
                path_nel_state = join(args.model_name_or_path, "nel_model.pkl")
                if exists(path_nel_state):
                    logger.info(f"Load nel model state dict from {path_nel_state}")
                    nel_model.load_state_dict(torch.load(path_nel_state))
                else:
                    for p in nel_model.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)

                model.to(args.device)
                nel_model.to(args.device)

                train_dataset, _ = load_and_cache_examples(args, tokenizer, answer_list, mode=f"train{f}")
                global_step, tr_loss = train(args, train_dataset, model, nel_model, answer_list, tokenizer, fold=str(f))
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            # update
            if args.mode_pace == 'add':
                value_cross_arg += args.pace
            elif args.mode_pace == 'multiple':
                value_cross_arg *= args.pace

            setattr(args, cross_arg, value_cross_arg)

    if args.do_cross_eval:
        cross_arg = args.cross_arg
        args.dir_prepro = join(args.dir_prepro, "cross")
        dir_output_raw = args.dir_output

        dir_ckpt_list = glob(join(dir_output_raw, cross_arg, "*", "checkpoint[0-4]*"))

        for dir_ckpt in dir_ckpt_list:
            if not exists(join(dir_ckpt, "dev_results.json")):
                fold = dir_ckpt.split("-")[0][-1]
                logger.info(f"  ******** {dir_ckpt} dev eval beginning. ********")
                model_path = dir_ckpt
                config = AutoConfig.from_pretrained(
                    model_path,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                    **tokenizer_args,
                )
                model = AutoModel.from_pretrained(
                    model_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                )

                path_args_history = join(model_path, "training_args.bin")
                args_his = torch.load(path_args_history)
                recover_nel_args(args, args_his)

                nel_model = NELModel(args)
                path_nel_state = join(model_path, "nel_model.pkl")
                logger.info(f"Load nel model state dict from {path_nel_state}")
                nel_model.load_state_dict(torch.load(path_nel_state))
                model.to(args.device)
                nel_model.to(args.device)

                results, _ = evaluate(args, model, nel_model, answer_list, tokenizer, mode=f"dev{fold}")[:2]
                json.dump(results, open(os.path.join(dir_ckpt, "dev_results.json"), 'w'), indent=2)

    if args.do_predict:

        if args.dir_eval is None:
            args.dir_eval = args.dir_output

        eval_dirs_raw = []
        if "checkpoint" in args.dir_eval:
            eval_dirs_raw.append(args.dir_eval)
        else:
            eval_dirs_raw = glob(join(args.dir_eval, "checkpoint*"))

        eval_dirs = []
        for d in eval_dirs_raw:
            if exists(join(d, "test_predictions.pkl")):
                continue
            eval_dirs.append(d)

        for dir_ckpt in eval_dirs:
            config = AutoConfig.from_pretrained(
                dir_ckpt,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                dir_ckpt,
                cache_dir=args.cache_dir if args.cache_dir else None,
                **tokenizer_args,
            )
            model = AutoModel.from_pretrained(
                dir_ckpt,
                from_tf=bool(".ckpt" in dir_ckpt),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )


            path_args_history = join(dir_ckpt, "training_args.bin")
            args_his = torch.load(path_args_history)
            recover_nel_args(args, args_his)

            nel_model = NELModel(args)
            path_nel_state = join(dir_ckpt, "nel_model.pkl")
            logger.info(f"Load nel model state dict from {path_nel_state}")
            nel_model.load_state_dict(torch.load(path_nel_state, map_location=lambda storage, loc: storage))

            model.to(args.device)
            nel_model.to(args.device)
            results, _, rank_list, all_sim_p, all_sim_n, guks = evaluate(args, model, nel_model, answer_list, tokenizer, mode="test")

            pickle.dump({
                'ranks': rank_list,
                'sim_p': all_sim_p,
                'sim_n': all_sim_n,
                'guks': guks
            }, open(os.path.join(dir_ckpt, "test_predictions.pkl"), 'wb'))
            json.dump(results, open(os.path.join(dir_ckpt, "test_results.json"), 'w'), indent=2)


if __name__ == "__main__":
    main()
