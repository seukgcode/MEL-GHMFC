#!bin/bash

export DIR_CODE="nel_model"

export PATH_DATASET="data/prepro_data/dm/nel_121.json"
export PATH_ANS="data/prepro_data/dm/qids_ordered.json"
export PATH_NEG_CONFIG="data/prepro_data/dm/neg_cache/neg.json"

export DIR_PREPRO="data/prepro_data/nel"
export DIR_OUTPUT="data/output_data/nel"
export DIR_SEARCH="data/prepro_data/nel/search_top100.json"

export DIR_EVAL="data/output_data/nel/new515/checkpoint-54000"

export BATCH_SIZE=1


python $DIR_CODE/nel_train.py --dir_prepro $DIR_PREPRO \
--path_ans_list $PATH_ANS \
--path_neg_config $PATH_NEG_CONFIG \
--path_candidates $DIR_SEARCH \
--dir_img_feat $DIR_PREPRO \
--dir_neg_feat $DIR_PREPRO \
--dir_output $DIR_OUTPUT \
--per_gpu_eval_batch_size $BATCH_SIZE \
--model_type bert \
--strip_accents \
--do_predict \
--do_eval \
--dir_eval $DIR_EVAL \
--gpu_id 1