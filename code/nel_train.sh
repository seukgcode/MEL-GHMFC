#!bin/bash

export DIR_CODE="nel_model"

export PATH_DATASET="data/prepro_data/dm/nel_121.json"
export PATH_ANS="data/prepro_data/dm/qids_ordered.json"
export PATH_NEG_CONFIG="data/prepro_data/dm/neg_cache/neg.json"

export DIR_PREPRO="data/prepro_data/nel"
export DIR_OUTPUT="data/output_data/nel/new515"

export DIR_SEARCH="data/prepro_data/nel/search_top100.json"

export MODE="train"

export EPOCHS=300

export SAVE_STEPS=1200
export BATCH_SIZE=32
export DROPOUT=0.3
export DECAY=0.05

export LR=5e-5

export NUM_ATTEN_LAYERS=2
export NEG_SAMPLE_NUM=1
export HIDDEN_SIZE=512
export NUM_HEADERS=8
export FF_SIZE=2048
export OUTPUT_SIZE=768

export FEAT_CATE="wp"
export LOSS_FUNCTION="triplet"
export LOSS_MARGIN=0.5
export SIMILARITY="cos"

export LOSS_SCALE=16

export GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU

export MODEL_NAME_OR_PATH='data/output_data/nel/new515/checkpoint-55200'

export SEED=123


#python $DIR_CODE/nel_process.py --dir_prepro $DIR_PREPRO \
#--path_dataset $PATH_DATASET \
#--shuffle --seed $SEED

python $DIR_CODE/nel_train.py --dir_prepro $DIR_PREPRO \
--path_ans_list $PATH_ANS \
--path_neg_config $PATH_NEG_CONFIG \
--dir_img_feat $DIR_PREPRO \
--dir_neg_feat $DIR_PREPRO \
--dir_output $DIR_OUTPUT \
--save_steps $SAVE_STEPS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--neg_sample_num $NEG_SAMPLE_NUM \
--model_type bert \
--model_name_or_path $MODEL_NAME_OR_PATH \
--num_train_epochs $EPOCHS \
--overwrite_output_dir \
--strip_accents \
--path_candidates $DIR_SEARCH \
--num_attn_layers $NUM_ATTEN_LAYERS \
--loss_scale $LOSS_SCALE \
--loss_margin $LOSS_MARGIN \
--loss_function $LOSS_FUNCTION \
--similarity $SIMILARITY \
--feat_cate $FEAT_CATE \
--learning_rate $LR \
--dropout $DROPOUT \
--weight_decay $DECAY \
--hidden_size $HIDDEN_SIZE \
--nheaders $NUM_HEADERS \
--ff_size $FF_SIZE \
--output_size $OUTPUT_SIZE \
--do_train \
--evaluate_during_training \
--gpu_id $GPU