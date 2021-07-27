#!bin/bash

export PATH_CODE=data_model
export MAX_SAMPLE_NUM=3
export NUM_CANDIDATES=100
export SEED=123

### construct NEL dataset
python $PATH_CODE/nel_ds_construct.py
python $PATH_CODE/dict_build.py  # construct data mapping

## Build a list of candidates
python $PATH_CODE/search_kg_fuzz.py --num_search $NUM_CANDIDATES

## Negative sampling preprocessing
python $PATH_CODE/kg_sample_process.py --max_sample_num $MAX_SAMPLE_NUM \
--overwrite_cache \
--seed $SEED

