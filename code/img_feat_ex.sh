#!/bin/bash

export PATH_CODE='resnet_model'
export PATH_PREPRO='data/prepro_data/nel'
export PATH_IMG='data/images'
export PATH_RESNET='/data/mydir/Models/resnet-200.t7'
export BATCH_MODE=1
export BATCH_SIZE=4
export GPU_ID=0
export CUDA_VISIBLE_DEVICES=0

th $PATH_CODE/extract.lua \
--path_resnet $PATH_RESNET \
--path_img_file_or_dir $PATH_IMG \
--path_prepro $PATH_PREPRO \
--batch_mode $BATCH_MODE \
--batch_size $BATCH_SIZE \
--gpuid $GPU_ID
