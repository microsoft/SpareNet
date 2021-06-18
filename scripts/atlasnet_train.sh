#!/bin/bash
python train.py --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model atlasnet \
       --weights ${path to checkpoint}