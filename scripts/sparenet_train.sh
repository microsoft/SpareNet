#!/bin/bash
python train.py --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model sparenet \
       --weights ${path to checkpoint}