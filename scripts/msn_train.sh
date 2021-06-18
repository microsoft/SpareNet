#!/bin/bash
python train.py --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model msn \
       --weights ${path to checkpoint}