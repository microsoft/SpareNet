#!/bin/bash
python train.py --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model grnet \
       --weights ${path to checkpoint}