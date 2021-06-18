#!/bin/bash
python test.py --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model atlasnet \
       --weights ${path to checkpoint} \
       --test_mode ${mode}