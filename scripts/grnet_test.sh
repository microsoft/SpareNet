#!/bin/bash
python test.py --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model grnet \
       --weights ${path to checkpoint} \
       --test_mode ${mode}