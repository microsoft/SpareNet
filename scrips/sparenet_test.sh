#!/bin/bash
python --gpu ${GPUS}\
       --work_dir ${WORK_DIR} \
       --model ${network} \
       --weights ${path to checkpoint} \
       --test_mode ${mode}