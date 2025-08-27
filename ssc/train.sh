#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,4"

NPROC_PER_NODE=2
MASTER_PORT=29500
SCRIPT_PATH="ssc/train.py"
OPTIONS_PATH="options/train_SSC_from_scratch.yml"

python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    -opt $OPTIONS_PATH \
    --launcher pytorch 
