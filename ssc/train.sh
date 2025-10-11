#!/bin/bash

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# NPROC_PER_NODE=4

export CUDA_VISIBLE_DEVICES="2,4"
# export CUDA_VISIBLE_DEVICES="5,6"
# export CUDA_VISIBLE_DEVICES="0,5"
NPROC_PER_NODE=2

MASTER_PORT=29500
# MASTER_PORT=29501
# MASTER_PORT=29502

SCRIPT_PATH="ssc/train.py"
# OPTIONS_PATH="options/train_VQ_96C_256E_CR_3_8_from_pretrain.yml"
OPTIONS_PATH="options/train_SSC_64C_64E_CR_2_8_from_pretrain.yml"


python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    -opt $OPTIONS_PATH \
    --launcher pytorch \
    --auto_resume
    # --debug

# tensorboard --logdir tb_logger