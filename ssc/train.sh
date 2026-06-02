#!/bin/bash

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# NPROC_PER_NODE=4

export CUDA_VISIBLE_DEVICES="6,7"
# export CUDA_VISIBLE_DEVICES="4,5"
# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="0,1"
NPROC_PER_NODE=2

# export CUDA_VISIBLE_DEVICES="5"
# NPROC_PER_NODE=1

# MASTER_PORT=29500
MASTER_PORT=29501
# MASTER_PORT=29502
# MASTER_PORT=29503

SCRIPT_PATH="ssc/train.py"
OPTIONS_PATH="options/baseline/train_MVQ_bpp2_32C_4V_16384E_from_pretrain.yml"
# OPTIONS_PATH="options/baseline/train_MVQ_bpp2_32C_4V_65536E_from_pretrain.yml"
# OPTIONS_PATH="options/16xD/train_SSC_bpp4_64C_16E_4D_from_pretrain.yml"
# OPTIONS_PATH="options/16xD/train_VQ_bpp2_64C_65536E_from_pretrain.yml"
# OPTIONS_PATH="options/train_SSC_from_scratch_64C.yml"



python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    -opt $OPTIONS_PATH \
    --launcher pytorch \
    --auto_resume \
    # --debug

# tensorboard --logdir tb_logger