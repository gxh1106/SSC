#!/bin/bash

# 如果任何命令执行失败，脚本将立即退出
set -e

export CUDA_VISIBLE_DEVICES="1"

CONFIG_FILE="options/Ke/train_SSC_96C_16E_4D_Ke4_from_pretrain.yml"
MODEL_PATH="experiments/Ke/train_SSC_96C_16E_4D_Ke4_newTrain/models/net_g_latest.pth"
KODAK24_PATH="datasets/Kodak24/kodim24.png"
OUTPUT_DIR="output/test_error_step"
ERROR_STEP=3

# 构造基础命令
CMD="python ssc/inference_net.py \
    --config \"$CONFIG_FILE\" \
    --model_path \"$MODEL_PATH\" \
    --input \"$KODAK24_PATH\" \
    --output \"$OUTPUT_DIR\" \
    --error_step $ERROR_STEP"

# 执行最终构建好的命令
# 使用 eval 来正确处理带引号的参数
eval $CMD
