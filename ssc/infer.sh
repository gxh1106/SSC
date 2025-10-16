#!/bin/bash

# 如果任何命令执行失败，脚本将立即退出
set -e

export CUDA_VISIBLE_DEVICES="6"

CONFIG_FILE="experiments/train_SSC_96C_64E_CR_3_8/train_SSC_96C_64E_CR_3_8_from_pretrain.yml"
MODEL_PATH="experiments/train_SSC_96C_64E_CR_3_8/models/net_g_latest.pth"
KODAK24_PATH="datasets/Kodak24"
OUTPUT_DIR="output/test_SwinSSC_EEP"


# 构造基础命令
CMD="python ssc/inference.py \
    --config \"$CONFIG_FILE\" \
    --model_path \"$MODEL_PATH\" \
    --input \"$KODAK24_PATH\" \
    --output \"$OUTPUT_DIR\""

# 执行最终构建好的命令
# 使用 eval 来正确处理带引号的参数
eval $CMD
