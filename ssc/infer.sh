#!/bin/bash

# 如果任何命令执行失败，脚本将立即退出
set -e

export CUDA_VISIBLE_DEVICES="1"

CONFIG_FILE="experiments/train_SSC_CR_1_8_multiSNR/train_SSC_from_pretrain.yml"
MODEL_PATH="experiments/train_SSC_CR_1_8_multiSNR/models/net_g_latest.pth"
KODAK24_PATH="datasets/Kodak24"
OUTPUT_DIR="output/test_SwinSSC"

# Tiling 参数:
# - 如果显存充足, 设为 "None" (字符串) 来测试整张图。
# - 如果显存有限, 设为一个能被窗口大小整除的数字 (例如 256)。
TILE_SIZE="None"
TILE_OVERLAP=32

# 构造基础命令
CMD="python ssc/inference.py \
    --config \"$CONFIG_FILE\" \
    --model_path \"$MODEL_PATH\" \
    --input \"$KODAK24_PATH\" \
    --output \"$OUTPUT_DIR\""

# 只有当 TILE_SIZE 不是字符串 "None" 且不为空时，才添加 tiling 参数
if [ -n "$TILE_SIZE" ] && [ "$TILE_SIZE" != "None" ]; then
    CMD="$CMD --tile \"$TILE_SIZE\" --tile_overlap \"$TILE_OVERLAP\""
fi

# 执行最终构建好的命令
# 使用 eval 来正确处理带引号的参数
eval $CMD
