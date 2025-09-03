#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

CONFIG_FILE="experiments/train_SSC_from_scratch/train_SSC_from_scratch.yml"
MODEL_PATH="experiments/train_SSC_from_scratch/models/net_g_80000.pth"
# Kodak24 数据集所在的文件夹路径
KODAK24_PATH="datasets/Kodak24"
# 结果输出文件夹
OUTPUT_DIR="output/SwinSSC_PSNR"

# Tiling 参数 (如果您的GPU显存有限，可以设置一个较小的值，如 256)
# 如果显存充足，可以设置为 `None` 来测试整张图
TILE_SIZE=None
TILE_OVERLAP=32

if [ "$TILE_SIZE" == "None" ]; then
    python ssc/inference.py \
        --config "$CONFIG_FILE" \
        --model_path "$MODEL_PATH" \
        --input "$KODAK24_PATH" \
        --output "$OUTPUT_DIR"
else
    python ssc/inference.py \
        --config "$CONFIG_FILE" \
        --model_path "$MODEL_PATH" \
        --input "$KODAK24_PATH" \
        --output "$OUTPUT_DIR" \
        --tile "$TILE_SIZE" \
        --tile_overlap "$TILE_OVERLAP"
fi
