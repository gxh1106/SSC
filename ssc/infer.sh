#!/bin/bash

# 如果任何命令执行失败，脚本将立即退出
set -e

# export CUDA_VISIBLE_DEVICES="7"
# export CUDA_VISIBLE_DEVICES="6"
# export CUDA_VISIBLE_DEVICES="5"
# export CUDA_VISIBLE_DEVICES="4"
# export CUDA_VISIBLE_DEVICES="3"
export CUDA_VISIBLE_DEVICES="2"
# export CUDA_VISIBLE_DEVICES="1"
# export CUDA_VISIBLE_DEVICES="0"

CSI_ERROR_TX=0.0  # 发送端CSI误差水平
CSI_ERROR_RX=0.0  # 接收端CSI误差水平


CONFIG_FILE="experiments/train_MVQ_bbp2_32C_4V_65536E/train_MVQ_bpp2_32C_4V_65536E_from_pretrain.yml"
MODEL_PATH="experiments/train_MVQ_bbp2_32C_4V_65536E/models/net_g_55000.pth"
OUTPUT_DIR="output/baseline/bpp2_32C_16E_4D/MVQ_4V_65536E"
# CONFIG_FILE="experiments/train_MVQ_bbp2_32C_4V_16384E/train_MVQ_bpp2_32C_4V_16384E_from_pretrain.yml"
# MODEL_PATH="experiments/train_MVQ_bbp2_32C_4V_16384E/models/net_g_25000.pth"
# OUTPUT_DIR="output/baseline/bpp2_32C_16E_4D/MVQ_4V_16384E"
# CONFIG_FILE="experiments/16xD/train_SSC_bpp2_32C_16E_4D/train_SSC_bpp2_32C_16E_4D_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_SSC_bpp2_32C_16E_4D/models/net_g_latest.pth"
# OUTPUT_DIR="output/CSI/bpp2_32C_16E_4D/CSIerr_TX${CSI_ERROR_TX}_RX${CSI_ERROR_RX}"
# CONFIG_FILE="experiments/16xD/train_SSC_bpp4_64C_16E_4D/train_SSC_bpp4_64C_16E_4D_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_SSC_bpp4_64C_16E_4D/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp4_64C_16E_4D/SSC_woIM"
# CONFIG_FILE="experiments/train_VQ_bpp2_64C_65536E/train_VQ_bpp2_64C_65536E_from_pretrain.yml"
# MODEL_PATH="experiments/train_VQ_bpp2_64C_65536E/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp4_64C_16E_4D/sDAC_FA-IM"
# CONFIG_FILE="experiments/diff_train/train_SSC_bbp2_32C_16E_4D_oldTrain/train_SSC_bpp2_32C_16E_4D_from_pretrain_oldTrain.yml"
# MODEL_PATH="experiments/diff_train/train_SSC_bbp2_32C_16E_4D_oldTrain/models/net_g_latest.pth"
# OUTPUT_DIR="output/diff_train/32C_4D_oldTrain"

# CONFIG_FILE="experiments/16xD/train_VQ_bpp6_96C_65536E/train_VQ_bpp2_96C_65536E_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_VQ_bpp6_96C_65536E/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp6_96C_16E_4D/sDAC_FA-IM"
# CONFIG_FILE="experiments/16xD/train_VQ_bpp4d5_96C_4096E/train_VQ_bpp1d5_96C_4096E_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_VQ_bpp4d5_96C_4096E/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp4d5_96C_16E_3D/sDAC_FA-IM"
# CONFIG_FILE="experiments/16xD/train_VQ_bpp3_96C_256E/train_VQ_bpp1_96C_256E_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_VQ_bpp3_96C_256E/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp3_96C_16E_2D/sDAC_FA-IM"
# CONFIG_FILE="experiments/16xD/train_VQ_bpp1d5_32C_4096E/train_VQ_bpp1d5_32C_4096E_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_VQ_bpp1d5_32C_4096E/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp1d5_32C_16E_3D/sDAC_FA-IM"
# CONFIG_FILE="experiments/16xD/train_VQ_bpp2_32C_65536E/train_VQ_bpp2_32C_65536E_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_VQ_bpp2_32C_65536E/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp2_32C_16E_4D/sDAC_FA-IM"

# CONFIG_FILE="experiments/16xD/train_SSC_bpp4d5_96C_16E_3D/train_SSC_bpp1d5_96C_16E_3D_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_SSC_bpp4d5_96C_16E_3D/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp4d5_96C_16E_3D/SSC_new"
# CONFIG_FILE="experiments/16xD/train_SSC_bpp3_96C_16E_2D/train_SSC_bpp1_96C_16E_2D_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_SSC_bpp3_96C_16E_2D/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp3_96C_16E_2D/SSC_woIM"
# CONFIG_FILE="experiments/16xD/train_SSC_bpp1d5_32C_16E_3D/train_SSC_bpp1d5_32C_16E_3D_from_pretrain.yml"
# MODEL_PATH="experiments/16xD/train_SSC_bpp1d5_32C_16E_3D/models/net_g_latest.pth"
# OUTPUT_DIR="output/16xD/bpp1d5_32C_16E_3D/SSC_"

KODAK24_PATH="datasets/Kodak24"

# 构造基础命令
CMD="python ssc/inference.py \
    --config \"$CONFIG_FILE\" \
    --model_path \"$MODEL_PATH\" \
    --input \"$KODAK24_PATH\" \
    --output \"$OUTPUT_DIR\" \
    --csi_error_tx $CSI_ERROR_TX \
    --csi_error_rx $CSI_ERROR_RX"

# 执行最终构建好的命令
# 使用 eval 来正确处理带引号的参数
eval $CMD
