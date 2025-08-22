#!/usr/bin/env bash
set -e

# 逗号分隔 GPU 列表，例如：0 或 0,1 或 2,3,4
GPU_ID="${GPU_ID:-0,2,3}"

# 数据集选项：div2k | cifar10
DATASET="${DATASET:-div2k}"
# 评测集（仅 DIV2K 有意义）：kodak | CLIC21 | ffhq
TESTSET="${TESTSET:-kodak}"

# 训练超参与模型配置
MODEL="${MODEL:-SwinJSCC_w/_SAandRA}"   # 其它：SwinJSCC_w/o_SAandRA | SwinJSCC_w/_SA | SwinJSCC_w/_RA
MODEL_SIZE="${MODEL_SIZE:-base}"        # small | base | large
CHANNEL="${CHANNEL:-awgn}"               # awgn | rayleigh
C_LIST="${C_LIST:-96}"                   # 例如 "32,64,96"
SNR_LIST="${SNR_LIST:-10}"               # 例如 "5,10,15"
LOSS="${LOSS:-MSE}"                      # MSE | MS-SSIM
EPOCHS="${EPOCHS:-100}"
WORKERS="${WORKERS:-8}"
SAVE_DIR="${SAVE_DIR:-./outputs}"

# 将 DATASET 映射到 main.py 的 --trainset 取值
DS_LC="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
case "$DS_LC" in
  cifar10) TRAINSET="CIFAR10" ;;
  div2k)   TRAINSET="DIV2K" ;;
  *) echo "[ERR] Unsupported DATASET: $DATASET"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "[INFO] Using GPUs: $CUDA_VISIBLE_DEVICES (count=$NPROC)"
echo "[INFO] Trainset: $TRAINSET, Testset: $TESTSET"
echo "[INFO] Save dir: $SAVE_DIR"

mkdir -p "$SAVE_DIR"

# 单进程 + DataParallel：可见多卡则自动用多卡
python -u main.py \
  --training \
  --trainset "$TRAINSET" \
  --testset "$TESTSET" \
  --distortion-metric "$LOSS" \
  --model "$MODEL" \
  --channel-type "$CHANNEL" \
  --C "$C_LIST" \
  --multiple-snr "$SNR_LIST" \
  --model_size "$MODEL_SIZE" \
  --epochs "$EPOCHS" \
  --save-dir "$SAVE_DIR" \
  --num-workers "$WORKERS" \
  "$@"

# 用法：
# bash train.sh                                 # 使用默认配置
# GPU_ID=0 bash train.sh                        # 单卡
# GPU_ID=0,1 LOSS=MS-SSIM EPOCHS=200 bash train.sh
# MODEL_SIZE=small C_LIST=64 SNR_LIST=5,10 bash train.sh