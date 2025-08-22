#!/usr/bin/env bash
set -e

# 选择 GPU（逗号分隔），例：0 或 0,1 或 2,3,4
GPU_ID="5,6"

# 训练配置（可按需修改）
DATASET="div2k"         # 可选：div2k | cifar10 | vimeo
DATA_PATH="./datasets"  # 数据目录
SAVE_PATH="./outputs"   # 输出目录
TAG="$(date +%Y%m%d-%H%M%S)"

# 自动计算并行数
NPROC=$(echo "$GPU_ID" | awk -F',' '{print NF}')
# 随机端口，避免冲突
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}

# 选择分布式启动器（优先 torchrun）
if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER="torchrun"
else
  LAUNCHER="python -m torch.distributed.run"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[INFO] Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "[INFO] World size: $NPROC"
echo "[INFO] Dataset: $DATASET"
echo "[INFO] Data path: $DATA_PATH"
echo "[INFO] Save path: $SAVE_PATH"
echo "[INFO] Tag: $TAG"

if [ "$NPROC" -gt 1 ]; then
  # 多卡 DDP
  $LAUNCHER \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    train_adapter.py \
      --DDP \
      --dataset "$DATASET" \
      --data_path "$DATA_PATH" \
      --save_path "$SAVE_PATH" \
      --tag "$TAG" \
      "$@"
else
  # 单卡/CPU
  python train_adapter.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --tag "$TAG" \
    "$@"
fi

# 用法示例：
# bash train.sh
# bash train.sh --log_interval 50   # 额外传参直接拼在最后