#!/bin/bash

# ==============================================================================
# Shell script to run the SwinJSCC training
#
# Instructions:
# 1. Modify the configuration variables in the section below to match your
#    desired training setup.
# 2. Make the script executable by running:
#    chmod +x run_training.sh
# 3. Run the script from your terminal:
#    ./run_training.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Set the dataset for training. Options: 'DIV2K', 'CIFAR10'
TRAINSET="DIV2K"

# Set the model architecture.
# Options: 'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'
MODEL="SwinJSCC_w/_SA"

# Set the model size. Options: 'small', 'base', 'large'
MODEL_SIZE="base"

# Set the channel type. Options: 'awgn', 'rayleigh'
CHANNEL_TYPE="awgn"

# Set the bottleneck dimension (number of channels).
# This is mainly used when the rate is not adapted by the model itself
# (e.g., for 'SwinJSCC_w/o_SAandRA'). We set a default value here.
C_VALUE="96"

# Set the distortion metric for evaluation. Options: 'MSE', 'MS-SSIM'
DISTORTION_METRIC="MSE"

# --- Execution ---
echo "================================================="
echo "Starting SwinJSCC Training"
echo "================================================="
echo "Dataset:           ${TRAINSET}"
echo "Model:             ${MODEL}"
echo "Model Size:        ${MODEL_SIZE}"
echo "Channel Type:      ${CHANNEL_TYPE}"
echo "Distortion Metric: ${DISTORTION_METRIC}"
echo "================================================="

# Construct the command
# The backslashes "\" at the end of each line allow us to write a single
# command across multiple lines for better readability.
COMMAND="python main.py \
    --training \
    --trainset ${TRAINSET} \
    --model ${MODEL} \
    --model_size ${MODEL_SIZE} \
    --channel-type ${CHANNEL_TYPE} \
    --distortion-metric ${DISTORTION_METRIC} \
    --C ${C_VALUE}"

# Print the command to the console before running
echo "Executing the following command:"
echo "${COMMAND}"
echo "================================================="

# Execute the command
${COMMAND}

echo "================================================="
echo "Training script finished."
echo "================================================="