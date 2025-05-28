#!/bin/bash
# filepath: download_qwen32b.sh

# Enable faster downloads with hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create the target directory
model_name=$1
mkdir -p model_store/

# Download the model to the specific directory
huggingface-cli download $model_name --local-dir model_store/$model_name

echo "Model downloaded to model_store/$model_name"