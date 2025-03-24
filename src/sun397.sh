#!/bin/bash

# Simplified batch feature extraction script

# Basic parameters
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/precomputed_features"
DATA_LOCATION="/home/haichao/zby/MetaNet-Bayes/data"
MODEL="ViT-B-32"
BATCH_SIZE=64  # Lower batch size for SUN397

# Ensure save directory exists
mkdir -p $SAVE_DIR

# Process SUN397 directly with specialized script
echo "Processing SUN397 dataset..."
python src/precompute_features_sun397.py \
    --model $MODEL \
    --save-dir $SAVE_DIR \
    --data-location $DATA_LOCATION \
    --batch-size $BATCH_SIZE

echo "SUN397 feature extraction complete!"