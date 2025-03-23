#!/bin/bash

# 设置基本参数
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/precomputed_features"
DATA_LOCATION="/home/haichao/zby/MetaNet-Bayes/data"
MODEL="ViT-B-32"
BATCH_SIZE=128

# 确保保存目录存在
mkdir -p $SAVE_DIR

# 定义数据集列表
DATASETS="Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN"

# 使用GNU Parallel在8个GPU上并行处理
parallel --jobs 8 --link \
  CUDA_VISIBLE_DEVICES={1} python src/precompute_features_subset.py \
  --model $MODEL \
  --save-dir $SAVE_DIR \
  --data-location $DATA_LOCATION \
  --batch-size $BATCH_SIZE \
  --datasets {2} ::: 0 1 2 3 4 5 6 7 ::: $DATASETS

# 确保清理任何可能残留的进程
echo "所有特征计算完成！"
echo "清理剩余资源..."
# 寻找并杀死任何可能卡住的相关Python进程
pkill -f "python src/precompute_features_subset.py"

echo "清理完成！"