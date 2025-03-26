#!/bin/bash

# SUN397 数据集增强特征提取脚本
# 增加10个随机增强版本

# 设置基本参数
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/precomputed_features"
DATA_LOCATION="/home/haichao/zby/MetaNet-Bayes/data"
MODEL="ViT-B-32"
BATCH_SIZE=64  # SUN397需要更小的批次大小
NUM_AUGMENTATIONS=10  # 生成10个增强版本

# 创建日志目录
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p $LOG_DIR

# 设置日志文件
LOG_FILE="${LOG_DIR}/SUN397_augmentation_$(date +%Y%m%d_%H%M%S).log"

echo "开始处理SUN397数据集，生成 $NUM_AUGMENTATIONS 个增强版本，日志: $LOG_FILE"

# 设置GPU ID - 使用0号GPU
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 运行SUN397特征提取脚本
python src/precompute_features_sun397.py \
    --model $MODEL \
    --save-dir $SAVE_DIR \
    --data-location $DATA_LOCATION \
    --batch-size $BATCH_SIZE \
    --num-augmentations $NUM_AUGMENTATIONS > $LOG_FILE 2>&1

# 检查退出状态
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "错误: 处理 SUN397 失败，代码 $EXIT_CODE，详见日志: $LOG_FILE"
else
    echo "成功处理 SUN397 数据集，生成了 $NUM_AUGMENTATIONS 个增强版本"
fi

echo "清理 GPU 资源..."
nvidia-smi --gpu-reset 2>/dev/null || true
echo "清理完成！"