#!/bin/bash

# 更健壮的批处理特征提取脚本
# 优化资源使用和错误处理
# 统一处理所有数据集

# 设置基本参数
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/precomputed_features"
DATA_LOCATION="/home/haichao/zby/MetaNet-Bayes/data"
MODEL="ViT-L-14"
BATCH_SIZE=128  # 通用批次大小
TIME_LIMITS=36000
NUM_AUGMENTATIONS=10  # 每个数据集创建的增强版本数量

# 获取可用GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个可用GPU"

# 确保保存目录存在
mkdir -p $SAVE_DIR

# 定义数据集列表 - 所有数据集统一处理
# DATASETS="Cars DTD EuroSAT GTSRB MNIST RESISC45 SVHN"
DATASETS="SUN397"

# 创建日志目录
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p $LOG_DIR

# 为每个数据集设置超时（秒）
declare -A TIMEOUTS
TIMEOUTS=(
    ["Cars"]=$TIME_LIMITS
    ["DTD"]=$TIME_LIMITS
    ["EuroSAT"]=$TIME_LIMITS
    ["GTSRB"]=$TIME_LIMITS
    ["MNIST"]=$TIME_LIMITS
    ["RESISC45"]=$TIME_LIMITS
    ["SUN397"]=$TIME_LIMITS
    ["SVHN"]=$TIME_LIMITS
)

# 为每个数据集设置批次超时（秒）
declare -A BATCH_TIMEOUTS
BATCH_TIMEOUTS=(
    ["Cars"]=$TIME_LIMITS
    ["DTD"]=$TIME_LIMITS
    ["EuroSAT"]=$TIME_LIMITS
    ["GTSRB"]=$TIME_LIMITS
    ["MNIST"]=$TIME_LIMITS
    ["RESISC45"]=$TIME_LIMITS
    ["SUN397"]=$TIME_LIMITS
    ["SVHN"]=$TIME_LIMITS
)

# 为特定数据集设置批次大小
declare -A BATCH_SIZES
BATCH_SIZES=(
    ["Cars"]=$BATCH_SIZE
    ["DTD"]=$BATCH_SIZE
    ["EuroSAT"]=$BATCH_SIZE
    ["GTSRB"]=$BATCH_SIZE
    ["MNIST"]=$BATCH_SIZE
    ["RESISC45"]=$BATCH_SIZE
    ["SUN397"]=$((BATCH_SIZE/4))  # SUN397使用较小的批次以防OOM
    ["SVHN"]=$BATCH_SIZE
)

# 创建独立子进程处理每个数据集
for i in {0..7}; do
    if [ $i -ge $(echo $DATASETS | wc -w) ]; then
        break  # 如果没有足够的数据集则退出
    fi

    # 获取当前数据集名称
    DATASET=$(echo $DATASETS | cut -d' ' -f$((i+1)))

    # 设置日志文件
    LOG_FILE="${LOG_DIR}/${DATASET}_$(date +%Y%m%d_%H%M%S).log"

    # 设置超时和批量大小
    TIMEOUT=${TIMEOUTS[$DATASET]}
    BATCH_TIMEOUT=${BATCH_TIMEOUTS[$DATASET]}
    CURR_BATCH_SIZE=${BATCH_SIZES[$DATASET]}

    # 计算GPU ID - 如果GPU不足则循环使用
    if [ $GPU_COUNT -gt 0 ]; then
        GPU_ID=$((i % GPU_COUNT))
    else
        GPU_ID=0
    fi

    echo "处理数据集 $DATASET 在 GPU $GPU_ID，创建 $NUM_AUGMENTATIONS 个增强版本，日志: $LOG_FILE"

    # 标准处理 - 所有数据集使用统一方法
    # --gpu-id $GPU_ID
    (
        timeout $TIMEOUT python src/precompute_features_batch.py \
            --model $MODEL \
            --save-dir $SAVE_DIR \
            --data-location $DATA_LOCATION \
            --batch-size $CURR_BATCH_SIZE \
            --dataset $DATASET \
            --gpu-id 7 \
            --batch-timeout $BATCH_TIMEOUT \
            --num-augmentations $NUM_AUGMENTATIONS \
            --verbose > $LOG_FILE 2>&1

        # 检查退出状态
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "警告: 处理 $DATASET 超时" >> $LOG_FILE
        elif [ $EXIT_CODE -ne 0 ]; then
            echo "错误: 处理 $DATASET 失败，代码 $EXIT_CODE" >> $LOG_FILE
        fi
    ) &

    # 稍微延迟启动下一个任务，避免资源冲突
    sleep 5
done

# 等待所有后台任务完成
wait

echo "所有特征计算完成！"
echo "清理剩余资源..."

# 清理 GPU 资源 (如果有可用的GPU)
if [ $GPU_COUNT -gt 0 ]; then
    for i in $(seq 0 $((GPU_COUNT-1))); do
        CUDA_VISIBLE_DEVICES=$i nvidia-smi --gpu-reset 2>/dev/null || true
    done
fi

echo "清理完成！"