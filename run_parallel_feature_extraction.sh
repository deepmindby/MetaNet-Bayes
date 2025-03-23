#!/bin/bash

# 更健壮的批处理特征提取脚本
# 优化资源使用和错误处理

# 设置基本参数
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/precomputed_features"
DATA_LOCATION="/home/haichao/zby/MetaNet-Bayes/data"
MODEL="ViT-B-32"
BATCH_SIZE=64  # 降低批次大小提高稳定性

# 确保保存目录存在
mkdir -p $SAVE_DIR

# 定义数据集列表
DATASETS="Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN"

# 创建日志目录
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p $LOG_DIR

# 为每个数据集设置超时（秒）
declare -A TIMEOUTS
TIMEOUTS=(
    ["Cars"]=3600
    ["DTD"]=3600
    ["EuroSAT"]=3600
    ["GTSRB"]=3600
    ["MNIST"]=3600
    ["RESISC45"]=3600
    ["SUN397"]=3600
    ["SVHN"]=3600
)

# 为每个数据集设置批次超时（秒）
declare -A BATCH_TIMEOUTS
BATCH_TIMEOUTS=(
    ["Cars"]=60
    ["DTD"]=60
    ["EuroSAT"]=60
    ["GTSRB"]=60
    ["MNIST"]=60
    ["RESISC45"]=60
    ["SUN397"]=60
    ["SVHN"]=60
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

    # 设置超时
    TIMEOUT=${TIMEOUTS[$DATASET]}
    BATCH_TIMEOUT=${BATCH_TIMEOUTS[$DATASET]}

    echo "处理数据集 $DATASET 在 GPU $i，日志: $LOG_FILE"

    # 使用timeout命令设置整体超时，并在后台运行
    (
        timeout $TIMEOUT python precompute_features_batch.py \
            --model $MODEL \
            --save-dir $SAVE_DIR \
            --data-location $DATA_LOCATION \
            --batch-size $BATCH_SIZE \
            --dataset $DATASET \
            --gpu-id $i \
            --batch-timeout $BATCH_TIMEOUT \
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

# 确保释放所有GPU资源
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i nvidia-smi --gpu-reset 2>/dev/null || true
done

echo "清理完成！"