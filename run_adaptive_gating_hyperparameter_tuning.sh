#!/bin/bash

# 自适应门控超参数调优脚本 - 简化版
# 直接在8张GPU上平均分配任务，不进行复杂的GPU检查

# 基础设置
MODEL="ViT-B-16"  # 可选: ViT-B-32, RN50, RN101 等
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/checkpoints_hyperparameter_tuning"
DATA_LOCATION="~/zby/MetaNet-Bayes"
BATCH_SIZE=64
EPOCHS=10
NUM_WORKERS=4
NUM_TASK_VECTORS=8
SEED=42

# GPU设置 - 直接使用所有GPU
GPUS=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPUS[@]}

# 创建实验日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SAVE_DIR}/logs_${TIMESTAMP}"
RESULTS_FILE="${LOG_DIR}/tuning_results.csv"
mkdir -p "$LOG_DIR"

# 初始化结果CSV文件
echo "dataset,base_threshold,beta,uncertainty_reg,blockwise,no_gating,no_metanet,train_accuracy,model,gpu_id,time_minutes" > "$RESULTS_FILE"

# 要测试的数据集
DATASETS=("MNIST")
#DATASETS=("MNIST" "RESISC45" "SUN397" "SVHN")
#DATASETS=("Cars" "DTD" "EuroSAT" "GTSRB" "MNIST" "RESISC45" "SUN397" "SVHN")

# 超参数搜索空间
BASE_THRESHOLDS=(0.01 0.03 0.05 0.06 0.07)
BETAS=(0.8 1.0 1.2)
UNCERTAINTY_REGS=(0.008 0.01 0.015 0.02)
LR_MULTIPLIERS=(20.0 30.0 40.0 50.0 60.0 70.0 100.0)
WEIGHT_DECAYS=(0.0005)
REG_COEFFICIENTS=(0.0008 0.001 0.0012 0.002)
MARGIN_WEIGHTS=(0.00005 0.0001 0.0002 0.0005 0.0006)
#BASE_THRESHOLDS=(0.05)
#BETAS=(1.0)
#UNCERTAINTY_REGS=(0.01)
#LR_MULTIPLIERS=(50.0)
#WEIGHT_DECAYS=(0.0005)
#REG_COEFFICIENTS=(0.0008 0.001)
#MARGIN_WEIGHTS=(0.0001)


# 设置实验标题
echo "========================================"
echo "多GPU超参数调优实验 - $(date)"
echo "模型: $MODEL"
echo "保存目录: $SAVE_DIR"
echo "数据集: ${DATASETS[@]}"
echo "GPU编号: ${GPUS[@]}"
echo "========================================"

# 简单进度条函数
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percent=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))

    printf "\r进度: [%${completed}s%${remaining}s] %d/%d (%d%%)" \
           "$(printf '%0.s#' $(seq 1 $completed))" \
           "$(printf '%0.s-' $(seq 1 $remaining))" \
           "$current" "$total" "$percent"
}

# 运行实验函数 - 直接指定GPU
run_experiment() {
    local dataset=$1
    local base_threshold=$2
    local beta=$3
    local uncertainty_reg=$4
    local blockwise=$5
    local no_gating=$6
    local no_metanet=$7
    local gpu_id=$8
    local task_id=$9

    # 为每个实验设置唯一端口，避免冲突
    local port=$((12000 + task_id))

    # 设置特定于此实验的保存目录
    local exp_name="exp_${dataset}_bt${base_threshold}_beta${beta}_ur${uncertainty_reg}"
    if $blockwise; then exp_name="${exp_name}_blockwise"; fi
    if $no_gating; then exp_name="${exp_name}_nogating"; fi
    if $no_metanet; then exp_name="${exp_name}_nometanet"; fi

    local exp_save_dir="${SAVE_DIR}/${exp_name}"
    local log_file="${LOG_DIR}/${exp_name}_gpu${gpu_id}.log"

    echo -e "\n启动实验: $exp_name (GPU $gpu_id)"

    # 记录开始时间
    start_time=$(date +%s)

    # 构建命令行参数
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -m src.train_with_adaptive_gating"
    cmd="$cmd --model $MODEL"
    cmd="$cmd --world-size 1"  # 单GPU训练
    cmd="$cmd --save-dir $exp_save_dir"
    cmd="$cmd --data-location $DATA_LOCATION"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --num-workers $NUM_WORKERS"
    cmd="$cmd --num-task-vectors $NUM_TASK_VECTORS"
    cmd="$cmd --seed $((SEED + gpu_id))"  # 不同GPU用不同种子
    cmd="$cmd --datasets $dataset"
    cmd="$cmd --base-threshold $base_threshold"
    cmd="$cmd --beta $beta"
    cmd="$cmd --uncertainty-reg $uncertainty_reg"
    cmd="$cmd --port $port"  # 添加端口号避免冲突

    # 添加布尔参数
    if $blockwise; then cmd="$cmd --blockwise-coef"; fi
    if $no_gating; then cmd="$cmd --no-gating"; fi
    if $no_metanet; then cmd="$cmd --no-metanet"; fi

    # 运行命令并记录输出
    echo "$cmd" > "$log_file"

    # 运行进程
    (
        # 确保环境变量正确设置
        export CUDA_VISIBLE_DEVICES=$gpu_id

        # 运行进程
        eval "$cmd" >> "$log_file" 2>&1
        exit_code=$?

        # 计算运行时间(分钟)
        end_time=$(date +%s)
        duration_minutes=$(( (end_time - start_time) / 60 ))

        # 提取训练精度结果 - 只关注训练精度，不添加评估部分
        local train_accuracy=$(grep -oP "Epoch \d+/\d+ - Training.*Loss: \K[0-9.]+" "$log_file" | tail -5 | awk '{ sum += $1 } END { if (NR > 0) print 100 - sum/NR; else print "N/A" }')
        if [ -z "$train_accuracy" ] || [ "$train_accuracy" == "N/A" ]; then
            # 尝试其他可能的格式 - 寻找最后的训练损失
            train_loss=$(grep -oP "Task Loss: \K[0-9.]+" "$log_file" | tail -1)
            if [ -n "$train_loss" ]; then
                # 简单地将损失转换为估计的精度 (仅用于记录)
                train_accuracy=$(echo "scale=2; 100 - $train_loss * 10" | bc)
            else
                train_accuracy="N/A"
            fi
        fi

        # 记录结果到CSV - 只记录训练指标
        echo "$dataset,$base_threshold,$beta,$uncertainty_reg,$blockwise,$no_gating,$no_metanet,$train_accuracy,$MODEL,$gpu_id,$duration_minutes" >> "$RESULTS_FILE"

        echo -e "\n完成实验: $exp_name, 训练指标: $train_accuracy (GPU $gpu_id), 耗时: ${duration_minutes}分钟"
    ) &

    # 返回进程PID以便可能的跟踪
    echo $!
}

# 清理旧的CUDA缓存
echo "清理CUDA缓存..."
for gpu_id in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_id python -c "import torch; torch.cuda.empty_cache()" &
done
wait

# 实验任务列表
declare -a EXPERIMENT_TASKS

# 添加实验任务的函数
add_experiment_task() {
    local dataset=$1
    local base_threshold=$2
    local beta=$3
    local uncertainty_reg=$4
    local blockwise=$5
    local no_gating=$6
    local no_metanet=$7

    EXPERIMENT_TASKS+=("$dataset $base_threshold $beta $uncertainty_reg $blockwise $no_gating $no_metanet")
}

# 创建实验任务列表
echo "准备实验任务..."

# 实验1: MetaNet带自适应门控
for dataset in "${DATASETS[@]}"; do
    # 标准MetaNet(带自适应门控)
    add_experiment_task "$dataset" 0.05 1.0 0.01 true false false
    # MetaNet无门控
    add_experiment_task "$dataset" 0.05 1.0 0.01 true true false
    # Atlas方法
    add_experiment_task "$dataset" 0.05 1.0 0.01 true false true
done

# 实验2: 比较不同的base_threshold
for dataset in "${DATASETS[@]}"; do
    for base_threshold in "${BASE_THRESHOLDS[@]}"; do
        if [ "$base_threshold" != "0.05" ]; then  # 避免重复
            add_experiment_task "$dataset" $base_threshold 1.0 0.01 true false false
        fi
    done
done

# 实验3: 比较不同的beta
for dataset in "${DATASETS[@]}"; do
    for beta in "${BETAS[@]}"; do
        if [ "$beta" != "1.0" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 $beta 0.01 true false false
        fi
    done
done

# 实验4: 比较不同的uncertainty_reg
for dataset in "${DATASETS[@]}"; do
    for uncertainty_reg in "${UNCERTAINTY_REGS[@]}"; do
        if [ "$uncertainty_reg" != "0.01" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 1.0 $uncertainty_reg true false false
        fi
    done
done

# 显示实验总数
TOTAL_EXPERIMENTS=${#EXPERIMENT_TASKS[@]}
echo "总共 $TOTAL_EXPERIMENTS 个实验任务将被执行"

# 随机排序实验任务以更均匀地分配负载
echo "随机排序实验任务..."
for i in $(seq $((TOTAL_EXPERIMENTS-1)) -1 0); do
    j=$((RANDOM % (i+1)))
    temp="${EXPERIMENT_TASKS[$i]}"
    EXPERIMENT_TASKS[$i]="${EXPERIMENT_TASKS[$j]}"
    EXPERIMENT_TASKS[$j]="$temp"
done

# 直接开始实验，平均分配到所有GPU
echo -e "\n开始执行实验..."
COMPLETED_TASKS=0
PIDS=()

for i in $(seq 0 $((TOTAL_EXPERIMENTS-1))); do
    # 计算GPU ID - 简单循环分配
    gpu_id=${GPUS[$((i % GPU_COUNT))]}

    # 解析任务参数
    task=${EXPERIMENT_TASKS[$i]}
    read -r dataset base_threshold beta uncertainty_reg blockwise no_gating no_metanet <<< "$task"

    # 启动实验
    run_experiment "$dataset" "$base_threshold" "$beta" "$uncertainty_reg" "$blockwise" "$no_gating" "$no_metanet" "$gpu_id" "$i"

    # 简单进度显示
    COMPLETED_TASKS=$((i+1))
    show_progress $COMPLETED_TASKS $TOTAL_EXPERIMENTS

    # 短暂暂停以避免同时启动过多进程
    sleep 0.5
done

echo -e "\n所有实验已启动! 等待完成..."

# 等待所有进程完成
wait

# 生成结果摘要
echo "生成结果摘要..."
python -c "
import pandas as pd
import os
import numpy as np
from datetime import datetime

# 读取结果
results = pd.read_csv('$RESULTS_FILE')
results['train_accuracy'] = pd.to_numeric(results['train_accuracy'], errors='coerce')

# 创建摘要文件
with open('${LOG_DIR}/training_summary.txt', 'w') as f:
    f.write(f'多GPU超参数调优训练摘要 - {datetime.now().strftime(\"%Y-%m-%d %H:%M\")}\\n')
    f.write('======================\\n\\n')

    # 实验概况
    f.write('训练概况:\\n')
    f.write(f'总实验数: {len(results)}\\n')
    f.write(f'已完成实验数: {len(results[results[\"train_accuracy\"] != \"N/A\"])}\\n')
    if results['train_accuracy'].notna().any():
        f.write(f'平均训练指标: {results[\"train_accuracy\"].mean():.2f}\\n')

    # 每个数据集的最佳参数
    f.write('\\n\\n每个数据集的最佳训练参数:\\n')
    for dataset in results['dataset'].unique():
        dataset_results = results[results['dataset'] == dataset]
        if dataset_results['train_accuracy'].notna().any():
            best_row = dataset_results.loc[dataset_results['train_accuracy'].idxmax()]
            f.write(f'\\n{dataset}:\\n')
            f.write(f'  最佳训练指标: {best_row[\"train_accuracy\"]}\\n')
            f.write(f'  参数: base_threshold={best_row[\"base_threshold\"]}, beta={best_row[\"beta\"]}, uncertainty_reg={best_row[\"uncertainty_reg\"]}\\n')
            f.write(f'  blockwise={best_row[\"blockwise\"]}, no_gating={best_row[\"no_gating\"]}, no_metanet={best_row[\"no_metanet\"]}\\n')

    # 各方法的比较
    f.write('\\n\\n不同方法的训练指标比较:\\n')
    methods = []
    if (results['no_metanet'] == False).any() and (results['no_gating'] == False).any():
        methods.append(('MetaNet with Gating', (results['no_metanet'] == False) & (results['no_gating'] == False)))
    if (results['no_metanet'] == False).any() and (results['no_gating'] == True).any():
        methods.append(('MetaNet without Gating', (results['no_metanet'] == False) & (results['no_gating'] == True)))
    if (results['no_metanet'] == True).any():
        methods.append(('Atlas (Direct Features)', results['no_metanet'] == True))

    for method_name, mask in methods:
        method_results = results[mask]
        if not method_results.empty and method_results['train_accuracy'].notna().any():
            avg_acc = method_results['train_accuracy'].mean()
            f.write(f'\\n{method_name}:\\n')
            f.write(f'  平均训练指标: {avg_acc:.2f}\\n')

            # 每个数据集的结果
            for dataset in method_results['dataset'].unique():
                dataset_method = method_results[method_results['dataset'] == dataset]
                if not dataset_method.empty and dataset_method['train_accuracy'].notna().any():
                    best_acc = dataset_method['train_accuracy'].max()
                    f.write(f'  {dataset}: {best_acc:.2f}\\n')

    f.write('\\n注意: 此摘要仅包含训练指标，不包含评估结果。请使用evaluate_model.sh脚本进行正式评估。\\n')
"

echo "训练实验完成！结果保存在: $RESULTS_FILE"
echo "训练摘要报告: ${LOG_DIR}/training_summary.txt"

# 提醒用户进行评估
echo "训练完成后，请使用evaluate_model.sh脚本对最佳模型进行评估。"