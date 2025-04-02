#!/bin/bash

# 自适应门控超参数调优脚本 - 增强版
# 直接在8张GPU上平均分配任务，不进行复杂的GPU检查

# 基础设置
MODEL="ViT-B-32"  # 可选: ViT-B-32, RN50, RN101 等
SAVE_DIR="/home/haichao/zby/MetaNet-Bayes/checkpoints_hyperparameter_tuning"
DATA_LOCATION="/home/haichao/zby/MetaNet-Bayes"  # 使用绝对路径
BATCH_SIZE=128
EPOCHS=10  # 减少训练轮数加快实验
NUM_WORKERS=4
NUM_TASK_VECTORS=8
SEED=42
BASE_LR=0.0005  # 固定学习率为0.0005

# GPU设置 - 直接使用所有GPU
GPUS=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPUS[@]}

# 创建实验日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SAVE_DIR}/logs_${TIMESTAMP}"
RESULTS_FILE="${LOG_DIR}/tuning_results.csv"
mkdir -p "$LOG_DIR"

# 初始化结果CSV文件
echo "dataset,base_threshold,beta,uncertainty_reg,lr_multiplier,weight_decay,reg_coefficient,margin_weight,blockwise,no_gating,no_metanet,accuracy,model,gpu_id,time_minutes" > "$RESULTS_FILE"

# 要测试的数据集
DATASETS=("DTD")  # 根据需要调整

# 基础超参数搜索空间
BASE_THRESHOLDS=(0.1)
BETAS=(1.0 2.0)
UNCERTAINTY_REGS=(0.05)

# 新增测试参数搜索空间
LR_MULTIPLIERS=(30 50 70)  # gating_log_params的学习率倍率
WEIGHT_DECAYS=(0.0001 0.0005 0.001)  # 权重衰减
REG_COEFFICIENTS=(0.0005 0.001 0.002)  # beta_reg和threshold_reg中的系数
MARGIN_WEIGHTS=(0.00005 0.0001 0.0002)  # margin_loss的权重

# 设置实验标题
echo "========================================"
echo "多GPU超参数调优实验 - $(date)"
echo "模型: $MODEL"
echo "基础学习率: $BASE_LR"
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
    local lr_multiplier=$5
    local weight_decay=$6
    local reg_coefficient=$7
    local margin_weight=$8
    local blockwise=$9
    local no_gating=${10}
    local no_metanet=${11}
    local gpu_id=${12}
    local task_id=${13}

    # 为每个实验设置唯一端口，避免冲突
    local port=$((12000 + task_id))

    # 设置特定于此实验的保存目录
    local exp_name="exp_${dataset}_bt${base_threshold}_beta${beta}_ur${uncertainty_reg}"
    exp_name="${exp_name}_lrm${lr_multiplier}_wd${weight_decay}_rc${reg_coefficient}_mw${margin_weight}"
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
    cmd="$cmd --lr $BASE_LR"  # 固定基础学习率
    cmd="$cmd --base-threshold $base_threshold"
    cmd="$cmd --beta $beta"
    cmd="$cmd --uncertainty-reg $uncertainty_reg"
    cmd="$cmd --port $port"  # 添加端口号避免冲突

    # 添加内部参数 (假设这些参数已经被添加到train_with_adaptive_gating.py中)
    cmd="$cmd --lr-multiplier $lr_multiplier"
    cmd="$cmd --weight-decay $weight_decay"
    cmd="$cmd --reg-coefficient $reg_coefficient"
    cmd="$cmd --margin-weight $margin_weight"

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

        # 提取准确率结果
        local accuracy=$(grep -oP "Best validation accuracy: \K[0-9.]+(?=%)" "$log_file" | tail -1)
        if [ -z "$accuracy" ]; then
            # 尝试其他可能的格式
            accuracy=$(grep -oP "最佳验证准确率: \K[0-9.]+(?=%)" "$log_file" | tail -1)
            if [ -z "$accuracy" ]; then
                accuracy=$(grep -E "Accuracy: [0-9.]+" "$log_file" | tail -1 | grep -oP "Accuracy: \K[0-9.]+")
                if [ -z "$accuracy" ]; then
                    accuracy="N/A"
                fi
            fi
        fi

        # 记录结果到CSV
        echo "$dataset,$base_threshold,$beta,$uncertainty_reg,$lr_multiplier,$weight_decay,$reg_coefficient,$margin_weight,$blockwise,$no_gating,$no_metanet,$accuracy,$MODEL,$gpu_id,$duration_minutes" >> "$RESULTS_FILE"

        echo -e "\n完成实验: $exp_name, 准确率: $accuracy% (GPU $gpu_id), 耗时: ${duration_minutes}分钟"
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
    local lr_multiplier=$5
    local weight_decay=$6
    local reg_coefficient=$7
    local margin_weight=$8
    local blockwise=$9
    local no_gating=${10}
    local no_metanet=${11}

    EXPERIMENT_TASKS+=("$dataset $base_threshold $beta $uncertainty_reg $lr_multiplier $weight_decay $reg_coefficient $margin_weight $blockwise $no_gating $no_metanet")
}

# 创建实验任务列表
echo "准备实验任务..."

# 实验1: MetaNet带自适应门控 (原始基础实验)
for dataset in "${DATASETS[@]}"; do
    # 使用默认值
    add_experiment_task "$dataset" 0.05 1.0 0.01 50 0.0005 0.001 0.0001 true false false

    # MetaNet无门控
    add_experiment_task "$dataset" 0.05 1.0 0.01 50 0.0005 0.001 0.0001 true true false

    # Atlas方法
    add_experiment_task "$dataset" 0.05 1.0 0.01 50 0.0005 0.001 0.0001 true false true
done

# 实验2: 比较不同的base_threshold
for dataset in "${DATASETS[@]}"; do
    for base_threshold in "${BASE_THRESHOLDS[@]}"; do
        if [ "$base_threshold" != "0.05" ]; then  # 避免重复
            add_experiment_task "$dataset" $base_threshold 1.0 0.01 50 0.0005 0.001 0.0001 true false false
        fi
    done
done

# 实验3: 比较不同的beta
for dataset in "${DATASETS[@]}"; do
    for beta in "${BETAS[@]}"; do
        if [ "$beta" != "1.0" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 $beta 0.01 50 0.0005 0.001 0.0001 true false false
        fi
    done
done

# 实验4: 测试学习率倍率 (新增)
for dataset in "${DATASETS[@]}"; do
    for lr_multiplier in "${LR_MULTIPLIERS[@]}"; do
        if [ "$lr_multiplier" != "50" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 1.0 0.01 $lr_multiplier 0.0005 0.001 0.0001 true false false
        fi
    done
done

# 实验5: 测试权重衰减 (新增)
for dataset in "${DATASETS[@]}"; do
    for weight_decay in "${WEIGHT_DECAYS[@]}"; do
        if [ "$weight_decay" != "0.0005" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 1.0 0.01 50 $weight_decay 0.001 0.0001 true false false
        fi
    done
done

# 实验6: 测试正则化系数 (新增)
for dataset in "${DATASETS[@]}"; do
    for reg_coefficient in "${REG_COEFFICIENTS[@]}"; do
        if [ "$reg_coefficient" != "0.001" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 1.0 0.01 50 0.0005 $reg_coefficient 0.0001 true false false
        fi
    done
done

# 实验7: 测试边界损失权重 (新增)
for dataset in "${DATASETS[@]}"; do
    for margin_weight in "${MARGIN_WEIGHTS[@]}"; do
        if [ "$margin_weight" != "0.0001" ]; then  # 避免重复
            add_experiment_task "$dataset" 0.05 1.0 0.01 50 0.0005 0.001 $margin_weight true false false
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
    read -r dataset base_threshold beta uncertainty_reg lr_multiplier weight_decay reg_coefficient margin_weight blockwise no_gating no_metanet <<< "$task"

    # 启动实验
    pid=$(run_experiment "$dataset" "$base_threshold" "$beta" "$uncertainty_reg" "$lr_multiplier" "$weight_decay" "$reg_coefficient" "$margin_weight" "$blockwise" "$no_gating" "$no_metanet" "$gpu_id" "$i")
    PIDS+=($pid)
    echo $pid

    # 更新进度
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
results_file = '$RESULTS_FILE'
try:
    results = pd.read_csv(results_file)
    results['accuracy'] = pd.to_numeric(results['accuracy'], errors='coerce')

    # 创建摘要文件
    summary_file = '${LOG_DIR}/summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f'多GPU超参数调优实验摘要 - {datetime.now().strftime(\"%Y-%m-%d %H:%M\")}\\n')
        f.write('======================\\n\\n')

        # 实验概况
        f.write('实验概况:\\n')
        f.write(f'总实验数: {len(results)}\\n')
        f.write(f'已完成实验数: {len(results[results[\"accuracy\"] != \"N/A\"])}\\n')
        if results['accuracy'].notna().any():
            f.write(f'平均准确率: {results[\"accuracy\"].mean():.2f}%\\n')

        # 每个数据集的最佳参数
        f.write('\\n\\n每个数据集的最佳参数:\\n')
        for dataset in results['dataset'].unique():
            dataset_results = results[results['dataset'] == dataset]
            if dataset_results['accuracy'].notna().any():
                best_row = dataset_results.loc[dataset_results['accuracy'].idxmax()]
                f.write(f'\\n{dataset}:\\n')
                f.write(f'  最佳准确率: {best_row[\"accuracy\"]}%\\n')
                f.write(f'  基础参数: base_threshold={best_row[\"base_threshold\"]}, beta={best_row[\"beta\"]}, uncertainty_reg={best_row[\"uncertainty_reg\"]}\\n')
                f.write(f'  优化器参数: lr_multiplier={best_row[\"lr_multiplier\"]}, weight_decay={best_row[\"weight_decay\"]}\\n')
                f.write(f'  正则化参数: reg_coefficient={best_row[\"reg_coefficient\"]}, margin_weight={best_row[\"margin_weight\"]}\\n')
                f.write(f'  结构参数: blockwise={best_row[\"blockwise\"]}, no_gating={best_row[\"no_gating\"]}, no_metanet={best_row[\"no_metanet\"]}\\n')

        # 各方法的比较
        f.write('\\n\\n不同方法的比较:\\n')
        methods = []
        if (results['no_metanet'] == False).any() and (results['no_gating'] == False).any():
            methods.append(('MetaNet with Gating', (results['no_metanet'] == False) & (results['no_gating'] == False)))
        if (results['no_metanet'] == False).any() and (results['no_gating'] == True).any():
            methods.append(('MetaNet without Gating', (results['no_metanet'] == False) & (results['no_gating'] == True)))
        if (results['no_metanet'] == True).any():
            methods.append(('Atlas (Direct Features)', results['no_metanet'] == True))

        for method_name, mask in methods:
            method_results = results[mask]
            if not method_results.empty and method_results['accuracy'].notna().any():
                avg_acc = method_results['accuracy'].mean()
                f.write(f'\\n{method_name}:\\n')
                f.write(f'  平均准确率: {avg_acc:.2f}%\\n')

                # 每个数据集的结果
                for dataset in method_results['dataset'].unique():
                    dataset_method = method_results[method_results['dataset'] == dataset]
                    if not dataset_method.empty and dataset_method['accuracy'].notna().any():
                        best_acc = dataset_method['accuracy'].max()
                        f.write(f'  {dataset}: {best_acc:.2f}%\\n')

        # 参数敏感性分析 - 基础参数
        metanet_results = results[(results['no_metanet'] == False) & (results['no_gating'] == False)]
        if not metanet_results.empty and metanet_results['accuracy'].notna().any():
            f.write('\\n\\n基础参数敏感性分析:\\n')

            # Base threshold影响
            f.write('\\nBase Threshold影响:\\n')
            for bt in sorted(metanet_results['base_threshold'].unique()):
                bt_results = metanet_results[metanet_results['base_threshold'] == bt]
                if not bt_results.empty and bt_results['accuracy'].notna().any():
                    bt_avg = bt_results['accuracy'].mean()
                    f.write(f'  {bt}: {bt_avg:.2f}%\\n')

            # Beta影响
            f.write('\\nBeta影响:\\n')
            for beta in sorted(metanet_results['beta'].unique()):
                beta_results = metanet_results[metanet_results['beta'] == beta]
                if not beta_results.empty and beta_results['accuracy'].notna().any():
                    beta_avg = beta_results['accuracy'].mean()
                    f.write(f'  {beta}: {beta_avg:.2f}%\\n')

            # Uncertainty reg影响
            f.write('\\nUncertainty Regularization影响:\\n')
            for ur in sorted(metanet_results['uncertainty_reg'].unique()):
                ur_results = metanet_results[metanet_results['uncertainty_reg'] == ur]
                if not ur_results.empty and ur_results['accuracy'].notna().any():
                    ur_avg = ur_results['accuracy'].mean()
                    f.write(f'  {ur}: {ur_avg:.2f}%\\n')

        # 参数敏感性分析 - 新增参数
        if 'lr_multiplier' in results.columns:
            f.write('\\n\\n新增参数敏感性分析:\\n')

            # 学习率倍率影响
            f.write('\\n学习率倍率影响:\\n')
            for lrm in sorted(metanet_results['lr_multiplier'].unique()):
                lrm_results = metanet_results[metanet_results['lr_multiplier'] == lrm]
                if not lrm_results.empty and lrm_results['accuracy'].notna().any():
                    lrm_avg = lrm_results['accuracy'].mean()
                    f.write(f'  {lrm}: {lrm_avg:.2f}%\\n')

            # 权重衰减影响
            f.write('\\n权重衰减影响:\\n')
            for wd in sorted(metanet_results['weight_decay'].unique()):
                wd_results = metanet_results[metanet_results['weight_decay'] == wd]
                if not wd_results.empty and wd_results['accuracy'].notna().any():
                    wd_avg = wd_results['accuracy'].mean()
                    f.write(f'  {wd}: {wd_avg:.2f}%\\n')

            # 正则化系数影响
            f.write('\\n正则化系数影响:\\n')
            for rc in sorted(metanet_results['reg_coefficient'].unique()):
                rc_results = metanet_results[metanet_results['reg_coefficient'] == rc]
                if not rc_results.empty and rc_results['accuracy'].notna().any():
                    rc_avg = rc_results['accuracy'].mean()
                    f.write(f'  {rc}: {rc_avg:.2f}%\\n')

            # 边界损失权重影响
            f.write('\\n边界损失权重影响:\\n')
            for mw in sorted(metanet_results['margin_weight'].unique()):
                mw_results = metanet_results[metanet_results['margin_weight'] == mw]
                if not mw_results.empty and mw_results['accuracy'].notna().any():
                    mw_avg = mw_results['accuracy'].mean()
                    f.write(f'  {mw}: {mw_avg:.2f}%\\n')

        # 添加GPU使用统计
        f.write('\\n\\nGPU使用统计:\\n')
        gpu_stats = results.groupby('gpu_id').agg({
            'accuracy': ['count', 'mean'],
            'time_minutes': ['mean', 'sum']
        }).fillna(0)

        if not gpu_stats.empty:
            for gpu_id, stats in gpu_stats.iterrows():
                if gpu_id >= 0 and gpu_id < $GPU_COUNT:  # 确保GPU ID有效
                    count = int(stats[('accuracy', 'count')])
                    if count > 0:  # 只显示有实验的GPU
                        mean_acc = stats[('accuracy', 'mean')]
                        mean_time = stats[('time_minutes', 'mean')]
                        total_time = stats[('time_minutes', 'sum')]
                        f.write(f'  GPU {gpu_id}: 运行了 {count} 个实验, 平均准确率: {mean_acc:.2f}%, ')
                        f.write(f'平均每个实验 {mean_time:.1f} 分钟, 总时间 {total_time:.1f} 分钟\\n')

    print(f'成功生成摘要文件: {summary_file}')
except Exception as e:
    print(f'生成摘要时出错: {e}')
    print(f'尝试读取的文件路径: {results_file}')
    print(f'当前工作目录: {os.getcwd()}')
"

echo "实验完成！结果保存在: $RESULTS_FILE"
echo "摘要报告: ${LOG_DIR}/summary.txt"

# 可选: 通知实验完成
HOSTNAME=$(hostname)
echo -e "\\a"  # 发出哔声提醒
echo "实验已在 $HOSTNAME 上完成 ($(date))"