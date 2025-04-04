#!/bin/bash

# 自动模型评估脚本
# 功能：根据数据集名称和模型类型自动查找最佳模型并评估

# 默认参数
BASE_DIR="/home/haichao/zby/MetaNet-Bayes"
CHECKPOINTS_DIR="${BASE_DIR}/checkpoints_hyperparameter_tuning"
DATA_LOCATION="${BASE_DIR}"
MODEL="ViT-L-14"
BATCH_SIZE=128
NUM_WORKERS=4
GPU_ID=0
VERBOSE=false
FORCE_REEVALUATE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset)
        DATASET="$2"
        shift
        shift
        ;;
        --model)
        MODEL="$2"
        shift
        shift
        ;;
        --gpu)
        GPU_ID="$2"
        shift
        shift
        ;;
        --base-dir)
        BASE_DIR="$2"
        CHECKPOINTS_DIR="${BASE_DIR}/checkpoints_hyperparameter_tuning"
        DATA_LOCATION="${BASE_DIR}"
        shift
        shift
        ;;
        --checkpoints-dir)
        CHECKPOINTS_DIR="$2"
        shift
        shift
        ;;
        --data-location)
        DATA_LOCATION="$2"
        shift
        shift
        ;;
        --batch-size)
        BATCH_SIZE="$2"
        shift
        shift
        ;;
        --num-workers)
        NUM_WORKERS="$2"
        shift
        shift
        ;;
        --verbose)
        VERBOSE=true
        shift
        ;;
        --force)
        FORCE_REEVALUATE=true
        shift
        ;;
        --help)
        echo "自动模型评估脚本"
        echo "用法: $0 --dataset DATASET [选项]"
        echo ""
        echo "必要参数:"
        echo "  --dataset DATASET     要评估的数据集名称(例如 SVHN, GTSRB)"
        echo ""
        echo "可选参数:"
        echo "  --model MODEL         模型类型(默认: ViT-L-14)"
        echo "  --gpu GPU_ID          GPU ID(默认: 0)"
        echo "  --base-dir DIR        基础目录(默认: /home/haichao/zby/MetaNet-Bayes)"
        echo "  --checkpoints-dir DIR 检查点目录(默认: BASE_DIR/checkpoints_hyperparameter_tuning)"
        echo "  --data-location DIR   数据位置(默认: BASE_DIR)"
        echo "  --batch-size SIZE     批量大小(默认: 128)"
        echo "  --num-workers NUM     工作线程数(默认: 4)"
        echo "  --verbose             输出详细信息"
        echo "  --force               强制重新评估(即使已有结果)"
        exit 0
        ;;
        *)
        echo "未知参数: $1"
        echo "使用 --help 查看帮助"
        exit 1
        ;;
    esac
done

# 验证必要参数
if [ -z "$DATASET" ]; then
    echo "错误: 必须指定数据集 (--dataset)"
    echo "使用 --help 查看帮助"
    exit 1
fi

# 创建临时目录保存日志和输出
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_DIR}/evaluation_results/${DATASET}_${MODEL}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/eval_log.txt"
DEBUG_FILE="${OUTPUT_DIR}/debug_log.txt"
RESULTS_FILE="${OUTPUT_DIR}/results.json"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

# 记录脚本执行信息
echo "========== 自动模型评估 ==========" | tee -a "$LOG_FILE"
echo "数据集: $DATASET" | tee -a "$LOG_FILE"
echo "模型: $MODEL" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "检查点目录: $CHECKPOINTS_DIR" | tee -a "$LOG_FILE"
echo "数据位置: $DATA_LOCATION" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"

# 查找与数据集相关的所有实验目录
echo "正在查找与数据集 $DATASET 相关的实验..." | tee -a "$LOG_FILE"
mapfile -t EXP_DIRS < <(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "exp_${DATASET}_*" | sort)

if [ ${#EXP_DIRS[@]} -eq 0 ]; then
    echo "错误: 没有找到与数据集 $DATASET 相关的实验目录" | tee -a "$LOG_FILE"
    exit 1
fi

echo "找到 ${#EXP_DIRS[@]} 个相关实验目录" | tee -a "$LOG_FILE"
if $VERBOSE; then
    for dir in "${EXP_DIRS[@]}"; do
        echo "  $dir" | tee -a "$LOG_FILE"
    done
fi

# 从实验目录提取参数的函数
extract_params_from_dir() {
    local dir_name="$1"
    local base_name=$(basename "$dir_name")

    # 提取参数值
    local bt=$(echo "$base_name" | grep -oP "bt\K[0-9.]+(?=_)" || echo "0.05")
    local beta=$(echo "$base_name" | grep -oP "beta\K[0-9.]+(?=_)" || echo "1.0")
    local ur=$(echo "$base_name" | grep -oP "ur\K[0-9.]+(?=_)" || echo "0.01")
    local lrm=$(echo "$base_name" | grep -oP "lrm\K[0-9.]+(?=_)" || echo "50.0")
    local wd=$(echo "$base_name" | grep -oP "wd\K[0-9.]+(?=_)" || echo "0.0005")
    local regc=$(echo "$base_name" | grep -oP "regc\K[0-9.]+(?=_)" || echo "0.001")
    local mw=$(echo "$base_name" | grep -oP "mw\K[0-9.]+(?=_)" || echo "0.0001")
    local blockwise=false
    if [[ "$base_name" == *"_blockwise"* ]]; then
        blockwise=true
    fi
    local no_gating=false
    if [[ "$base_name" == *"_nogating"* ]]; then
        no_gating=true
    fi
    local no_metanet=false
    if [[ "$base_name" == *"_nometanet"* ]]; then
        no_metanet=true
    fi

    # 构建参数字符串(JSON格式)
    echo "{\"base_threshold\": $bt, \"beta\": $beta, \"uncertainty_reg\": $ur, \"lr_multiplier\": $lrm, \"weight_decay\": $wd, \"reg_coefficient\": $regc, \"margin_weight\": $mw, \"blockwise\": $blockwise, \"no_gating\": $no_gating, \"no_metanet\": $no_metanet}"
}

# 读取缓存的评估结果(如果有)
CACHE_FILE="${BASE_DIR}/evaluation_results/${DATASET}_${MODEL}_results_cache.json"
BEST_EXP_DIR=""
BEST_ACCURACY=0
BEST_PARAMS=""

if [ -f "$CACHE_FILE" ] && [ "$FORCE_REEVALUATE" = false ]; then
    echo "找到缓存的评估结果，读取中..." | tee -a "$LOG_FILE"

    # 使用Python解析JSON缓存
    python3 -c "
import json
import sys

try:
    with open('$CACHE_FILE', 'r') as f:
        cache = json.load(f)

    # 找到准确率最高的实验
    best_exp = None
    best_acc = 0

    for exp in cache:
        accuracy = float(exp.get('accuracy', 0))
        if accuracy > best_acc:
            best_acc = accuracy
            best_exp = exp

    if best_exp:
        print(f\"最佳实验: {best_exp['exp_dir']}\\n\"
              f\"测试准确率: {best_exp['accuracy']}%\\n\"
              f\"参数: {json.dumps(best_exp['params'])}\\n\")
    else:
        print('缓存中没有有效的结果')
        sys.exit(1)

except Exception as e:
    print(f'读取缓存失败: {e}')
    sys.exit(1)
" | tee -a "$LOG_FILE"

    # 如果缓存读取成功，使用缓存中的最佳模型
    if [ $? -eq 0 ]; then
        # 提取最佳模型信息
        BEST_ACCURACY=$(grep -oP "测试准确率: \K[0-9.]+(?=%)" <<< "$(grep "测试准确率:" "$LOG_FILE" | tail -1)")
        BEST_EXP_DIR=$(grep -oP "最佳实验: \K.*" <<< "$(grep "最佳实验:" "$LOG_FILE" | tail -1)")
        BEST_PARAMS=$(grep -oP "参数: \K.*" <<< "$(grep "参数:" "$LOG_FILE" | tail -1)")

        echo "使用缓存中的最佳实验: $BEST_EXP_DIR" | tee -a "$LOG_FILE"
    else
        echo "缓存读取失败或没有有效结果，将进行完整评估" | tee -a "$LOG_FILE"
    fi
fi

# 如果没有从缓存中获取最佳模型，遍历所有实验目录进行评估
if [ -z "$BEST_EXP_DIR" ]; then
    echo "开始评估所有实验..." | tee -a "$LOG_FILE"

    # 创建数组保存所有评估结果
    ALL_RESULTS=()

    for exp_dir in "${EXP_DIRS[@]}"; do
        exp_name=$(basename "$exp_dir")
        echo "评估实验: $exp_name" | tee -a "$LOG_FILE"

        # 提取实验参数
        params=$(extract_params_from_dir "$exp_dir")
        echo "提取的参数: $params" | tee -a "$LOG_FILE"

        # 在实验目录中查找模型目录
        model_dir="${exp_dir}/${MODEL}"

        if [ ! -d "$model_dir" ]; then
            echo "警告: 在实验 $exp_name 中未找到模型目录" | tee -a "$LOG_FILE"
            continue
        fi

        # 从模型参数中提取参数
        bt=$(echo "$params" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['base_threshold'])")
        beta=$(echo "$params" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['beta'])")
        blockwise=$(echo "$params" | python3 -c "import json,sys; print(str(json.loads(sys.stdin.read())['blockwise']).lower())")
        no_gating=$(echo "$params" | python3 -c "import json,sys; print(str(json.loads(sys.stdin.read())['no_gating']).lower())")
        no_metanet=$(echo "$params" | python3 -c "import json,sys; print(str(json.loads(sys.stdin.read())['no_metanet']).lower())")

        # 构建评估命令
        eval_cmd="CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.eval_with_adaptive_gating"
        eval_cmd="$eval_cmd --model $MODEL"
        eval_cmd="$eval_cmd --model-dir $model_dir"
        eval_cmd="$eval_cmd --data-location $DATA_LOCATION"
        eval_cmd="$eval_cmd --batch-size $BATCH_SIZE"
        eval_cmd="$eval_cmd --num-workers $NUM_WORKERS"
        eval_cmd="$eval_cmd --datasets $DATASET"
        eval_cmd="$eval_cmd --base-threshold $bt"
        eval_cmd="$eval_cmd --beta $beta"
        eval_cmd="$eval_cmd --verbose"

        # 添加blockwise参数(如果需要)
        if [ "$blockwise" = "true" ]; then
            eval_cmd="$eval_cmd --blockwise-coef"
        fi

        # 添加no_gating参数(如果需要)
        if [ "$no_gating" = "true" ]; then
            eval_cmd="$eval_cmd --no-gating"
        fi

        # 添加no_metanet参数(如果需要)
        if [ "$no_metanet" = "true" ]; then
            eval_cmd="$eval_cmd --no-metanet"
        fi

        # 记录评估命令
        # echo "评估命令: $eval_cmd" | tee -a "$LOG_FILE"

        # 执行评估并将输出重定向到临时文件
        exp_log_file="${OUTPUT_DIR}/${exp_name}_eval.log"
        echo "执行评估，结果将保存到 $exp_log_file..." | tee -a "$LOG_FILE"
        eval "$eval_cmd" > "$exp_log_file" 2>&1

        # 检查评估是否成功
        if [ $? -ne 0 ]; then
            echo "评估失败，查看日志: $exp_log_file" | tee -a "$LOG_FILE"
            continue
        fi

        # 从日志提取准确率
        accuracy=$(grep -oP "Accuracy: \K[0-9.]+(?=%)" "$exp_log_file" | tail -1)

        # 如果找不到，尝试其他格式
        if [ -z "$accuracy" ]; then
            accuracy=$(grep -oP "Test accuracy: \K[0-9.]+(?=%)" "$exp_log_file" | tail -1)
        fi

        if [ -z "$accuracy" ]; then
            accuracy=$(grep -oP "test accuracy: \K[0-9.]+" "$exp_log_file" | tail -1)
        fi

        if [ -z "$accuracy" ]; then
            accuracy=$(grep -i "accuracy" "$exp_log_file" | grep -oP "[0-9]+\.[0-9]+(?=%)" | tail -1)
        fi

        if [ -z "$accuracy" ]; then
            echo "警告: 未能从评估日志中提取准确率" | tee -a "$LOG_FILE"
            accuracy="0.0"
        fi

        echo "测试准确率: $accuracy%" | tee -a "$LOG_FILE"

        # 保存结果
        result="{\"exp_dir\": \"$exp_dir\", \"accuracy\": $accuracy, \"params\": $params}"
        ALL_RESULTS+=("$result")

        # 更新最佳结果
        if (( $(echo "$accuracy > $BEST_ACCURACY" | bc -l) )); then
            BEST_ACCURACY=$accuracy
            BEST_EXP_DIR=$exp_dir
            BEST_PARAMS=$params
            echo "新的最佳模型! 准确率: $BEST_ACCURACY%" | tee -a "$LOG_FILE"
        fi

        echo "---------------------------------" | tee -a "$LOG_FILE"
    done

    # 保存所有结果到缓存文件
    if [ ${#ALL_RESULTS[@]} -gt 0 ]; then
        echo "[${ALL_RESULTS[*]}]" > "$CACHE_FILE"
        echo "评估结果已缓存到: $CACHE_FILE" | tee -a "$LOG_FILE"
    fi
fi

# 如果找到了最佳模型，输出总结信息
if [ -n "$BEST_EXP_DIR" ] && [ "$BEST_ACCURACY" != "0.0" ]; then
    # 准备总结信息
    echo "==================== 评估总结 ====================" | tee -a "$SUMMARY_FILE"
    echo "数据集: $DATASET" | tee -a "$SUMMARY_FILE"
    echo "模型: $MODEL" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "最佳实验目录: $BEST_EXP_DIR" | tee -a "$SUMMARY_FILE"
    echo "最佳测试准确率: $BEST_ACCURACY%" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # 提取参数并格式化显示
    echo "最佳模型参数:" | tee -a "$SUMMARY_FILE"
    echo "$BEST_PARAMS" | python3 -c "
import json
import sys

params = json.loads(sys.stdin.read())
for key, value in params.items():
    print(f'  {key} = {value}')
" | tee -a "$SUMMARY_FILE"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "评估日期: $(date)" | tee -a "$SUMMARY_FILE"
    echo "详细日志: $LOG_FILE" | tee -a "$SUMMARY_FILE"
    echo "==================================================" | tee -a "$SUMMARY_FILE"

    # 生成Python格式的最佳参数
    echo "# $DATASET 数据集的最佳参数" > "${OUTPUT_DIR}/best_params.py"
    echo "# 测试准确率: $BEST_ACCURACY%" >> "${OUTPUT_DIR}/best_params.py"
    echo "" >> "${OUTPUT_DIR}/best_params.py"

    echo "$BEST_PARAMS" | python3 -c "
import json
import sys

params = json.loads(sys.stdin.read())
for key, value in params.items():
    if isinstance(value, bool):
        if key == 'blockwise':
            print(f'args.blockwise_coef = {str(value).lower()}')
        else:
            print(f'args.{key} = {str(value).lower()}')
    else:
        print(f'args.{key} = {value}')
" >> "${OUTPUT_DIR}/best_params.py"

    # 打印总结信息
    cat "$SUMMARY_FILE"

    echo ""
    echo "最佳参数以Python格式保存在: ${OUTPUT_DIR}/best_params.py"
    echo "所有评估结果保存在: $OUTPUT_DIR"

    # 创建模型评估结果的符号链接
    LATEST_LINK="${BASE_DIR}/evaluation_results/${DATASET}_${MODEL}_latest"
    rm -f "$LATEST_LINK" 2>/dev/null
    ln -s "$OUTPUT_DIR" "$LATEST_LINK"
    echo "创建了指向最新结果的符号链接: $LATEST_LINK"
else
    echo "错误: 未能找到或评估最佳模型" | tee -a "$LOG_FILE"
    exit 1
fi

echo "评估完成!"