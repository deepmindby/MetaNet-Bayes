#!/bin/bash

# 自动模型评估脚本
# 功能：根据数据集名称和模型类型自动查找最佳模型并评估

# 默认参数
BASE_DIR="/home/haichao/zby/MetaNet-Bayes"
CHECKPOINTS_DIR="${BASE_DIR}/checkpoints_hyperparameter_tuning"
DATA_LOCATION="${BASE_DIR}"
MODEL="ViT-B-32"
BATCH_SIZE=128
NUM_WORKERS=4
GPU_ID=0
VERBOSE=false
FORCE_REEVALUATE=false
GATING_NO_METANET=false

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
        --gating-no-metanet)
        GATING_NO_METANET=true
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
        echo "  --model MODEL         模型类型(默认: ViT-B-32)"
        echo "  --gpu GPU_ID          GPU ID(默认: 0)"
        echo "  --base-dir DIR        基础目录(默认: /home/haichao/zby/MetaNet-Bayes)"
        echo "  --checkpoints-dir DIR 检查点目录(默认: BASE_DIR/checkpoints_hyperparameter_tuning)"
        echo "  --data-location DIR   数据位置(默认: BASE_DIR)"
        echo "  --batch-size SIZE     批量大小(默认: 128)"
        echo "  --num-workers NUM     工作线程数(默认: 4)"
        echo "  --verbose             输出详细信息"
        echo "  --force               强制重新评估(即使已有结果)"
        echo "  --gating-no-metanet   评估Atlas+Gating模型"
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
if [ "$GATING_NO_METANET" = true ]; then
    echo "评估模式: Atlas+Gating" | tee -a "$LOG_FILE"
fi
echo "===================================" | tee -a "$LOG_FILE"

# 实验目录搜索模式
EXP_SEARCH_PATTERN="exp_${DATASET}_*"
if [ "$GATING_NO_METANET" = true ]; then
    # 如果评估Atlas+Gating模型，则特别查找包含gatingnometanet的目录
    EXP_SEARCH_PATTERN="exp_${DATASET}_*gatingnometanet*"
fi

# 查找与数据集相关的所有实验目录
echo "正在查找与数据集 $DATASET 相关的实验..." | tee -a "$LOG_FILE"
mapfile -t EXP_DIRS < <(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "$EXP_SEARCH_PATTERN" | sort)

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

    local gating_no_metanet=false
    if [[ "$base_name" == *"_gatingnometanet"* ]]; then
        gating_no_metanet=true
    fi

    # 构建参数字符串(JSON格式)
    echo "{\"base_threshold\": $bt, \"beta\": $beta, \"uncertainty_reg\": $ur, \"lr_multiplier\": $lrm, \"weight_decay\": $wd, \"reg_coefficient\": $regc, \"margin_weight\": $mw, \"blockwise\": $blockwise, \"no_gating\": $no_gating, \"no_metanet\": $no_metanet, \"gating_no_metanet\": $gating_no_metanet}"
}

# 读取缓存的评估结果(如果有)
CACHE_FILE="${BASE_DIR}/evaluation_results/${DATASET}_${MODEL}_results_cache.json"
BEST_EXP_DIR=""
BEST_ACCURACY=0
BEST_PARAMS=""

if [ -f "$CACHE_FILE" ] && [ "$FORCE_REEVALUATE" = false ]; then
    echo "找到缓存的评估结果，读取中..." | tee -a "$LOG_FILE"

    # 使用安全方法解析JSON缓存
    # 修复: 使用标准的JSON解析方法，防止格式错误
    if python3 -c "
import json
import sys

try:
    with open('$CACHE_FILE', 'r') as f:
        content = f.read()
        cache = json.loads(content)

    # 确保cache是列表
    if not isinstance(cache, list):
        print('缓存文件格式错误，需要JSON数组')
        sys.exit(1)

    # 找到准确率最高的实验
    best_exp = None
    best_acc = 0

    for exp in cache:
        if not isinstance(exp, dict):
            continue

        # 如果指定了atlas+gating模式，只考虑gating_no_metanet=true的结果
        if $GATING_NO_METANET and not exp.get('params', {}).get('gating_no_metanet', False):
            continue

        # 如果没有指定atlas+gating模式，但结果是atlas+gating，则跳过
        if not $GATING_NO_METANET and exp.get('params', {}).get('gating_no_metanet', False):
            continue

        accuracy = float(exp.get('accuracy', 0))
        if accuracy > best_acc:
            best_acc = accuracy
            best_exp = exp

    if best_exp:
        print(f\"最佳实验: {best_exp.get('exp_dir', 'unknown')}\\n\"
              f\"测试准确率: {best_exp.get('accuracy', 0)}%\\n\"
              f\"参数: {json.dumps(best_exp.get('params', {}))}\\n\")
    else:
        print('缓存中没有有效的结果')
        sys.exit(1)

except Exception as e:
    print(f'读取缓存失败: {e}')
    sys.exit(1)
" >> "$LOG_FILE" 2>&1; then
        # 提取最佳模型信息
        BEST_ACCURACY=$(grep -oP "测试准确率: \K[0-9.]+(?=%)" <<< "$(grep "测试准确率:" "$LOG_FILE" | tail -1)")
        BEST_EXP_DIR=$(grep -oP "最佳实验: \K.*" <<< "$(grep "最佳实验:" "$LOG_FILE" | tail -1)")
        BEST_PARAMS=$(grep -oP "参数: \K.*" <<< "$(grep "参数:" "$LOG_FILE" | tail -1)")

        echo "使用缓存中的最佳实验: $BEST_EXP_DIR" | tee -a "$LOG_FILE"
    else
        echo "缓存读取失败或没有有效结果，将进行完整评估" | tee -a "$LOG_FILE"
        # 确保清除可能的部分读取结果
        BEST_EXP_DIR=""
        BEST_ACCURACY=0
        BEST_PARAMS=""
    fi
fi

# 如果没有从缓存中获取最佳模型，遍历所有实验目录进行评估
if [ -z "$BEST_EXP_DIR" ]; then
    echo "开始评估所有实验..." | tee -a "$LOG_FILE"

    # 创建数组保存所有评估结果
    ALL_RESULTS=()
    VALID_RESULTS_COUNT=0
    FAILED_RESULTS_COUNT=0

    for exp_dir in "${EXP_DIRS[@]}"; do
        exp_name=$(basename "$exp_dir")
        echo "评估实验: $exp_name" | tee -a "$LOG_FILE"

        # 提取实验参数
        params=$(extract_params_from_dir "$exp_dir")
        echo "提取的参数: $params" | tee -a "$LOG_FILE"

        # 检查是否匹配我们希望评估的模型类型
        is_gating_no_metanet=$(echo "$params" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['gating_no_metanet'])")

        if [ "$GATING_NO_METANET" = true ] && [ "$is_gating_no_metanet" != "True" ]; then
            echo "跳过非Atlas+Gating模型" | tee -a "$LOG_FILE"
            continue
        fi

        if [ "$GATING_NO_METANET" = false ] && [ "$is_gating_no_metanet" = "True" ]; then
            echo "跳过Atlas+Gating模型（当前未指定评估此类型）" | tee -a "$LOG_FILE"
            continue
        fi

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
        gating_no_metanet=$(echo "$params" | python3 -c "import json,sys; print(str(json.loads(sys.stdin.read())['gating_no_metanet']).lower())")

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
        eval_cmd="$eval_cmd --debug"  # 添加debug标志以获取更多信息

        # 添加模型类型特定参数
        if [ "$blockwise" = "true" ]; then
            eval_cmd="$eval_cmd --blockwise-coef"
        fi

        if [ "$no_gating" = "true" ]; then
            eval_cmd="$eval_cmd --no-gating"
        fi

        if [ "$no_metanet" = "true" ]; then
            eval_cmd="$eval_cmd --no-metanet"
        fi

        if [ "$gating_no_metanet" = "true" ]; then
            eval_cmd="$eval_cmd --gating-no-metanet"
        fi

        # 记录评估命令
        echo "评估命令: $eval_cmd" | tee -a "$DEBUG_FILE"

        # 执行评估并将输出重定向到临时文件
        exp_log_file="${OUTPUT_DIR}/${exp_name}_eval.log"
        echo "执行评估，结果将保存到 $exp_log_file..." | tee -a "$LOG_FILE"
        eval "$eval_cmd" > "$exp_log_file" 2>&1

        # 检查评估是否成功
        if [ $? -ne 0 ]; then
            echo "评估失败，查看日志: $exp_log_file" | tee -a "$LOG_FILE"
            FAILED_RESULTS_COUNT=$((FAILED_RESULTS_COUNT+1))
            continue
        fi

        # 改进: 支持多种格式的准确率提取 - 尝试多种模式
        accuracy=""

        # 尝试"Accuracy: XX.XX%"格式
        if grep -q "Accuracy: [0-9.]\+%" "$exp_log_file"; then
            accuracy=$(grep -oP "Accuracy: \K[0-9.]+(?=%)" "$exp_log_file" | tail -1)
        fi

        # 尝试"Test accuracy: XX.XX%"格式
        if [ -z "$accuracy" ] && grep -q "Test accuracy: [0-9.]\+%" "$exp_log_file"; then
            accuracy=$(grep -oP "Test accuracy: \K[0-9.]+(?=%)" "$exp_log_file" | tail -1)
        fi

        # 尝试"test accuracy: XX.XX"格式
        if [ -z "$accuracy" ] && grep -q "test accuracy: [0-9.]\+" "$exp_log_file"; then
            accuracy=$(grep -oP "test accuracy: \K[0-9.]+" "$exp_log_file" | tail -1)
        fi

        # 尝试任何包含"accuracy"和数字的行
        if [ -z "$accuracy" ]; then
            accuracy=$(grep -i "accuracy" "$exp_log_file" | grep -oP "[0-9]+\.[0-9]+(?=%)" | tail -1)
        fi

        # 最后检查是否获取到了准确率
        if [ -z "$accuracy" ]; then
            # 失败时，检查常见错误
            if grep -q "Could not find" "$exp_log_file"; then
                error_msg="找不到匹配的模型文件"
            elif grep -q "out of memory" "$exp_log_file"; then
                error_msg="GPU内存不足"
            elif grep -q "RuntimeError" "$exp_log_file"; then
                error_msg="运行时错误"
            else
                error_msg="未知错误，可能是日志格式问题"
            fi

            echo "警告: 未能从评估日志中提取准确率 - $error_msg" | tee -a "$LOG_FILE"
            echo "请检查日志文件: $exp_log_file" | tee -a "$LOG_FILE"
            echo "记录为故障案例" | tee -a "$LOG_FILE"
            FAILED_RESULTS_COUNT=$((FAILED_RESULTS_COUNT+1))
            accuracy="0.0"
        else
            VALID_RESULTS_COUNT=$((VALID_RESULTS_COUNT+1))
        fi

        # 检查是否有gating ratio信息
        gating_ratio=""
        if grep -q "gating ratio" "$exp_log_file"; then
            gating_ratio=$(grep -oP "gating ratio=\K[0-9.]+(?=\))" "$exp_log_file" | tail -1)
            if [ -n "$gating_ratio" ]; then
                # 转换为百分比
                gating_ratio=$(echo "$gating_ratio * 100" | bc)
            fi
        fi

        # 获取Atlas with Gating特有的参数变化信息
        threshold_change=""
        beta_change=""
        if [ "$gating_no_metanet" = "true" ] && grep -q "αT.*change:" "$exp_log_file"; then
            threshold_change=$(grep -oP "αT:.*?\(\K[+-][0-9.]+(?=%\))" "$exp_log_file" | tail -1)
            beta_change=$(grep -oP "β:.*?\(\K[+-][0-9.]+(?=%\))" "$exp_log_file" | tail -1)
        fi

        echo "测试准确率: $accuracy%" | tee -a "$LOG_FILE"
        if [ -n "$gating_ratio" ]; then
            echo "门控比例: $gating_ratio%" | tee -a "$LOG_FILE"
        fi
        if [ -n "$threshold_change" ] && [ -n "$beta_change" ]; then
            echo "参数变化: αT: ${threshold_change}%, β: ${beta_change}%" | tee -a "$LOG_FILE"
        fi

        # 保存结果
        result="{\"exp_dir\": \"$exp_dir\", \"accuracy\": $accuracy, \"params\": $params"
        if [ -n "$gating_ratio" ]; then
            result="$result, \"gating_ratio\": $gating_ratio"
        fi
        if [ -n "$threshold_change" ] && [ -n "$beta_change" ]; then
            result="$result, \"threshold_change\": $threshold_change, \"beta_change\": $beta_change"
        fi
        result="$result}"
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
        # 确保正确格式化JSON数组
        echo "[" > "$CACHE_FILE"
        for i in "${!ALL_RESULTS[@]}"; do
            echo "${ALL_RESULTS[$i]}" >> "$CACHE_FILE"
            if [ $i -lt $((${#ALL_RESULTS[@]} - 1)) ]; then
                echo "," >> "$CACHE_FILE"
            fi
        done
        echo "]" >> "$CACHE_FILE"

        # 验证缓存文件是否为有效JSON
        if ! python3 -c "import json; json.load(open('$CACHE_FILE'))" &>/dev/null; then
            echo "警告: 生成的缓存文件不是有效的JSON，尝试修复..." | tee -a "$LOG_FILE"
            # 简单修复: 重新创建缓存文件
            echo "[" > "$CACHE_FILE"
            for i in "${!ALL_RESULTS[@]}"; do
                # 仅保留有效的JSON对象
                if python3 -c "import json; json.loads('${ALL_RESULTS[$i]}')" &>/dev/null; then
                    echo "${ALL_RESULTS[$i]}" >> "$CACHE_FILE"
                    if [ $i -lt $((${#ALL_RESULTS[@]} - 1)) ]; then
                        echo "," >> "$CACHE_FILE"
                    fi
                fi
            done
            echo "]" >> "$CACHE_FILE"
        fi

        echo "评估结果已缓存到: $CACHE_FILE" | tee -a "$LOG_FILE"
    fi

    # 报告模型评估统计
    echo "评估统计:" | tee -a "$LOG_FILE"
    echo "  成功评估: $VALID_RESULTS_COUNT" | tee -a "$LOG_FILE"
    echo "  评估失败: $FAILED_RESULTS_COUNT" | tee -a "$LOG_FILE"
    echo "  总实验数: ${#EXP_DIRS[@]}" | tee -a "$LOG_FILE"

    # 分析模型精度为0.0的原因
    if [ $FAILED_RESULTS_COUNT -gt 0 ]; then
        echo "为什么有些模型精度为0.0?" | tee -a "$LOG_FILE"
        echo "可能的原因:" | tee -a "$LOG_FILE"
        echo "1. 模型文件未正确生成或找不到 - 检查训练是否成功完成" | tee -a "$LOG_FILE"
        echo "2. 评估命令参数不匹配训练参数 - 确保参数提取正确" | tee -a "$LOG_FILE"
        echo "3. 评估过程中出现运行时错误 - 查看具体日志文件" | tee -a "$LOG_FILE"
        echo "4. 无法从日志中提取准确率 - 日志格式可能不一致" | tee -a "$LOG_FILE"
        echo "建议: 检查各个日志文件以确定具体问题" | tee -a "$LOG_FILE"
    fi
fi

# 如果找到了最佳模型，输出总结信息
if [ -n "$BEST_EXP_DIR" ] && [ "$BEST_ACCURACY" != "0.0" ]; then
    # 准备总结信息
    echo "==================== 评估总结 ====================" | tee -a "$SUMMARY_FILE"
    echo "数据集: $DATASET" | tee -a "$SUMMARY_FILE"
    echo "模型: $MODEL" | tee -a "$SUMMARY_FILE"

    if [ "$GATING_NO_METANET" = true ]; then
        echo "评估模式: Atlas+Gating" | tee -a "$SUMMARY_FILE"
    fi

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
    if [ "$GATING_NO_METANET" = true ]; then
        LATEST_LINK="${LATEST_LINK}_atlas_gating"
    fi
    rm -f "$LATEST_LINK" 2>/dev/null
    ln -s "$OUTPUT_DIR" "$LATEST_LINK"
    echo "创建了指向最新结果的符号链接: $LATEST_LINK"
else
    echo "警告: 未能找到或评估最佳模型，请检查日志文件以了解详情" | tee -a "$LOG_FILE"
    # 尝试提供更多信息来帮助诊断问题
    if [ -z "$BEST_EXP_DIR" ]; then
        echo "  原因: 没有找到有效的实验目录" | tee -a "$LOG_FILE"
    elif [ "$BEST_ACCURACY" = "0.0" ]; then
        echo "  原因: 所有模型的评估精度都为0.0，请检查评估日志" | tee -a "$LOG_FILE"
        echo "  建议: 使用 --debug 标志查看更详细的评估输出" | tee -a "$LOG_FILE"
    fi
    exit 1
fi

echo "评估完成!"