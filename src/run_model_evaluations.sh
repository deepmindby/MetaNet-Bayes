#!/bin/bash

# Model evaluation script
# This script evaluates different models with different settings
# and ensures results are stored in model-specific directories

# Set default parameters
SAVE_DIR="./results"
MODEL_DIR="./checkpoints_precomputed"
DATA_DIR="./MetaNet-Bayes"
BATCH_SIZE=128
NUM_WORKERS=4

# Define models to evaluate - each will have its own separate directory
MODELS=("ViT-B-32" "ViT-L-14")

# Define datasets to evaluate on
DATASETS=("Cars" "DTD" "EuroSAT" "GTSRB" "MNIST" "RESISC45" "SUN397" "SVHN")

# Make sure base directories exist
mkdir -p "$SAVE_DIR"

# Function to run standard evaluation
run_standard_eval() {
    MODEL=$1
    echo "===================================================="
    echo "Running standard evaluation for model: $MODEL"
    echo "===================================================="

    python src/eval_with_precomputed.py \
        --model "$MODEL" \
        --save-dir "$SAVE_DIR" \
        --model-dir "$MODEL_DIR" \
        --data-location "$DATA_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --datasets "${DATASETS[@]}" \
        --verbose
}

# Function to run causal evaluation
run_causal_eval() {
    MODEL=$1
    echo "===================================================="
    echo "Running causal evaluation for model: $MODEL"
    echo "===================================================="

    python src/eval_with_precomputed.py \
        --model "$MODEL" \
        --save-dir "$SAVE_DIR" \
        --model-dir "$MODEL_DIR" \
        --data-location "$DATA_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --datasets "${DATASETS[@]}" \
        --causal_intervention \
        --top_k_ratio 0.1 \
        --verbose
}

# Function to run blockwise evaluation
run_blockwise_eval() {
    MODEL=$1
    echo "===================================================="
    echo "Running blockwise evaluation for model: $MODEL"
    echo "===================================================="

    python src/eval_with_precomputed.py \
        --model "$MODEL" \
        --save-dir "$SAVE_DIR" \
        --model-dir "$MODEL_DIR" \
        --data-location "$DATA_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --datasets "${DATASETS[@]}" \
        --blockwise-coef \
        --verbose
}

# Function to run adaptive gating evaluation
run_adaptive_gating_eval() {
    MODEL=$1
    echo "===================================================="
    echo "Running adaptive gating evaluation for model: $MODEL"
    echo "===================================================="

    python src/eval_with_adaptive_gating.py \
        --model "$MODEL" \
        --save-dir "$SAVE_DIR" \
        --model-dir "$MODEL_DIR" \
        --data-location "$DATA_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --datasets "${DATASETS[@]}" \
        --blockwise-coef \
        --base-threshold 0.05 \
        --beta 1.0 \
        --verbose
}

# Function to run zero-shot evaluation
run_zero_shot_eval() {
    MODEL=$1
    PROMPT_ENSEMBLE=$2

    ENSEMBLE_FLAG=""
    ENSEMBLE_STR=""
    if [ "$PROMPT_ENSEMBLE" = true ]; then
        ENSEMBLE_FLAG="--prompt-ensemble"
        ENSEMBLE_STR=" with prompt ensemble"
    fi

    echo "===================================================="
    echo "Running zero-shot evaluation for model: $MODEL$ENSEMBLE_STR"
    echo "===================================================="

    python src/eval_zero_shot.py \
        --model "$MODEL" \
        --data-location "$DATA_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --datasets "${DATASETS[@]}" \
        --save-dir "$SAVE_DIR/zero_shot" \
        $ENSEMBLE_FLAG
}

# Run evaluations for each model
for MODEL in "${MODELS[@]}"; do
    # Create model-specific log directory
    mkdir -p "$SAVE_DIR/$MODEL/logs"
    LOG_DIR="$SAVE_DIR/$MODEL/logs"

    # Generate timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # Run standard evaluation
    run_standard_eval "$MODEL" > "$LOG_DIR/standard_eval_${TIMESTAMP}.log" 2>&1

    # Run causal evaluation
    run_causal_eval "$MODEL" > "$LOG_DIR/causal_eval_${TIMESTAMP}.log" 2>&1

    # Run blockwise evaluation
    run_blockwise_eval "$MODEL" > "$LOG_DIR/blockwise_eval_${TIMESTAMP}.log" 2>&1

    # Run adaptive gating evaluation
    run_adaptive_gating_eval "$MODEL" > "$LOG_DIR/adaptive_gating_eval_${TIMESTAMP}.log" 2>&1

    # Run zero-shot evaluations
    run_zero_shot_eval "$MODEL" false > "$LOG_DIR/zero_shot_eval_${TIMESTAMP}.log" 2>&1
    run_zero_shot_eval "$MODEL" true > "$LOG_DIR/zero_shot_eval_ensemble_${TIMESTAMP}.log" 2>&1

    echo "Completed all evaluations for model: $MODEL"
    echo "Logs saved to: $LOG_DIR"
done

echo "All evaluations completed successfully!"