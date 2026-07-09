#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 PROJECT_ROOT MODEL TASK VARIANT PYTHON_ENV CONFIG_REL_PATH [MAIN_ARGS...]"
    exit 1
fi

PROJECT_ROOT=$1
MODEL=$2
TASK=$3
VARIANT=$4
PYTHON_ENV=$5
CONFIG_REL_PATH=$6
shift 6

CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL_PATH}"
OUTPUT_DIR="${PROJECT_ROOT}/results/foundation_models/${MODEL}/${TASK}/${VARIANT}"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/foundation_models/${MODEL}/${TASK}/${VARIANT}"
TENSORBOARD_DIR="${PROJECT_ROOT}/event_logs/foundation_models/${MODEL}/${TASK}/${VARIANT}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# Activate environment: supports both venv paths and conda env names
activate_env() {
    if [[ -f "${PYTHON_ENV}/bin/activate" ]]; then
        # venv or conda prefix path
        source "${PYTHON_ENV}/bin/activate"
    elif [[ -f "${PYTHON_ENV}/bin/python" ]]; then
        export PATH="${PYTHON_ENV}/bin:${PATH}"
    else
        # Treat as conda env name
        module load conda 2>/dev/null || true
        set +u
        export PS1="${PS1-}"
        eval "$(conda shell.bash hook)"
        conda activate "$PYTHON_ENV"
        set -u
    fi
}

activate_env

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo "Variant: ${VARIANT}"
echo "Config: ${CONFIG_PATH}"
echo "Python env: ${PYTHON_ENV}"
echo "Python bin: $(which python)"
echo "Node: ${HOSTNAME}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"
echo "=================================================="

python main.py \
    --config "$CONFIG_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --checkpoint_dir="$CHECKPOINT_DIR" \
    --tensorboard_dir="$TENSORBOARD_DIR" \
    "$@"

echo "End time: $(date)"
