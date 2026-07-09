#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 PROJECT_ROOT MODEL TASK VARIANT CONDA_ENV CONFIG_REL_PATH [MAIN_ARGS...]"
    exit 1
fi

PROJECT_ROOT=$1
MODEL=$2
TASK=$3
VARIANT=$4
CONDA_ENV_NAME=$5
CONFIG_REL_PATH=$6
shift 6

CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL_PATH}"
OUTPUT_DIR="${PROJECT_ROOT}/results/foundation_models/${MODEL}/${TASK}/${VARIANT}"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/foundation_models/${MODEL}/${TASK}/${VARIANT}"
TENSORBOARD_DIR="${PROJECT_ROOT}/event_logs/foundation_models/${MODEL}/${TASK}/${VARIANT}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

activate_conda_env() {
    set +u
    export PS1="${PS1-}"

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"
        set -u
        return
    fi

    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
        set -u
        return
    fi

    if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
        set -u
        return
    fi

    set -u
    echo "Could not find conda initialization script for env '${CONDA_ENV_NAME}'."
    exit 1
}

resolve_python_bin() {
    local env_py

    for env_root in "${HOME}/.conda/envs" "${HOME}/miniconda3/envs" "${HOME}/anaconda3/envs"; do
        env_py="${env_root}/${CONDA_ENV_NAME}/bin/python"
        if [[ -x "$env_py" ]]; then
            echo "$env_py"
            return
        fi
    done

    activate_conda_env
    command -v python
}

PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo "Variant: ${VARIANT}"
echo "Config: ${CONFIG_PATH}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "Node: ${HOSTNAME}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"
echo "=================================================="

"$PYTHON_BIN" main.py \
    --config "$CONFIG_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --checkpoint_dir="$CHECKPOINT_DIR" \
    --tensorboard_dir="$TENSORBOARD_DIR" \
    "$@"

echo "End time: $(date)"
