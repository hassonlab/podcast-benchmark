#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./commands/run-local.sh --model MODEL --task TASK [options] [-- EXTRA_MAIN_ARGS...]

ALWAYS put --model

Run a task locally (no SLURM). For quick debugging on login nodes or
interactive sessions (salloc).

Default debug policy (lighter than submit-task):
  - fold_ids = [1]
  - lag = 0

Required:
  --model MODEL           brainbert | popt | diver
  --task TASK             task name under configs/foundation_models/<model>/<task>/

Optional:
  --variant VARIANT       config variant (default: supersubject)
  --config PATH           direct config path (overrides --model/--task/--variant)
  --gpus N                GPU count; 0 = CPU-only (default: auto-detect)
  --cpus N                CPU count (default: model-specific)
  --env PATH_OR_NAME      Python env: venv path or conda env name
                          (default: PROJECT_ROOT/decoding_env)
  --job-name NAME         label for log messages
  --fold N                shorthand for --training_params.fold_ids=[N]
  --fold-ids VALUE        raw override value, e.g. "[1]" or "[1,2]"
  --lag N                 shorthand for --training_params.lag=N
  --epochs N              shorthand for --training_params.epochs=N
  --override KEY=VALUE    extra config override, repeatable
  --dry-run               print command without running
  --help                  show this message

Examples:
  # Quick CPU test on login node
  ./commands/run-local.sh --model brainbert --task content_noncontent --epochs 1

  # With GPU on interactive node (auto-detected)
  ./commands/run-local.sh --model brainbert --task content_noncontent --epochs 1

  # Force CPU-only
  ./commands/run-local.sh --model brainbert --task content_noncontent --gpus 0

  # Direct config path
  ./commands/run-local.sh --config configs/baselines/neural_conv_decoder/glove.yml
EOF
}

# cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark
#./commands/run-local.sh --model popt --task gpt_surprise --epochs 3
# ./commands/run-local.sh --model diver --task gpt_surprise --epochs 3  --override "model_spec.feature_cache=True" 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL=""
TASK=""
VARIANT="supersubject"
RAW_CONFIG=""
GPU_COUNT=""
CPU_COUNT=""
PYTHON_ENV=""
JOB_NAME=""
OUTPUT_SUFFIX=""
DRY_RUN="${DRY_RUN:-0}"
HAS_FOLD_OVERRIDE=0
HAS_LAG_OVERRIDE=0

declare -a MAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="${2:?missing value for --model}"
            shift 2
            ;;
        --task)
            TASK="${2:?missing value for --task}"
            shift 2
            ;;
        --variant)
            VARIANT="${2:?missing value for --variant}"
            shift 2
            ;;
        --config)
            RAW_CONFIG="${2:?missing value for --config}"
            shift 2
            ;;
        --gpus)
            GPU_COUNT="${2:?missing value for --gpus}"
            shift 2
            ;;
        --cpus)
            CPU_COUNT="${2:?missing value for --cpus}"
            shift 2
            ;;
        --env)
            PYTHON_ENV="${2:?missing value for --env}"
            shift 2
            ;;
        --job-name)
            JOB_NAME="${2:?missing value for --job-name}"
            shift 2
            ;;
        --fold)
            MAIN_ARGS+=("--training_params.fold_ids=[$2]")
            HAS_FOLD_OVERRIDE=1
            shift 2
            ;;
        --fold-ids)
            MAIN_ARGS+=("--training_params.fold_ids=${2:?missing value for --fold-ids}")
            HAS_FOLD_OVERRIDE=1
            shift 2
            ;;
        --lag)
            MAIN_ARGS+=("--training_params.lag=${2:?missing value for --lag}")
            HAS_LAG_OVERRIDE=1
            shift 2
            ;;
        --epochs)
            MAIN_ARGS+=("--training_params.epochs=${2:?missing value for --epochs}")
            shift 2
            ;;
        --override)
            if [[ "${2:-}" != *=* ]]; then
                echo "--override expects KEY=VALUE, got: ${2:-<missing>}"
                exit 1
            fi
            case "${2}" in
                training_params.fold_ids=*)
                    HAS_FOLD_OVERRIDE=1
                    ;;
                training_params.lag=*|training_params.min_lag=*|training_params.max_lag=*|training_params.lag_step_size=*)
                    HAS_LAG_OVERRIDE=1
                    ;;
            esac
            MAIN_ARGS+=("--${2}")
            shift 2
            ;;
        --output-suffix)
            OUTPUT_SUFFIX="${2:?missing value for --output-suffix}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                MAIN_ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Resolve config path
if [[ -n "$RAW_CONFIG" ]]; then
    CONFIG_REL_PATH="${RAW_CONFIG}"
elif [[ -n "$MODEL" && -n "$TASK" ]]; then
    CONFIG_REL_PATH="configs/foundation_models/${MODEL}/${TASK}/${VARIANT}.yml"
else
    usage
    exit 1
fi

CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL_PATH}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: ${CONFIG_PATH}"
    exit 1
fi

# Debug defaults (lighter than submit-task)
if [[ "$HAS_FOLD_OVERRIDE" != "1" ]]; then
    MAIN_ARGS+=("--training_params.fold_ids=[1]")
fi

if [[ "$HAS_LAG_OVERRIDE" != "1" ]]; then
    MAIN_ARGS+=(--training_params.lag=0)
fi

# Default env: venv in project root
if [[ -z "$PYTHON_ENV" ]]; then
    PYTHON_ENV="${PROJECT_ROOT}/decoding_env"
fi

# Default CPUs (same logic as submit-task.sh)
if [[ -z "$CPU_COUNT" ]]; then
    if [[ "$MODEL" == "diver" ]]; then
        CPU_COUNT=12
    else
        CPU_COUNT=8
    fi
fi

# Auto-detect GPU: if not specified, check nvidia-smi
if [[ -z "$GPU_COUNT" ]]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_COUNT=1
    else
        GPU_COUNT=0
    fi
fi

if [[ "$GPU_COUNT" == "0" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

# Output dirs (same layout as slurm_worker)
if [[ -n "$MODEL" && -n "$TASK" ]]; then
    SUFFIX_PATH="${VARIANT}"
    if [[ -n "$OUTPUT_SUFFIX" ]]; then
        SUFFIX_PATH="${VARIANT}/${OUTPUT_SUFFIX}"
    fi
    OUTPUT_DIR="${PROJECT_ROOT}/results/foundation_models/${MODEL}/${TASK}/${SUFFIX_PATH}"
    CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/foundation_models/${MODEL}/${TASK}/${SUFFIX_PATH}"
    TENSORBOARD_DIR="${PROJECT_ROOT}/event_logs/foundation_models/${MODEL}/${TASK}/${SUFFIX_PATH}"
    mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"
    MAIN_ARGS+=(
        "--output_dir=${OUTPUT_DIR}"
        "--checkpoint_dir=${CHECKPOINT_DIR}"
        "--tensorboard_dir=${TENSORBOARD_DIR}"
    )
fi

# Activate environment (same pattern as slurm_worker_nersc.sh)
activate_env() {
    if [[ -f "${PYTHON_ENV}/bin/activate" ]]; then
        source "${PYTHON_ENV}/bin/activate"
    elif [[ -f "${PYTHON_ENV}/bin/python" ]]; then
        export PATH="${PYTHON_ENV}/bin:${PATH}"
    else
        module load conda 2>/dev/null || true
        set +u
        export PS1="${PS1-}"
        eval "$(conda shell.bash hook)"
        conda activate "$PYTHON_ENV"
        set -u
    fi
}

activate_env

if [[ -z "$JOB_NAME" && -n "$MODEL" && -n "$TASK" ]]; then
    JOB_NAME="pb-${MODEL}-${TASK}-${VARIANT}"
fi

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Running locally (gpus=${GPU_COUNT} cpus=${CPU_COUNT})"
echo "  job-name: ${JOB_NAME:-local}"
echo "  config: ${CONFIG_REL_PATH}"
echo "  env: ${PYTHON_ENV}"
echo "  python: $(which python)"
echo "  node: ${HOSTNAME}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  overrides: ${MAIN_ARGS[*]}"
echo "  start: $(date)"
echo "=================================================="

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1, not running."
    echo "python main.py --config ${CONFIG_PATH} ${MAIN_ARGS[*]}"
    exit 0
fi

python main.py \
    --config "$CONFIG_PATH" \
    "${MAIN_ARGS[@]}"

echo "End time: $(date)"
