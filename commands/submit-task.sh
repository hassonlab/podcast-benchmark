#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./commands/submit-task.sh --model MODEL --task TASK [options] [-- EXTRA_MAIN_ARGS...]

Default standard policy:
  - fold_ids = [1,5]
  - lags = [-1000, -500, 0, 500, 1000]
    implemented as min_lag=-1000, max_lag=1500, lag_step_size=500

Required:
  --model MODEL           brainbert | popt | diver
  --task TASK             task name under configs/foundation_models/<model>/<task>/

Optional:
  --variant VARIANT       config variant (default: supersubject)
  --partition NAME        Slurm partition (default: debug)
  --node NAME             Slurm node list, e.g. node1
  --gpus N                GPU count (default: 1)
  --cpus N                CPU count (default: model-specific)
  --mem-gb N              Memory in GB (default: cpus * 4)
  --env NAME              Conda env override
  --job-name NAME         Slurm job name override
  --fold N                shorthand for --training_params.fold_ids=[N]
  --fold-ids VALUE        raw override value, e.g. "[1]" or "[1,2]"
  --lag N                 shorthand for --training_params.lag=N
  --epochs N              shorthand for --training_params.epochs=N
  --override KEY=VALUE    extra config override, repeatable
  --dry-run               print sbatch command without submitting
  --help                  show this message

Examples:
  ./commands/submit-task.sh --model brainbert --task llm_decoding
  ./commands/submit-task.sh --model popt --task volume_level --fold 1 --lag 0 --node node1
  ./commands/submit-task.sh --model diver --task content_noncontent --override training_params.epochs=3
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL=""
TASK=""
VARIANT="supersubject"
PARTITION="${SBATCH_PARTITION:-debug}"
GPU_COUNT="${SBATCH_GPUS:-1}"
CPU_COUNT=""
MEM_GB=""
CONDA_ENV_NAME=""
NODELIST="${SBATCH_NODELIST:-}"
JOB_NAME=""
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
        --partition)
            PARTITION="${2:?missing value for --partition}"
            shift 2
            ;;
        --node|--nodelist)
            NODELIST="${2:?missing value for --node}"
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
        --mem-gb)
            MEM_GB="${2:?missing value for --mem-gb}"
            shift 2
            ;;
        --env)
            CONDA_ENV_NAME="${2:?missing value for --env}"
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

if [[ -z "$MODEL" || -z "$TASK" ]]; then
    usage
    exit 1
fi

CONFIG_REL_PATH="configs/foundation_models/${MODEL}/${TASK}/${VARIANT}.yml"
CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL_PATH}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: ${CONFIG_PATH}"
    exit 1
fi

if [[ "$HAS_FOLD_OVERRIDE" != "1" ]]; then
    MAIN_ARGS+=("--training_params.fold_ids=[1,5]")
fi

if [[ "$HAS_LAG_OVERRIDE" != "1" ]]; then
    MAIN_ARGS+=(
        "--training_params.min_lag=-1000"
        "--training_params.max_lag=1500"
        "--training_params.lag_step_size=500"
    )
fi

if [[ -z "$CONDA_ENV_NAME" ]]; then
    CONDA_ENV_NAME="${FOUNDATION_CONDA_ENV:-decoding_env}"
fi

if [[ -z "$CPU_COUNT" ]]; then
    if [[ "$MODEL" == "diver" ]]; then
        CPU_COUNT=12
    else
        CPU_COUNT=8
    fi
fi

if [[ -z "$MEM_GB" ]]; then
    MEM_GB=$(( CPU_COUNT * 4 ))
fi

MAX_CPUS=$(( GPU_COUNT * 12 ))
MAX_MEM_GB=$(( CPU_COUNT * 5 ))

if (( GPU_COUNT < 1 || GPU_COUNT > 4 )); then
    echo "This submit wrapper expects 1-4 GPUs. Got: ${GPU_COUNT}"
    exit 1
fi

if (( CPU_COUNT < 1 || CPU_COUNT > MAX_CPUS )); then
    echo "Requested CPUs (${CPU_COUNT}) exceed cluster rule: max 12 CPUs per GPU (${MAX_CPUS})."
    exit 1
fi

if (( MEM_GB < 1 || MEM_GB > MAX_MEM_GB )); then
    echo "Requested memory (${MEM_GB}G) exceeds cluster rule: max 5 GB per CPU (${MAX_MEM_GB}G)."
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs/foundation_models"

if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="pb-${MODEL}-${TASK}-${VARIANT}"
fi

SBATCH_ARGS=(
    -p "$PARTITION"
    --nodes=1
    --ntasks=1
    --gres="gpu:${GPU_COUNT}"
    --cpus-per-task="${CPU_COUNT}"
    --mem="${MEM_GB}G"
    --job-name="${JOB_NAME}"
    --output="${PROJECT_ROOT}/logs/foundation_models/%x-%j.out"
    --error="${PROJECT_ROOT}/logs/foundation_models/%x-%j.err"
)

if [[ -n "$NODELIST" ]]; then
    SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

echo "Submitting ${MODEL}/${TASK}/${VARIANT}"
echo "  partition=${PARTITION} gpus=${GPU_COUNT} cpus=${CPU_COUNT} mem=${MEM_GB}G env=${CONDA_ENV_NAME}"
echo "  config=${CONFIG_REL_PATH}"
echo "  overrides=${MAIN_ARGS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1, not submitting job."
    echo "sbatch ${SBATCH_ARGS[*]} ${SCRIPT_DIR}/internal/slurm_worker.sh ${PROJECT_ROOT} ${MODEL} ${TASK} ${VARIANT} ${CONDA_ENV_NAME} ${CONFIG_REL_PATH} ${MAIN_ARGS[*]}"
    exit 0
fi

sbatch \
    "${SBATCH_ARGS[@]}" \
    "${SCRIPT_DIR}/internal/slurm_worker.sh" \
    "$PROJECT_ROOT" \
    "$MODEL" \
    "$TASK" \
    "$VARIANT" \
    "$CONDA_ENV_NAME" \
    "$CONFIG_REL_PATH" \
    "${MAIN_ARGS[@]}"
