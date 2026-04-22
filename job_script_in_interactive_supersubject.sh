#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Run many jobs in parallel on an interactive GPU node.
# (supersubject version — all subjects, all channels)
#
# Unlike submit-task.sh (which sbatch's ONE job to the SLURM queue),
# this script is meant for an already-allocated interactive node
# (e.g. via salloc or interactive-nersc.sh). It launches multiple
# python processes sharing the same GPU(s).
#
# Usage:
#   ./commands/job_script_in_interactive_supersubject.sh [--max-parallel N] [--dry-run]
#
# Options:
#   --max-parallel N   Max concurrent jobs (default: 1)
#   --dry-run          Print jobs without running
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

MAX_PARALLEL=1
DRY_RUN=0
CLI_MODELS=()

# cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark/commands
# CUDA_VISIBLE_DEVICES=0 bash job_script_in_interactive_supersubject.sh --models diver
# CUDA_VISIBLE_DEVICES=1 bash job_script_in_interactive_supersubject.sh --models popt
# CUDA_VISIBLE_DEVICES=2 bash job_script_in_interactive_supersubject.sh --models brainbert

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-parallel) MAX_PARALLEL="${2:?missing value}"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --models)       IFS=',' read -ra CLI_MODELS <<< "${2:?missing value}"; shift 2 ;;
        --help|-h)
            sed -n '/^# =====/,/^# =====/p' "$0" | grep '^#' | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- Job matrix ----
models=(
    "brainbert"
    "popt"
    "diver"
)
# Override models from CLI if --models was given
if (( ${#CLI_MODELS[@]} > 0 )); then
    models=("${CLI_MODELS[@]}")
fi
    # "brainbert" 0s
    # "popt" 2
    # "diver" 1


tasks=(
    #"whisper_embedding"   # done (all 3 models)
    #"gpt_surprise"        # done (all 3 models)
    #"pos"
    "content_noncontent"
    "gpt_surprise_multiclass"
    #"iu_boundary"
    #"llm_embedding_pretraining"
    #"sentence_onset"
    #"volume_level"
    #
    # "llm_decoding"        # bug: tensor shape mismatch
    # "word_embedding"      # bug: preserve_ensemble kwarg
)

lags=(0)

variants=(
    "supersubject"
    #"persubject_concat"
)

# ---- Sig10 mode ----
# Set USE_SIG10=1 to loop per-subject with sig10 electrode files.
# When enabled, supersubject variant maps to subject{N}_full configs,
# and electrode_file_path is overridden to processed_data/sig10/sub{N}_sig.csv.
# When disabled (default), runs with the variant config as-is (SigFull).
USE_SIG10=0
subjects=(1 2 3 4 5 6 7 8 9)

# ---- Logging ----
LOG_DIR="${PROJECT_ROOT}/logs/interactive_batch"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/summary_${TIMESTAMP}.log"

failed_jobs=()
succeeded_jobs=()
skipped_jobs=()
total_jobs=0
pids=()       # currently running PIDs
job_names=()  # label for each PID

# Wait until running jobs < MAX_PARALLEL, reaping finished ones
wait_for_slot() {
    while (( ${#pids[@]} >= MAX_PARALLEL )); do
        reap_finished
        if (( ${#pids[@]} >= MAX_PARALLEL )); then
            sleep 2
        fi
    done
}

# Check all running PIDs; remove finished ones
reap_finished() {
    local new_pids=()
    local new_names=()
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        name=${job_names[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_names+=("$name")
        else
            wait "$pid" && {
                succeeded_jobs+=("$name")
                echo "[DONE]  $name (pid=$pid)"
            } || {
                failed_jobs+=("$name")
                echo "[FAIL]  $name (pid=$pid) — see log in ${LOG_DIR}/"
            }
        fi
    done
    pids=("${new_pids[@]+"${new_pids[@]}"}")
    job_names=("${new_names[@]+"${new_names[@]}"}")
}

# ---- Main loop ----
echo "=========================================="
echo "Interactive batch runner (supersubject)"
echo "  max_parallel: ${MAX_PARALLEL}"
echo "  log_dir: ${LOG_DIR}"
echo "  start: $(date)"
echo "=========================================="
for task in "${tasks[@]}"; do
    for lag in "${lags[@]}"; do
        for model in "${models[@]}"; do
            for variant in "${variants[@]}"; do

                # Build subject list: if USE_SIG10, loop per-subject; otherwise single iteration
                if [[ "${USE_SIG10}" == "1" ]]; then
                    iter_subjects=("${subjects[@]}")
                else
                    iter_subjects=("")
                fi

                for subj in "${iter_subjects[@]}"; do

                    # Resolve variant & overrides for sig10 mode
                    actual_variant="$variant"
                    sig10_overrides=()
                    suffix_args=()
                    subj_tag=""

                    if [[ -n "$subj" && "${USE_SIG10}" == "1" ]]; then
                        # For supersubject variant, map to subject{N}_full config
                        if [[ "$variant" == "supersubject" ]]; then
                            actual_variant="subject${subj}_full"
                        fi
                        sig_file="processed_data/sig10/sub${subj}_sig.csv"
                        if [[ ! -f "${PROJECT_ROOT}/${sig_file}" ]]; then
                            echo "[WARN]  sig10 file not found: ${sig_file}, skipping"
                            skipped_jobs+=("${model}/${task}/${variant}_sig10/sub${subj}/lag${lag}")
                            continue
                        fi
                        sig10_overrides=(
                            --override "task_config.data_params.electrode_file_path=${sig_file}"
                        )
                        suffix_args=(--output-suffix "sig10")
                        subj_tag="/sub${subj}"
                    fi

                    config="configs/foundation_models/${model}/${task}/${actual_variant}.yml"
                    if [[ ! -f "${PROJECT_ROOT}/${config}" ]]; then
                        skipped_jobs+=("${model}/${task}/${actual_variant}${subj_tag}/lag${lag}")
                        continue
                    fi

                    job_label="${model}/${task}/${actual_variant}${subj_tag}/lag${lag}"
                    job_log="${LOG_DIR}/${model}_${task}_${actual_variant}${subj:+_sub${subj}}_lag${lag}_${TIMESTAMP}.log"
                    total_jobs=$((total_jobs + 1))

                    if [[ "$DRY_RUN" == "1" ]]; then
                        echo "[DRY]   $job_label"
                        continue
                    fi

                    wait_for_slot

                    echo "[START] $job_label (slot ${#pids[@]}+1/${MAX_PARALLEL})"

                    "${PROJECT_ROOT}/commands/run-local.sh" \
                        --model "$model" \
                        --task "$task" \
                        --variant "$actual_variant" \
                        --fold-ids "[1,5]" \
                        --lag "$lag" \
                        --override "model_spec.feature_cache=True" \
                        "${sig10_overrides[@]+"${sig10_overrides[@]}"}" \
                        "${suffix_args[@]+"${suffix_args[@]}"}" \
                        > "$job_log" 2>&1 &

                    pids+=($!)
                    job_names+=("$job_label")

                done
            done
        done
    done
done

# Wait for remaining jobs
while (( ${#pids[@]} > 0 )); do
    reap_finished
    if (( ${#pids[@]} > 0 )); then
        sleep 2
    fi
done

# ---- Summary ----
{
    echo "=========================================="
    echo "Batch completed: $(date)"
    echo "  Total:     ${total_jobs}"
    echo "  Succeeded: ${#succeeded_jobs[@]}"
    echo "  Failed:    ${#failed_jobs[@]}"
    echo "  Skipped:   ${#skipped_jobs[@]} (config not found)"
    echo ""

    if (( ${#failed_jobs[@]} > 0 )); then
        echo "FAILED JOBS:"
        for j in "${failed_jobs[@]}"; do
            echo "  - $j"
        done
        echo ""
    fi

    if (( ${#skipped_jobs[@]} > 0 )); then
        echo "SKIPPED (no config):"
        for j in "${skipped_jobs[@]}"; do
            echo "  - $j"
        done
    fi
    echo "=========================================="
} | tee "$SUMMARY_LOG"

if (( ${#failed_jobs[@]} > 0 )); then
    echo ""
    echo "Check individual logs: ls ${LOG_DIR}/*_${TIMESTAMP}.log"
    exit 1
fi