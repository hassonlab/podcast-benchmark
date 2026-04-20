#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Run many jobs in parallel on an interactive GPU node.
# (sig10 version — per-subject, top 10% significant channels)
#
# Loops over individual subjects, using electrode files from
# processed_data/sig10/sub{NN}_sig.csv to select sig channels.
#
# For the supersubject (all subjects, all channels) version, see:
#   job_script_in_interactive_supersubject.sh
#
# Usage:
#   ./commands/job_script_in_interactive.sh [--max-parallel N] [--dry-run]
#
# Options:
#   --max-parallel N   Max concurrent jobs (default: 1)
#   --dry-run          Print jobs without running
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAX_PARALLEL=2
DRY_RUN=0
CLI_MODELS=()

# cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark/commands
# CUDA_VISIBLE_DEVICES=0 bash job_script_in_interactive.sh --models diver
# CUDA_VISIBLE_DEVICES=2 bash job_script_in_interactive.sh --models popt
# CUDA_VISIBLE_DEVICES=2 bash job_script_in_interactive.sh --models brainbert

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
    # "whisper_embedding"
    # "gpt_surprise"
    # "pos"
    "content_noncontent"
    # "gpt_surprise_multiclass"
    # "iu_boundary"
    #"sentence_onset"
    #"llm_decoding"
    # "llm_embedding_pretraining"
    #"word_embedding"
    #"volume_level"
)

# tasks=(
# )



lags=(0) #(-250, 0, 250, 500)

#!removed
    # "llm_decoding"
    # "volume_level"

# ---- Per-subject sig10 mode ----
# Set to non-empty to loop over individual subjects with sig10 electrode files.
# Each subject uses processed_data/sig10/sub{NN}_sig.csv as electrode_file_path.
# Leave empty ("") to use the variant config as-is (e.g. supersubject with all channels).
USE_SIG10=0 # 0 or 1

subjects=(9)


variants=(
    "subject_full"
)

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
echo "Interactive batch runner"
echo "  max_parallel: ${MAX_PARALLEL}"
echo "  log_dir: ${LOG_DIR}"
echo "  start: $(date)"
echo "=========================================="

for lag in "${lags[@]}"; do
    for task in "${tasks[@]}"; do
        for model in "${models[@]}"; do

            # Always loop per-subject (subject_full variant needs subject{N}_full.yml)
            # USE_SIG10 only controls whether sig10 electrode overrides are applied
            iter_subjects=("${subjects[@]}")

            for subj in "${iter_subjects[@]}"; do
                for variant in "${variants[@]}"; do

                    # Config file is always subject{N}_full.yml
                    config="configs/foundation_models/${model}/${task}/subject${subj}_full.yml"
                    if [[ ! -f "${PROJECT_ROOT}/${config}" ]]; then
                        skipped_jobs+=("${model}/${task}/subject${subj}_full/lag${lag}")
                        continue
                    fi

                    # Output path: results/.../subject_full/subject{N}_full/
                    #           or results/.../subject_sig10/subject{N}_sig10/
                    sig10_overrides=()
                    if [[ "${USE_SIG10}" == "1" ]]; then
                        sig_file="processed_data/sig10/sub${subj}_sig.csv"
                        if [[ ! -f "${PROJECT_ROOT}/${sig_file}" ]]; then
                            echo "[WARN]  sig10 file not found: ${sig_file}, skipping subject ${subj}"
                            skipped_jobs+=("${model}/${task}/subject_sig10/subject${subj}_sig10/lag${lag}")
                            continue
                        fi
                        sig10_overrides=(
                            --override "task_config.data_params.electrode_file_path=${sig_file}"
                        )
                        group_variant="subject_sig10"
                        output_suffix="subject${subj}_sig10"
                    else
                        group_variant="subject_full"
                        output_suffix="subject${subj}_full"
                    fi

                    job_label="${model}/${task}/${group_variant}/${output_suffix}/lag${lag}"
                    job_log="${LOG_DIR}/${model}_${task}_${output_suffix}_lag${lag}_${TIMESTAMP}.log"

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
                        --config "$config" \
                        --variant "$group_variant" \
                        --output-suffix "$output_suffix" \
                        --fold-ids "[1,5]" \
                        --lag "$lag" \
                        --override "model_spec.feature_cache=True" \
                        "${sig10_overrides[@]+"${sig10_overrides[@]}"}" \
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