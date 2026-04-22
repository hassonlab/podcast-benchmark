#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Run baseline jobs in parallel on an interactive GPU node.
# (sig10 version — per-subject, top 10% significant channels)
#
# Loops over baseline configs and subjects, overriding
# electrode_file_path to processed_data/sig10/sub{N}_sig.csv.
#
# Usage:
#   bash job_script_in_interactive_baseline.sh [--max-parallel N] [--dry-run]
#
# Options:
#   --max-parallel N   Max concurrent jobs (default: 1)
#   --dry-run          Print jobs without running
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

MAX_PARALLEL=6
DRY_RUN=0

# cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark
# CUDA_VISIBLE_DEVICES=0 bash job_script_in_interactive_baseline.sh --dry-run

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-parallel) MAX_PARALLEL="${2:?missing value}"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --help|-h)
            sed -n '/^# =====/,/^# =====/p' "$0" | grep '^#' | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- Baseline configs to run ----
# Each entry is a path relative to PROJECT_ROOT
#podcast-benchmark/configs/baselines/neural_conv_decoder
# arbitrary.yml           glove.yml  #*! gpt_surprise_multiclass.yml  iu_boundaries.yml        llm.yml          pos.yml             whisper_embedding.yml
# content_noncontent.yml  gpt2.yml   gpt_surprise.yml             llm_two_stage_multi.yml  placeholder.yml  sentence_onset.yml
baseline_configs=(
    "configs/baselines/neural_conv_decoder/content_noncontent.yml"
    "configs/baselines/neural_conv_decoder/sentence_onset.yml"
    "configs/baselines/neural_conv_decoder/iu_boundaries.yml"
    "configs/baselines/neural_conv_decoder/gpt_surprise.yml"
    "configs/baselines/neural_conv_decoder/gpt_surprise_multiclass.yml"
    "configs/baselines/neural_conv_decoder/pos.yml"
    "configs/baselines/neural_conv_decoder/glove.yml"
    "configs/baselines/neural_conv_decoder/whisper_embedding.yml"
    # "configs/baselines/neural_conv_decoder/llm.yml"
)

lags=(0)
subjects=(1 2 3 4 5 6 7 8 9)

# ---- Logging ----
LOG_DIR="${PROJECT_ROOT}/logs/interactive_batch_baseline"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/summary_${TIMESTAMP}.log"

failed_jobs=()
succeeded_jobs=()
skipped_jobs=()
total_jobs=0
pids=()
job_names=()

wait_for_slot() {
    while (( ${#pids[@]} >= MAX_PARALLEL )); do
        reap_finished
        if (( ${#pids[@]} >= MAX_PARALLEL )); then
            sleep 2
        fi
    done
}

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
echo "Interactive batch runner (baselines, sig10)"
echo "  max_parallel: ${MAX_PARALLEL}"
echo "  log_dir: ${LOG_DIR}"
echo "  start: $(date)"
echo "=========================================="

for config_path in "${baseline_configs[@]}"; do
    if [[ ! -f "${PROJECT_ROOT}/${config_path}" ]]; then
        skipped_jobs+=("${config_path} (not found)")
        continue
    fi

    config_name="$(basename "${config_path}" .yml)"

    for lag in "${lags[@]}"; do
        for subj in "${subjects[@]}"; do

            sig_file="processed_data/sig10/sub${subj}_sig.csv"
            if [[ ! -f "${PROJECT_ROOT}/${sig_file}" ]]; then
                echo "[WARN]  sig10 file not found: ${sig_file}, skipping"
                skipped_jobs+=("${config_name}/sub${subj}/lag${lag}")
                continue
            fi

            job_label="${config_name}_sig10/sub${subj}/lag${lag}"
            job_log="${LOG_DIR}/${config_name}_sub${subj}_sig10_lag${lag}_${TIMESTAMP}.log"
            total_jobs=$((total_jobs + 1))

            if [[ "$DRY_RUN" == "1" ]]; then
                echo "[DRY]   $job_label"
                continue
            fi

            wait_for_slot

            echo "[START] $job_label (slot ${#pids[@]}+1/${MAX_PARALLEL})"

            "${PROJECT_ROOT}/commands/run-local.sh" \
                --config "$config_path" \
                --fold-ids "[1,5]" \
                --lag "$lag" \
                --override "task_config.data_params.electrode_file_path=${sig_file}" \
                --override "output_dir=results/baseline/sub${subj}" \
                --override "checkpoint_dir=checkpoints/baseline/sub${subj}" \
                --override "tensorboard_dir=event_logs/baseline/sub${subj}" \
                > "$job_log" 2>&1 &

            pids+=($!)
            job_names+=("$job_label")

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
