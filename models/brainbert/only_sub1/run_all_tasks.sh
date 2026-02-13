#!/bin/bash
# Run all finetuning tasks for BrainBERT on subject 1 only
#
# This script runs all finetuning configs in the configs/ directory
# Each config file should be named finetuning_{task_name}.yaml

set -e
set -o pipefail

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# GPU Configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Configuration
CONFIGS_DIR="$SCRIPT_DIR/configs"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/results"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
TENSORBOARD_DIR="$SCRIPT_DIR/event_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/brainbert_sub1_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run evaluation for a task
run_task() {
    local config_file=$1
    local config_path="$CONFIGS_DIR/$config_file"
    
    # Extract task name from filename (finetuning_{task_name}.yaml)
    local task_name=$(basename "$config_file" .yaml | sed 's/finetuning_//')
    
    log "========================================="
    log "Starting task: $task_name"
    log "Config: $config_path"
    log "========================================="
    
    local start_time=$(date +%s)
    
    # Run the evaluation
    # Use PIPESTATUS to get the actual exit code of python, not tee
    python main.py --config "$config_path" \
  --output_dir="$OUTPUT_DIR" \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --tensorboard_dir="$TENSORBOARD_DIR" \
  2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log "✓ Successfully completed: $task_name (Duration: ${duration}s)"
    else
        log "✗ Failed: $task_name (Duration: ${duration}s)"
    fi
    
    return $exit_code
}

# Main execution
log "========================================="
log "BrainBERT - Subject 1 Only"
log "Started at: $(date)"
log "Log file: $LOG_FILE"
log "Using GPU: $CUDA_VISIBLE_DEVICES"
log "========================================="

# Find all finetuning config files
CONFIG_FILES=("$CONFIGS_DIR"/finetuning_*.yaml)

if [ ${#CONFIG_FILES[@]} -eq 0 ] || [ ! -f "${CONFIG_FILES[0]}" ]; then
    log "No finetuning config files found in $CONFIGS_DIR"
    log "Expected files: finetuning_{task_name}.yaml"
    exit 1
fi

# Track results
TOTAL_TASKS=${#CONFIG_FILES[@]}
SUCCESSFUL_TASKS=0
FAILED_TASKS=0
FAILED_TASK_LIST=()

# Run each task
# Temporarily disable set -e so we can continue even if a task fails
set +e
for config_file in "${CONFIG_FILES[@]}"; do
    config_file=$(basename "$config_file")
    
    run_task "$config_file"
    task_exit_code=$?
    
    if [ $task_exit_code -eq 0 ]; then
        SUCCESSFUL_TASKS=$((SUCCESSFUL_TASKS + 1))
    else
        FAILED_TASKS=$((FAILED_TASKS + 1))
        task_name=$(basename "$config_file" .yaml | sed 's/finetuning_//')
        FAILED_TASK_LIST+=("$task_name")
    fi
    
    log ""
done
set -e

# Summary
log "========================================="
log "Evaluation Summary"
log "========================================="
log "Total tasks: $TOTAL_TASKS"
log "Successful: $SUCCESSFUL_TASKS"
log "Failed: $FAILED_TASKS"
log ""

if [ $FAILED_TASKS -gt 0 ]; then
    log "Failed tasks:"
    for task in "${FAILED_TASK_LIST[@]}"; do
        log "  - $task"
    done
    log ""
fi

log "Finished at: $(date)"
log "Log file: $LOG_FILE"

# Exit with error code if any task failed
if [ $FAILED_TASKS -gt 0 ]; then
    exit 1
else
    exit 0
fi
