#!/bin/bash
# Run all finetuning tasks for example_foundation_model_fixed on subject 1 only
# (excluding content_noncontent_task, which should have been run already)
#
# This script runs all finetuning configs in the configs/ directory
# Each config file should be named finetuning_{task_name}.yaml

set -e
set -o pipefail

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

    # Fallback to common paths


    # Try to activate conda environment

# GPU Configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0

echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Configuration
CONFIGS_DIR="$SCRIPT_DIR/configs"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/results"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
TENSORBOARD_DIR="$SCRIPT_DIR/event_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/example_foundation_model_fixed_sub1_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
log() {
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
    python main.py --config "$config_path"" \
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
log "Example Foundation Model Fixed - Subject 1 Only (After Content/Non-content)"
log "Started at: $(date)"
log "Log file: $LOG_FILE"
log "Using GPU: $CUDA_VISIBLE_DEVICES"
log "Note: Skipping content_noncontent_task (should have been run already)"
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

# Run each task (skip content_noncontent_task)
for config_file in "${CONFIG_FILES[@]}"; do
    config_file=$(basename "$config_file")
    
    # Extract task name to check if we should skip it
    task_name=$(basename "$config_file" .yaml | sed 's/finetuning_//')
    
    # Skip content_noncontent_task
    if [ "$task_name" = "content_noncontent_task" ]; then
        log "Skipping $task_name (already completed)"
        log ""
        continue

    if run_task "$config_file"; then
        ((SUCCESSFUL_TASKS++))
    else
        ((FAILED_TASKS++))
        FAILED_TASK_LIST+=("$task_name")

    log ""
done

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
if [ $FAILED_TASKS -gt 0 ]; then
log "Finished at: $(date)"
log "Log file: $LOG_FILE"

# Exit with error code if any task failed
if [ $FAILED_TASKS -gt 0 ]; then
    exit 1
else
    exit 0
fi
