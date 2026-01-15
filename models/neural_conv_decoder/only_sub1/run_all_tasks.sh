#!/bin/bash
# Run all tasks for neural_conv_decoder on subject 1 only
#
# This script runs all configs in the configs/ directory
# Each config file should be named neural_conv_decoder_{task_name}.yaml

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
LOG_FILE="$LOG_DIR/neural_conv_decoder_sub1_${TIMESTAMP}.log"

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
    
    # Extract task name from filename (neural_conv_decoder_{task_name}.yaml)
    local task_name=$(basename "$config_file" .yaml | sed 's/neural_conv_decoder_//')
    
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
log "Neural Conv Decoder - Subject 1 Only"
log "Started at: $(date)"
log "Log file: $LOG_FILE"
log "Using GPU: $CUDA_VISIBLE_DEVICES"
log "========================================="

# Find all config files
CONFIG_FILES=("$CONFIGS_DIR"/neural_conv_decoder_*.yaml)

if [ ${#CONFIG_FILES[@]} -eq 0 ] || [ ! -f "${CONFIG_FILES[0]}" ]; then
    log "No config files found in $CONFIGS_DIR"
    log "Expected files: neural_conv_decoder_{task_name}.yaml"
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
    
    if run_task "$config_file"; then
        ((SUCCESSFUL_TASKS++))
    else
        ((FAILED_TASKS++))
        task_name=$(basename "$config_file" .yaml | sed 's/neural_conv_decoder_//')
        FAILED_TASK_LIST+=("$task_name")

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
if [ $FAILED_TASKS -gt 0 ]; then
log "Finished at: $(date)"
log "Log file: $LOG_FILE"

# Exit with error code if any task failed
if [ $FAILED_TASKS -gt 0 ]; then
    exit 1
else
    exit 0
fi
