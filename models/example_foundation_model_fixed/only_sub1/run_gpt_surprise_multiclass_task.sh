#!/bin/bash
# Run gpt_surprise_multiclass_task for example_foundation_model_fixed on subject 1 only

set -e
set -o pipefail

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"



# GPU Configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
fi
# Configuration
CONFIG_PATH="$SCRIPT_DIR/configs/finetuning_gpt_surprise_multiclass_task.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/results"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
TENSORBOARD_DIR="$SCRIPT_DIR/event_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/example_foundation_model_fixed_gpt_surprise_multiclass_sub1_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

echo "========================================="
echo "Example Foundation Model Fixed - GPT Surprise Multiclass Task"
echo "Started at: $(date)"
echo "Config: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "TensorBoard dir: $TENSORBOARD_DIR"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

python main.py --config "$CONFIG_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --tensorboard_dir="$TENSORBOARD_DIR" \
  2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

if [ $exit_code -eq 0 ]; then
    echo "✓ Successfully completed"
else
    echo "✗ Failed with exit code: $exit_code"
    exit $exit_code
fi


