#!/usr/bin/env python3
"""
Generate training targets from training_matrix.yaml with optional filtering.

Usage:
    python scripts/generate_training_targets.py [--models MODEL1,MODEL2] [--tasks TASK1,TASK2]

Output format (one per line):
    model_name|task_name|config_file
"""

import argparse
import yaml
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate filtered training targets from training matrix"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to filter (e.g., example_foundation_model,neural_conv_decoder)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task names to filter (e.g., word_embedding_decoding_task,sentence_onset_task)",
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        default="training_matrix.yaml",
        help="Path to training matrix YAML file",
    )
    return parser.parse_args()


def load_training_matrix(matrix_file):
    """Load the training matrix from YAML file."""
    try:
        with open(matrix_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Training matrix file '{matrix_file}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)


def filter_targets(training_matrix, model_filter=None, task_filter=None):
    """
    Filter training targets based on models and tasks.

    Args:
        training_matrix: Dict mapping {model_name: {task_name: [config_files]}}
        model_filter: List of model names to include (None = all)
        task_filter: List of task names to include (None = all)

    Returns:
        List of tuples (model_name, task_name, config_file)
    """
    targets = []

    for model_name, tasks in training_matrix.items():
        # Skip if model not in filter
        if model_filter and model_name not in model_filter:
            continue

        for task_name, config_files in tasks.items():
            # Skip if task not in filter
            if task_filter and task_name not in task_filter:
                continue

            # Add all config files for this model/task combination
            for config_file in config_files:
                targets.append((model_name, task_name, config_file))

    return targets


def main():
    args = parse_args()

    # Parse filter arguments
    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(",")]

    task_filter = None
    if args.tasks:
        task_filter = [t.strip() for t in args.tasks.split(",")]

    # Load and filter training matrix
    training_matrix = load_training_matrix(args.matrix_file)
    targets = filter_targets(training_matrix, model_filter, task_filter)

    # Output targets (one per line, pipe-separated)
    for model_name, task_name, config_file in targets:
        print(f"{model_name}|{task_name}|{config_file}")

    # Print summary to stderr for visibility
    if targets:
        print(f"\nGenerated {len(targets)} training targets", file=sys.stderr)
        if model_filter:
            print(f"  Models: {', '.join(model_filter)}", file=sys.stderr)
        if task_filter:
            print(f"  Tasks: {', '.join(task_filter)}", file=sys.stderr)
    else:
        print("Warning: No targets matched the specified filters", file=sys.stderr)


if __name__ == "__main__":
    main()
