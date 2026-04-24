#!/usr/bin/env python

import argparse
from pathlib import Path

import yaml


def _parse_csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit model/task/config targets from training_matrix.yaml."
    )
    parser.add_argument(
        "--matrix",
        default="training_matrix.yaml",
        help="Path to the training matrix YAML file.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model filters, e.g. baselines/neural_conv_decoder.",
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated task filters, e.g. sentence_onset_task.",
    )
    args = parser.parse_args()

    model_filter = _parse_csv_arg(args.models)
    task_filter = _parse_csv_arg(args.tasks)

    matrix_path = Path(args.matrix)
    with matrix_path.open("r") as f:
        matrix = yaml.safe_load(f) or {}

    for model_name, tasks in matrix.items():
        if model_filter and model_name not in model_filter:
            continue
        for task_name, configs in tasks.items():
            if task_filter and task_name not in task_filter:
                continue
            for config_name in configs:
                print(f"{model_name}|{task_name}|{config_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
