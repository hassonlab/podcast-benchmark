#!/usr/bin/env python3
"""Submit one-shot foundation controls as ordinary multi-lag jobs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


FOUNDATION_MODELS = ("brainbert", "diver", "popt")
CONTROL_TYPES = {
    "random-init": Path("configs/controls/foundation_random_init"),
    "shuffled-targets": Path("configs/controls/foundation_shuffled_targets"),
}
EXCLUDED_TASKS = {"llm_decoding"}
RUN_VARIANTS = ("supersubject", *(f"subject{i}_full" for i in range(1, 10)))
LAG_RANGES = ((-1000, 0), (0, 1000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-types",
        nargs="+",
        choices=tuple(CONTROL_TYPES),
        default=list(CONTROL_TYPES),
    )
    parser.add_argument("--sbatch-flags", default="")
    parser.add_argument("--config-overrides", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def eligible_control_configs(control_root: Path) -> list[Path]:
    configs = []
    for model in FOUNDATION_MODELS:
        model_prefix = f"{model}_"
        for config in sorted(control_root.glob(f"{model}_*.yml")):
            remainder = config.stem.removeprefix(model_prefix)
            variant = next(
                (
                    candidate
                    for candidate in RUN_VARIANTS
                    if remainder.endswith(f"_{candidate}")
                ),
                None,
            )
            if variant is None:
                continue
            task = remainder[: -(len(variant) + 1)]
            if task in EXCLUDED_TASKS:
                continue
            configs.append(config)
    return configs


def validate_control_config(config: Path, control_type: str) -> None:
    with config.open() as f:
        raw = yaml.safe_load(f) or {}

    task_config = raw.get("task_config", {})
    if "_word_embedding_" in config.name:
        embedding_type = task_config.get("task_specific_config", {}).get(
            "embedding_type"
        )
        if embedding_type != "gpt-2xl":
            raise ValueError(
                f"Expected GPT-2 XL word embeddings, got {embedding_type!r}: {config}"
            )

    if control_type == "shuffled-targets":
        if raw.get("training_params", {}).get("shuffle_targets") is not True:
            raise ValueError(f"Shuffled-target control is not enabled: {config}")
        return

    preprocessors = task_config.get("data_params", {}).get(
        "preprocessor_params", []
    )
    foundation_specs = [
        params.get("foundation_model_spec")
        for params in preprocessors
        if isinstance(params, dict) and "foundation_model_spec" in params
    ]
    if not foundation_specs or not any(spec.get("random_init") is True for spec in foundation_specs):
        raise ValueError(f"Random initialization is not enabled: {config}")


def command_for_range(
    config: Path,
    control_type: str,
    min_lag: int,
    max_lag: int,
    sbatch_flags: str = "",
    config_overrides: str = "",
) -> list[str]:
    range_tag = f"{min_lag}-to-{max_lag}"
    config_tag = config.stem.replace("_", "-")
    trial_control = control_type.replace("-", "_")
    return [
        "sbatch",
        f"--job-name=foundation-{control_type}-{config_tag}-{range_tag}",
        *shlex.split(sbatch_flags),
        "submit.sh",
        "main.py",
        "--config",
        str(config),
        "--training_params.num_null_repetitions=1",
        "--training_params.lag=null",
        f"--training_params.min_lag={min_lag}",
        f"--training_params.max_lag={max_lag}",
        "--training_params.lag_step_size=500",
        f"--trial_name={config.stem}_{trial_control}_{range_tag}",
        *shlex.split(config_overrides),
    ]


def main() -> int:
    args = parse_args()
    submitted = 0
    selected_configs = 0

    for control_type in args.control_types:
        control_root = CONTROL_TYPES[control_type]
        configs = eligible_control_configs(control_root)
        for config in configs:
            validate_control_config(config, control_type)
            selected_configs += 1
            for min_lag, max_lag in LAG_RANGES:
                command = command_for_range(
                    config,
                    control_type,
                    min_lag,
                    max_lag,
                    args.sbatch_flags,
                    args.config_overrides,
                )
                if args.dry_run:
                    print(shlex.join(command))
                else:
                    subprocess.run(command, check=True)
                submitted += 1

    action = "would submit" if args.dry_run else "submitted"
    print(f"{action} {submitted} jobs from {selected_configs} control configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
