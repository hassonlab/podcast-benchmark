#!/usr/bin/env python3
"""Submit single-lag repeated-null jobs for foundation benchmark conditions."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

import yaml


FOUNDATION_MODELS = ("brainbert", "diver", "popt")
DEFAULT_LAGS = (-500, 0, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-config",
        type=Path,
        default=Path("benchmark-results/paper_results.yml"),
    )
    parser.add_argument(
        "--control-name",
        default="random-init",
        help="Label used in Slurm job and trial names.",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAGS),
        help="Single lags to submit as separate repeated-null jobs.",
    )
    parser.add_argument(
        "--run-scope",
        choices=("all", "super_subject", "per_subject"),
        default="all",
        help="Restrict submission to super-subject or individual-subject runs.",
    )
    parser.add_argument(
        "--controls-root",
        type=Path,
        default=Path("configs/controls/foundation_random_init"),
    )
    parser.add_argument("--sbatch-flags", default="")
    parser.add_argument("--config-overrides", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def result_csvs(run_dir: Path, scope: str) -> Iterable[tuple[str, Path]]:
    if scope == "super_subject":
        yield "supersubject", run_dir / "lag_performance.csv"
        return
    if scope == "per_subject":
        for path in sorted(run_dir.glob("subject_*/lag_performance.csv")):
            subject = path.parent.name.removeprefix("subject_")
            yield f"subject{subject}_full", path


def control_path(controls_root: Path, model: str, task: str, variant: str) -> Path:
    return controls_root / f"{model}_{task}_{variant}.yml"


def job_name(config: Path, control_name: str = "random-init") -> str:
    name = config.stem.replace("_", "-")
    return f"decoder-training-{control_name}-{name}"


def command_for_condition(
    config: Path,
    lag: int,
    sbatch_flags: str,
    config_overrides: str,
    control_name: str = "random-init",
) -> list[str]:
    trial_control_name = control_name.replace("-", "_")
    command = [
        "sbatch",
        f"--job-name={job_name(config, control_name)}",
        "--dependency=singleton",
        *shlex.split(sbatch_flags),
        "submit.sh",
        "main.py",
        "--config",
        str(config),
        f"--training_params.lag={lag}",
        f"--trial_name={config.stem}_{trial_control_name}_lag_{lag}",
        *shlex.split(config_overrides),
    ]
    return command


def main() -> int:
    args = parse_args()
    config = load_yaml(args.results_config)
    results = config.get("results", {})
    submitted = 0
    controls = set()

    for model in FOUNDATION_MODELS:
        for task, scopes in sorted(results.get(model, {}).items()):
            for scope, raw_paths in sorted(scopes.items()):
                if args.run_scope != "all" and scope != args.run_scope:
                    continue
                paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
                for raw_path in paths:
                    run_dir = Path(raw_path)
                    for variant, csv_path in result_csvs(run_dir, scope):
                        if not csv_path.exists():
                            raise FileNotFoundError(
                                f"Benchmark lag file not found: {csv_path}"
                            )
                        control = control_path(
                            args.controls_root, model, task, variant
                        )
                        if not control.exists():
                            raise FileNotFoundError(
                                f"Control config not found: {control}"
                            )

                        for lag in args.lags:
                            command = command_for_condition(
                                control,
                                lag,
                                args.sbatch_flags,
                                args.config_overrides,
                                args.control_name,
                            )
                            print(f"{control}: lag {lag} ms")
                            if args.dry_run:
                                print(shlex.join(command))
                            else:
                                subprocess.run(command, check=True)
                            submitted += 1
                        controls.add(control)

    action = "would submit" if args.dry_run else "submitted"
    print(f"{action} {submitted} jobs for {len(controls)} control configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
