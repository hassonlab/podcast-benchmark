#!/usr/bin/env python3
"""Submit random-init controls for lags present in foundation benchmark results."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


FOUNDATION_MODELS = ("brainbert", "diver", "popt")
MAX_LAGS_PER_JOB = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-config",
        type=Path,
        default=Path("benchmark-results/paper_results.yml"),
    )
    parser.add_argument(
        "--controls-root",
        type=Path,
        default=Path("configs/controls/foundation_random_init"),
    )
    parser.add_argument("--lags-per-job", type=int, default=MAX_LAGS_PER_JOB)
    parser.add_argument("--sbatch-flags", default="")
    parser.add_argument("--config-overrides", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_lags(path: Path) -> tuple[int, ...]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lags" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a 'lags' column")
        return tuple(
            sorted(
                {
                    int(float(row["lags"]))
                    for row in reader
                    if row.get("lags", "").strip()
                }
            )
        )


def arithmetic_batches(
    lags: Sequence[int], max_batch_size: int
) -> tuple[tuple[int, ...], ...]:
    """Split sorted lags into exact arithmetic sequences of bounded size."""
    if not 1 <= max_batch_size <= MAX_LAGS_PER_JOB:
        raise ValueError(
            f"lags_per_job must be between 1 and {MAX_LAGS_PER_JOB}, "
            f"got {max_batch_size}"
        )

    remaining = sorted(set(lags))
    batches = []
    while remaining:
        batch = [remaining.pop(0)]
        step = None
        while remaining and len(batch) < max_batch_size:
            candidate_step = remaining[0] - batch[-1]
            if step is None:
                step = candidate_step
            elif candidate_step != step:
                break
            batch.append(remaining.pop(0))
        batches.append(tuple(batch))
    return tuple(batches)


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


def job_name(config: Path) -> str:
    name = config.stem.replace("_", "-")
    return f"decoder-training-random-init-{name}"


def command_for_batch(
    config: Path,
    lags: Sequence[int],
    default_step: int,
    sbatch_flags: str,
    config_overrides: str,
) -> list[str]:
    step = lags[1] - lags[0] if len(lags) > 1 else default_step
    command = [
        "sbatch",
        f"--job-name={job_name(config)}",
        "--dependency=singleton",
        *shlex.split(sbatch_flags),
        "submit.sh",
        "main.py",
        "--config",
        str(config),
        f"--training_params.min_lag={lags[0]}",
        f"--training_params.max_lag={lags[-1] + step}",
        f"--training_params.lag_step_size={step}",
        *shlex.split(config_overrides),
    ]
    return command


def main() -> int:
    args = parse_args()
    if not 1 <= args.lags_per_job <= MAX_LAGS_PER_JOB:
        raise SystemExit(
            f"--lags-per-job must be between 1 and {MAX_LAGS_PER_JOB}; "
            f"got {args.lags_per_job}"
        )

    config = load_yaml(args.results_config)
    results = config.get("results", {})
    submitted = 0
    controls = set()

    for model in FOUNDATION_MODELS:
        for task, scopes in sorted(results.get(model, {}).items()):
            for scope, raw_paths in sorted(scopes.items()):
                paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
                for raw_path in paths:
                    run_dir = Path(raw_path)
                    for variant, csv_path in result_csvs(run_dir, scope):
                        if not csv_path.exists():
                            raise FileNotFoundError(f"Benchmark lag file not found: {csv_path}")
                        control = control_path(
                            args.controls_root, model, task, variant
                        )
                        if not control.exists():
                            raise FileNotFoundError(f"Control config not found: {control}")

                        control_config = load_yaml(control)
                        default_step = int(
                            control_config["training_params"]["lag_step_size"]
                        )
                        for batch in arithmetic_batches(
                            read_lags(csv_path), args.lags_per_job
                        ):
                            command = command_for_batch(
                                control,
                                batch,
                                default_step,
                                args.sbatch_flags,
                                args.config_overrides,
                            )
                            print(
                                f"{control}: {batch[0]}..{batch[-1]} "
                                f"({len(batch)} lags)"
                            )
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
