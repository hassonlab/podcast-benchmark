#!/usr/bin/env python3
"""Submit multi-lag control jobs for foundation benchmark conditions.

By default each condition is one job that uses the lags baked into its control
config. With ``--full-lags`` the lag grid is overridden and split into chunks that
fit the per-job time budget.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


FOUNDATION_MODELS = ("brainbert", "diver", "popt")
RUN_SCOPES = ("super_subject", "per_subject")

# Worst-subject hours per lag measured from single-seed random-init logs
# (Elapsed Time / lags, so includes one-time feature caching). Super-subject
# volume-level jobs never finished in the logs; they reuse these values, so
# re-measure after the first chunk.
LAG_HOURS = {
    ("brainbert", "content_noncontent"): 0.21,
    ("brainbert", "gpt_surprise"): 0.10,
    ("brainbert", "gpt_surprise_multiclass"): 0.17,
    ("brainbert", "iu_boundary"): 0.07,
    ("brainbert", "pos"): 0.20,
    ("brainbert", "sentence_onset"): 0.07,
    ("brainbert", "volume_level"): 0.73,
    ("brainbert", "whisper_embedding"): 0.24,
    ("brainbert", "word_embedding"): 0.19,
    ("diver", "content_noncontent"): 0.39,
    ("diver", "gpt_surprise"): 0.27,
    ("diver", "gpt_surprise_multiclass"): 0.42,
    ("diver", "iu_boundary"): 0.10,
    ("diver", "pos"): 0.36,
    ("diver", "sentence_onset"): 0.11,
    ("diver", "volume_level"): 1.40,
    ("diver", "whisper_embedding"): 0.45,
    ("diver", "word_embedding"): 0.41,
    ("popt", "content_noncontent"): 0.18,
    ("popt", "gpt_surprise"): 0.10,
    ("popt", "gpt_surprise_multiclass"): 0.16,
    ("popt", "iu_boundary"): 0.04,
    ("popt", "pos"): 0.21,
    ("popt", "sentence_onset"): 0.04,
    ("popt", "volume_level"): 0.59,
    ("popt", "whisper_embedding"): 0.14,
    ("popt", "word_embedding"): 0.11,
}
DEFAULT_LAG_HOURS = 0.5
SAFETY_FRACTION = 0.85


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
    parser.add_argument(
        "--full-lags",
        action="store_true",
        help="Override the control lag grid and split it into time-budgeted chunks.",
    )
    parser.add_argument("--min-lag", type=int, default=-1000)
    parser.add_argument("--max-lag", type=int, default=1000, help="Inclusive.")
    parser.add_argument("--lag-step", type=int, default=100)
    parser.add_argument("--job-hours", type=float, default=10.0)
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
    sbatch_flags: str,
    config_overrides: str,
    control_name: str = "random-init",
    parsable: bool = False,
) -> list[str]:
    command = [
        "sbatch",
        *(["--parsable"] if parsable else []),
        f"--job-name={job_name(config, control_name)}",
        "--dependency=singleton",
        *shlex.split(sbatch_flags),
        "submit.sh",
        "main.py",
        "--config",
        str(config),
        *shlex.split(config_overrides),
    ]
    return command


def lag_grid(min_lag: int, max_lag: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("lag step must be positive")
    if min_lag > max_lag:
        raise ValueError("min lag must not exceed max lag")
    return list(range(min_lag, max_lag + 1, step))


def lags_per_job(model: str, task: str, job_hours: float) -> int:
    hours_per_lag = LAG_HOURS.get((model, task), DEFAULT_LAG_HOURS)
    return max(1, math.floor(job_hours * SAFETY_FRACTION / hours_per_lag))


def plan_lag_chunks(lags: Sequence[int], per_job: int) -> list[list[int]]:
    """Split lags into balanced consecutive chunks of at most ``per_job`` lags."""
    chunk_count = math.ceil(len(lags) / per_job)
    base, extra = divmod(len(lags), chunk_count)
    chunks, start = [], 0
    for index in range(chunk_count):
        size = base + (1 if index < extra else 0)
        chunks.append(list(lags[start : start + size]))
        start += size
    return chunks


def chunk_trial_name(base_trial_name: str, chunk: Sequence[int]) -> str:
    return f"{base_trial_name}_lags_{chunk[0]}_{chunk[-1]}"


def chunk_overrides(
    base_trial_name: str, chunk: Sequence[int], step: int, config_overrides: str
) -> str:
    # num_null_repetitions=1 keeps a single seed per lag. Extra overrides come
    # first so a user cannot silently break the chunk's lag range.
    parts = [
        config_overrides,
        "--training_params.num_null_repetitions=1",
        f"--training_params.min_lag={chunk[0]}",
        f"--training_params.max_lag={chunk[-1] + step}",
        f"--training_params.lag_step_size={step}",
        f"--trial_name={chunk_trial_name(base_trial_name, chunk)}",
    ]
    return " ".join(part for part in parts if part)


def sbatch_time_flag(job_hours: float) -> str:
    total_minutes = int(round(job_hours * 60))
    return f"--time={total_minutes // 60:02d}:{total_minutes % 60:02d}:00"


def submit(command: list[str], dry_run: bool, parsable: bool) -> str:
    print(shlex.join(command))
    if dry_run:
        return "DRYRUN"
    result = subprocess.run(command, check=True, capture_output=parsable, text=True)
    return result.stdout.strip().split(";")[0] if parsable else ""


def main() -> int:
    args = parse_args()
    config = load_yaml(args.results_config)
    results = config.get("results", {})
    submitted = 0
    controls = set()

    full_lags = lag_grid(args.min_lag, args.max_lag, args.lag_step)
    sbatch_flags = args.sbatch_flags
    if args.full_lags:
        sbatch_flags = f"{sbatch_time_flag(args.job_hours)} {sbatch_flags}".strip()

    for model in FOUNDATION_MODELS:
        for task, scopes in sorted(results.get(model, {}).items()):
            for scope, raw_paths in sorted(scopes.items()):
                # Skips nested entries such as random_init (existing controls).
                if scope not in RUN_SCOPES:
                    continue
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

                        if not args.full_lags:
                            command = command_for_condition(
                                control,
                                sbatch_flags,
                                args.config_overrides,
                                args.control_name,
                            )
                            print(f"{control}: configured lags")
                            submit(command, args.dry_run, parsable=False)
                            submitted += 1
                            controls.add(control)
                            continue

                        base_trial = load_yaml(control)["trial_name"]
                        chunks = plan_lag_chunks(
                            full_lags,
                            lags_per_job(model, task, args.job_hours),
                        )
                        print(f"{control}: {len(chunks)} job(s) of <= "
                              f"{len(chunks[0])} lags")
                        for chunk in chunks:
                            command = command_for_condition(
                                control,
                                sbatch_flags,
                                chunk_overrides(
                                    base_trial,
                                    chunk,
                                    args.lag_step,
                                    args.config_overrides,
                                ),
                                args.control_name,
                            )
                            submit(command, args.dry_run, parsable=False)
                            submitted += 1
                        controls.add(control)

    action = "would submit" if args.dry_run else "submitted"
    print(f"{action} {submitted} jobs for {len(controls)} control configs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
