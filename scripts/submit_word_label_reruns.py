#!/usr/bin/env python
"""Submit the complete rerun matrix for tasks that use the canonical word-label CSV."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


TASKS = (
    "sentence_onset_task",
    "gpt_surprise_task",
    "gpt_surprise_multiclass_task",
    "content_noncontent_task",
    "pos_task",
)
FOUNDATION_MODELS = (
    "foundation_models/brainbert",
    "foundation_models/diver",
    "foundation_models/popt",
)
CNN_MODEL = "baselines/neural_conv_decoder"
LINEAR_MODEL = "baselines/simple_models"


@dataclass(frozen=True)
class Job:
    config: str
    trial_name: str
    overrides: tuple[str, ...]


def _config_tag(config: str) -> str:
    tag = re.sub(r"^configs/", "", config)
    tag = re.sub(r"\.ya?ml$", "", tag)
    return re.sub(r"[^A-Za-z0-9]+", "-", tag).strip("-")


def _trial_name(model: str, task: str, mode: str, unit: str, start: int, end: int) -> str:
    model_name = model.rsplit("/", 1)[-1]
    task_name = task.removesuffix("_task")
    return f"wordlabels-{model_name}-{task_name}-{mode}-{unit}-lag{start}-to-{end}"


def _lag_overrides(start: int, runtime_max: int, step: int) -> tuple[str, ...]:
    return (
        f"--training_params.min_lag={start}",
        f"--training_params.max_lag={runtime_max}",
        f"--training_params.lag_step_size={step}",
    )


def build_jobs(matrix_path: str = "training_matrix.yaml") -> list[Job]:
    with Path(matrix_path).open() as f:
        matrix = yaml.safe_load(f)
    jobs: list[Job] = []

    # Foundation models: one supersubject config and nine single-subject configs per task,
    # each split at 0 ms. Runtime max_lag is exclusive, so 1100 includes the +1000 ms lag.
    for model in FOUNDATION_MODELS:
        for task in TASKS:
            configs = matrix[model][task]
            selected = [
                config
                for config in configs
                if re.fullmatch(r"subject\d+_full\.yml", Path(config).name)
            ]
            # Foundation supersubject configs exist beside the subject configs but are not listed
            # in training_matrix.yaml, so include that requested run mode explicitly.
            selected.append(str(Path(selected[0]).with_name("supersubject.yml")))
            for config in selected:
                mode = "supersubject" if Path(config).name == "supersubject.yml" else "subject"
                unit = "all" if mode == "supersubject" else re.search(
                    r"subject(\d+)", Path(config).name
                ).group(1)
                for start, display_end, runtime_max in ((-1000, 0, 0), (0, 1000, 1100)):
                    trial = _trial_name(model, task, mode, unit, start, display_end)
                    jobs.append(Job(config, trial, _lag_overrides(start, runtime_max, 100)))

    # Torch linear baselines: one job for the complete supersubject sweep and one config-level
    # per-subject job that processes all configured subjects. Runtime max 1025 includes +1000.
    for task in TASKS:
        for config in matrix[LINEAR_MODEL][task]:
            name = Path(config).name
            if name not in {"supersubject.yml", "per_subject.yml"}:
                continue
            mode = "supersubject" if name == "supersubject.yml" else "per-subject"
            trial = _trial_name(LINEAR_MODEL, task, mode, "all", -1000, 1000)
            jobs.append(Job(config, trial, _lag_overrides(-1000, 1025, 25)))

    # CNN supersubject sweeps are split at zero, like foundation sweeps but at 25 ms steps.
    for task in TASKS:
        config = next(
            config
            for config in matrix[CNN_MODEL][task]
            if Path(config).name == "supersubject.yml"
        )
        for start, display_end, runtime_max in ((-1000, 0, 0), (0, 1000, 1025)):
            trial = _trial_name(CNN_MODEL, task, "supersubject", "all", start, display_end)
            jobs.append(Job(config, trial, _lag_overrides(start, runtime_max, 25)))

    # CNN subject and region sweeps use non-overlapping 250 ms chunks. Each job lets the config's
    # run mode process all configured subjects or all atlas regions together. The last chunk has
    # an exclusive runtime max of 1025 so the requested +1000 ms endpoint is included exactly once.
    chunks = [
        (start, start + 250, start + 250 if start < 750 else 1025)
        for start in range(-1000, 1000, 250)
    ]
    for task in TASKS:
        configs = matrix[CNN_MODEL][task]
        subject_config = next(c for c in configs if Path(c).name == "per_subject.yml")
        region_config = next(c for c in configs if Path(c).name == "per_region.yml")
        for start, display_end, runtime_max in chunks:
            trial = _trial_name(CNN_MODEL, task, "per-subject", "all", start, display_end)
            jobs.append(Job(
                subject_config,
                trial,
                _lag_overrides(start, runtime_max, 25),
            ))
            trial = _trial_name(CNN_MODEL, task, "per-region", "all", start, display_end)
            jobs.append(Job(
                region_config,
                trial,
                _lag_overrides(start, runtime_max, 25),
            ))

    if len({job.trial_name for job in jobs}) != len(jobs):
        raise ValueError("generated trial names are not unique")
    return jobs


def command_for(job: Job, sbatch_flags: str = "") -> list[str]:
    return [
        "sbatch",
        *shlex.split(sbatch_flags),
        f"--job-name={job.trial_name}",
        "submit.sh",
        "main.py",
        "--config",
        job.config,
        f"--trial_name={job.trial_name}",
        *job.overrides,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="training_matrix.yaml")
    parser.add_argument("--sbatch-flags", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.matrix)
    print(f"Prepared {len(jobs)} jobs with unique trial names")
    for job in jobs:
        command = command_for(job, args.sbatch_flags)
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
