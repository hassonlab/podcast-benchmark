#!/usr/bin/env python3
"""Materialize paper-result shards into one cleaned directory per config entry."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

CONDITIONS = ("super_subject", "per_subject", "per_region")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Defaults to <output-root>/paper_results.yml.",
    )
    parser.add_argument(
        "--path-base",
        type=Path,
        default=Path.cwd(),
        help="Base directory for resolving relative paths in the input config.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute result paths in the generated config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping:
    with path.open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def result_paths(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        paths = [Path(path) for path in value]
        if not paths:
            raise ValueError("Result path lists must contain at least one path")
        return paths
    raise TypeError(f"Result path must be a path string or list of paths, got {value!r}")


def resolve_path(path: Path, base: Path) -> Path:
    cleaned = Path(str(path).rstrip("/"))
    return cleaned if cleaned.is_absolute() else base / cleaned


def display_path(path: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def combine_lag_dataframes(frames: Sequence[pd.DataFrame], label: str) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"No lag performance dataframes to combine for {label}")
    if len(frames) == 1:
        return frames[0].copy()

    combined = pd.concat(frames, ignore_index=True)
    if "lags" not in combined.columns:
        return combined

    duplicated_lags = combined.loc[combined["lags"].duplicated(), "lags"].unique()
    if len(duplicated_lags):
        duplicate_text = ", ".join(str(lag) for lag in sorted(duplicated_lags))
        raise ValueError(
            f"Multiple result directories for {label} contain duplicate lags: "
            f"{duplicate_text}"
        )
    return combined.sort_values("lags").reset_index(drop=True)


def read_root_lag_frames(run_dirs: Sequence[Path], label: str) -> list[pd.DataFrame]:
    frames = []
    for run_dir in run_dirs:
        csv_path = run_dir / "lag_performance.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected {csv_path} for {label}")
        frames.append(pd.read_csv(csv_path))
    return frames


def entity_lag_frames(
    run_dirs: Sequence[Path],
    prefix: str,
    label: str,
) -> dict[str, list[pd.DataFrame]]:
    frames: dict[str, list[pd.DataFrame]] = {}
    for run_dir in run_dirs:
        found = False
        for csv_path in sorted(run_dir.glob(f"{prefix}*/lag_performance.csv")):
            found = True
            frames.setdefault(csv_path.parent.name, []).append(pd.read_csv(csv_path))
        if not found:
            raise FileNotFoundError(
                f"Expected {prefix}*/lag_performance.csv files under {run_dir} "
                f"for {label}"
            )
    return frames


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_super_subject(run_dirs: Sequence[Path], output_dir: Path, label: str) -> None:
    prepare_output_dir(output_dir)
    combined = combine_lag_dataframes(read_root_lag_frames(run_dirs, label), label)
    combined.to_csv(output_dir / "lag_performance.csv", index=False)


def write_entity_condition(
    run_dirs: Sequence[Path],
    output_dir: Path,
    prefix: str,
    label: str,
) -> None:
    prepare_output_dir(output_dir)
    for entity, frames in sorted(entity_lag_frames(run_dirs, prefix, label).items()):
        entity_dir = output_dir / entity
        entity_dir.mkdir(parents=True, exist_ok=True)
        combined = combine_lag_dataframes(frames, f"{label}/{entity}")
        combined.to_csv(entity_dir / "lag_performance.csv", index=False)


def write_per_subject(run_dirs: Sequence[Path], output_dir: Path, label: str) -> None:
    has_root_csv = [(run_dir / "lag_performance.csv").exists() for run_dir in run_dirs]
    has_subject_csvs = [
        any(run_dir.glob("subject_*/lag_performance.csv"))
        for run_dir in run_dirs
    ]

    if any(has_root_csv) and any(has_subject_csvs):
        raise ValueError(
            f"Result directories for {label} mix direct lag_performance.csv files "
            "and subject_*/lag_performance.csv files"
        )
    if any(has_root_csv):
        write_super_subject(run_dirs, output_dir, label)
    else:
        write_entity_condition(run_dirs, output_dir, "subject_", label)


def clean_config(
    config: Mapping,
    output_root: Path,
    path_base: Path,
    absolute_paths: bool,
) -> dict:
    cleaned = copy.deepcopy(config)
    results = config.get("results", {})
    if not isinstance(results, Mapping):
        raise ValueError("Config key 'results' must be a mapping")

    for model, tasks in results.items():
        if not isinstance(tasks, Mapping):
            continue
        for task, conditions in tasks.items():
            if not isinstance(conditions, Mapping):
                continue
            for condition in CONDITIONS:
                raw_value = conditions.get(condition)
                if not raw_value:
                    continue

                run_dirs = [
                    resolve_path(path, path_base)
                    for path in result_paths(raw_value)
                ]
                label = f"{model}/{task}/{condition}"
                output_dir = output_root / str(model) / str(task) / condition

                if condition == "super_subject":
                    write_super_subject(run_dirs, output_dir, label)
                elif condition == "per_subject":
                    write_per_subject(run_dirs, output_dir, label)
                elif condition == "per_region":
                    write_entity_condition(run_dirs, output_dir, "region_", label)
                else:
                    raise ValueError(f"Unsupported condition: {condition}")

                cleaned["results"][model][task][condition] = display_path(
                    output_dir,
                    absolute_paths,
                )
                print(f"Wrote {label} -> {output_dir}")

    return cleaned


def main() -> int:
    args = parse_args()
    output_config = args.output_config or args.output_root / "paper_results.yml"
    config = load_yaml(args.config)
    cleaned = clean_config(
        config,
        args.output_root,
        args.path_base,
        args.absolute_paths,
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w") as f:
        yaml.safe_dump(cleaned, f, sort_keys=False)
    print(f"Wrote cleaned config -> {output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
