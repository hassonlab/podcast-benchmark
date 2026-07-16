#!/usr/bin/env python3
"""Materialize paper-result shards into one cleaned directory per config entry."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

CONDITIONS = ("super_subject", "per_subject", "per_region")
LAG_KEYS = ("lag", "min_lag", "max_lag", "lag_step_size")
RUN_MODES = {
    "super_subject": "combined",
    "per_subject": "per_subject",
    "per_region": "per_region",
}


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


def load_run_config(run_dir: Path, label: str) -> dict:
    path = run_dir / "config.yml"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} for {label}")
    with path.open("r") as f:
        # main.py historically wrote RunMode values with Python YAML tags.
        config = yaml.unsafe_load(f) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Run config must be a mapping: {path}")
    return plain_yaml_value(config)


def plain_yaml_value(value: Any) -> Any:
    """Convert saved runtime values into data accepted by yaml.safe_dump."""
    if isinstance(value, Enum):
        return plain_yaml_value(value.value)
    if isinstance(value, Mapping):
        return {
            plain_yaml_value(key): plain_yaml_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [plain_yaml_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return plain_yaml_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


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
    raise TypeError(
        f"Result path must be a path string or list of paths, got {value!r}"
    )


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
    return (
        combined.drop_duplicates(subset="lags", keep="last")
        .sort_values("lags")
        .reset_index(drop=True)
    )


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


def collect_entity_frames(
    run_dirs: Sequence[Path], prefix: str, label: str
) -> dict[str, pd.DataFrame]:
    return {
        entity: combine_lag_dataframes(frames, f"{label}/{entity}")
        for entity, frames in sorted(entity_lag_frames(run_dirs, prefix, label).items())
    }


def collect_per_subject_frames(
    run_dirs: Sequence[Path], label: str
) -> dict[str, pd.DataFrame]:
    has_root_csv = [(run_dir / "lag_performance.csv").exists() for run_dir in run_dirs]
    has_subject_csvs = [
        any(run_dir.glob("subject_*/lag_performance.csv")) for run_dir in run_dirs
    ]
    if any(has_root_csv) and any(has_subject_csvs):
        raise ValueError(
            f"Result directories for {label} mix direct lag_performance.csv files "
            "and subject_*/lag_performance.csv files"
        )
    if any(has_root_csv):
        if not all(has_root_csv):
            raise FileNotFoundError(
                f"Some result directories for {label} have no lag CSVs"
            )
        subject_frames: dict[str, list[pd.DataFrame]] = {}
        for run_dir in run_dirs:
            match = re.search(r"subject[_-]?(\d+)", run_dir.name)
            if not match:
                raise ValueError(
                    f"Direct lag_performance.csv directory {run_dir} for {label} "
                    "must identify its subject in the directory name"
                )
            subject = f"subject_{int(match.group(1))}"
            subject_frames.setdefault(subject, []).append(
                pd.read_csv(run_dir / "lag_performance.csv")
            )
        return {
            subject: combine_lag_dataframes(frames, f"{label}/{subject}")
            for subject, frames in sorted(subject_frames.items())
        }
    return collect_entity_frames(run_dirs, "subject_", label)


def collect_condition_frames(
    run_dirs: Sequence[Path], condition: str, label: str
) -> dict[str, pd.DataFrame]:
    if condition == "super_subject":
        return {
            "": combine_lag_dataframes(read_root_lag_frames(run_dirs, label), label)
        }
    if condition == "per_subject":
        return collect_per_subject_frames(run_dirs, label)
    if condition == "per_region":
        return collect_entity_frames(run_dirs, "region_", label)
    raise ValueError(f"Unsupported condition: {condition}")


def normalized_for_comparison(config: Mapping) -> dict:
    normalized = copy.deepcopy(dict(config))
    # Older saved configs omitted this optional field while newer configs write
    # it explicitly as null. Those forms have the same runtime meaning.
    normalized.setdefault("atlas_path", None)
    for key in ("run_mode", "regions", "trial_name", "format_fields"):
        normalized.pop(key, None)
    training = normalized.get("training_params")
    if isinstance(training, Mapping):
        training = dict(training)
        for key in LAG_KEYS:
            training.pop(key, None)
        # Logging changes neither training nor the result values being joined.
        training.pop("tensorboard_logging", None)
        normalized["training_params"] = training
    data = normalized.get("task_config", {}).get("data_params")
    if isinstance(data, Mapping):
        data = dict(data)
        data.pop("subject_ids", None)
        data.pop("per_subject_electrodes", None)
        # Chunking controls temporary storage and memory use. It does not change
        # the experiment scope represented by the joined result table.
        data.pop("chunked_preprocessing", None)
        normalized["task_config"] = dict(normalized["task_config"])
        normalized["task_config"]["data_params"] = data
    return normalized


def configured_lag_bounds(
    config: Mapping, model: str, task: str
) -> tuple[int | None, int | None] | None:
    bounds = (
        config.get("cleaning", {})
        .get("lag_bounds", {})
        .get(model, {})
        .get(task)
    )
    if bounds is None:
        return None
    if not isinstance(bounds, Mapping):
        raise TypeError(f"cleaning.lag_bounds.{model}.{task} must be a mapping")
    minimum = bounds.get("min")
    maximum = bounds.get("max")
    if minimum is None and maximum is None:
        raise ValueError(
            f"cleaning.lag_bounds.{model}.{task} must set min and/or max"
        )
    minimum = int(minimum) if minimum is not None else None
    maximum = int(maximum) if maximum is not None else None
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(
            f"cleaning.lag_bounds.{model}.{task} has min greater than max"
        )
    return minimum, maximum


def apply_lag_bounds(
    frames: Mapping[str, pd.DataFrame],
    bounds: tuple[int | None, int | None] | None,
    label: str,
) -> dict[str, pd.DataFrame]:
    if bounds is None:
        return dict(frames)
    minimum, maximum = bounds
    filtered = {}
    for entity, frame in frames.items():
        if "lags" not in frame.columns:
            raise ValueError(
                f"lag_performance.csv for {label}/{entity} has no 'lags' column"
            )
        numeric = pd.to_numeric(frame["lags"], errors="raise")
        keep = pd.Series(True, index=frame.index)
        if minimum is not None:
            keep &= numeric >= minimum
        if maximum is not None:
            keep &= numeric <= maximum
        selected = frame.loc[keep].reset_index(drop=True)
        if selected.empty:
            raise ValueError(
                f"Lag bounds remove every result row for {label}/{entity}"
            )
        filtered[entity] = selected
    return filtered


def merged_subject_metadata(
    configs: Sequence[Mapping], label: str
) -> tuple[list[int], dict | None]:
    subject_ids: set[int] = set()
    electrodes: dict[int, Any] = {}
    saw_electrodes = False
    for config in configs:
        data = config.get("task_config", {}).get("data_params", {})
        for subject_id in data.get("subject_ids") or []:
            subject_ids.add(int(subject_id))
        mapping = data.get("per_subject_electrodes")
        if mapping is None:
            continue
        saw_electrodes = True
        for subject_id, names in mapping.items():
            subject_id = int(subject_id)
            if subject_id in electrodes and electrodes[subject_id] != names:
                raise ValueError(
                    f"Source configs for {label} disagree on electrodes for "
                    f"subject {subject_id}"
                )
            electrodes[subject_id] = names
    selected = sorted(subject_ids)
    selected_electrodes = {
        subject_id: electrodes[subject_id]
        for subject_id in selected
        if subject_id in electrodes
    }
    return selected, (selected_electrodes if saw_electrodes else None)


def exact_lags(frames: Mapping[str, pd.DataFrame], label: str) -> list[int]:
    expected: list[int] | None = None
    expected_entity = ""
    for entity, frame in frames.items():
        if "lags" not in frame.columns:
            raise ValueError(
                f"lag_performance.csv for {label}/{entity} has no 'lags' column"
            )
        numeric = pd.to_numeric(frame["lags"], errors="raise")
        if numeric.isna().any() or any(float(lag) != int(lag) for lag in numeric):
            raise ValueError(
                f"Runnable config generation requires integral lags for {label}/{entity}"
            )
        lags = sorted({int(lag) for lag in numeric})
        if not lags:
            raise ValueError(f"No lags found for {label}/{entity}")
        if expected is None:
            expected, expected_entity = lags, entity
        elif lags != expected:
            raise ValueError(
                f"Entities {expected_entity or '<root>'} and {entity} for {label} "
                "do not have the same lag coverage"
            )
    if expected is None:
        raise ValueError(f"No lag results found for {label}")
    return expected


def apply_lags(config: dict, lags: Sequence[int], label: str) -> None:
    training = config.setdefault("training_params", {})
    if len(lags) == 1:
        training["lag"] = lags[0]
        return
    steps = {right - left for left, right in zip(lags, lags[1:])}
    if len(steps) != 1 or next(iter(steps)) <= 0:
        raise ValueError(
            f"Lags for {label} are not one regular range and cannot be represented "
            "by a single runnable config"
        )
    step = next(iter(steps))
    training["lag"] = None
    training["min_lag"] = lags[0]
    training["max_lag"] = lags[-1] + step
    training["lag_step_size"] = step


def joined_run_config(
    run_dirs: Sequence[Path],
    frames: Mapping[str, pd.DataFrame],
    condition: str,
    model: str,
    task: str,
    label: str,
) -> dict:
    configs = [load_run_config(run_dir, label) for run_dir in run_dirs]
    reference = normalized_for_comparison(configs[0])
    for run_dir, config in zip(run_dirs[1:], configs[1:]):
        if normalized_for_comparison(config) != reference:
            raise ValueError(
                f"Source config {run_dir / 'config.yml'} conflicts with other "
                f"non-scope settings for {label}"
            )

    joined = copy.deepcopy(configs[-1])
    joined["run_mode"] = RUN_MODES[condition]
    joined["trial_name"] = f"{model}_{task}_{condition}"
    joined["format_fields"] = None
    subject_ids, electrodes = merged_subject_metadata(configs, label)
    data = joined.setdefault("task_config", {}).setdefault("data_params", {})
    data["subject_ids"] = subject_ids
    data["per_subject_electrodes"] = electrodes
    chunked = data.get("chunked_preprocessing")
    if isinstance(chunked, Mapping):
        chunked = dict(chunked)
        chunked["cache_dir"] = ".cache/preprocessed_chunks"
        data["chunked_preprocessing"] = chunked
    if condition == "per_region":
        joined["regions"] = sorted(
            entity.removeprefix("region_").upper() for entity in frames
        )
    else:
        joined["regions"] = None
    apply_lags(joined, exact_lags(frames, label), label)
    return joined


def write_condition(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    run_config: Mapping,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for entity, frame in frames.items():
        entity_dir = output_dir / entity if entity else output_dir
        entity_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(entity_dir / "lag_performance.csv", index=False)
    with (output_dir / "config.yml").open("w") as f:
        yaml.safe_dump(dict(run_config), f, sort_keys=False)


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
                    resolve_path(path, path_base) for path in result_paths(raw_value)
                ]
                label = f"{model}/{task}/{condition}"
                output_dir = output_root / str(model) / str(task) / condition

                # Collect and validate everything before replacing existing output.
                frames = collect_condition_frames(run_dirs, condition, label)
                frames = apply_lag_bounds(
                    frames,
                    configured_lag_bounds(config, str(model), str(task)),
                    label,
                )
                run_config = joined_run_config(
                    run_dirs, frames, condition, str(model), str(task), label
                )
                write_condition(output_dir, frames, run_config)
                cleaned["results"][model][task][condition] = display_path(
                    output_dir, absolute_paths
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
