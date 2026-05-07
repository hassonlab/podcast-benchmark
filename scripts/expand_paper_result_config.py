#!/usr/bin/env python3
"""Expand a paper-results config with matching result shards."""

from __future__ import annotations

import argparse
import csv
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import yaml

DEFAULT_MIN_LAG = -800
DEFAULT_MAX_LAG = 975
DEFAULT_STEP = 25
DEFAULT_REGIONS = ("EAC", "PC", "PRC", "IFG", "MTG", "ITG", "TPJ", "TP", "RIGHT")
SCOPES = ("super_subject", "per_subject", "per_region")
LAG_TRAINING_KEYS = {"lag", "min_lag", "max_lag", "lag_step_size"}
TASK_KEY_ALIASES = {
    "volume_level_decoding": "volume_level",
    "whisper_embedding_decoding": "whisper_embedding",
    "word_embedding_decoding": "word_embedding",
}


@dataclass(frozen=True)
class RunMetadata:
    path: Path
    task_name: str
    run_mode: str
    fingerprint: str
    subject_ids: tuple[str, ...]
    model_key: Optional[str] = None
    task_key: Optional[str] = None
    has_config: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_results_dense.yml"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--min-lag", type=int, default=DEFAULT_MIN_LAG)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--model-prefix",
        default="baseline",
        help="Only expand model keys equal to this value or starting with it.",
    )
    return parser.parse_args()


def load_yaml(path: Path, *, unsafe: bool = False):
    with path.open("r") as f:
        return (yaml.unsafe_load if unsafe else yaml.safe_load)(f) or {}


def as_path_list(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [Path(path) for path in value]
    raise TypeError(f"Expected path or path list, got {value!r}")


def normalize_path(path: Path, root: Path) -> Path:
    cleaned = Path(str(path).rstrip("/"))
    return cleaned if cleaned.is_absolute() else root / cleaned


def config_path_for_run(run_dir: Path) -> Path:
    return run_dir / "config.yml"


def read_lags(csv_path: Path) -> set[int]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lags" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain a 'lags' column")
        return {
            int(float(row["lags"]))
            for row in reader
            if row.get("lags", "").strip()
        }


def normalize_region_name(region_dir_name: str) -> str:
    return region_dir_name.removeprefix("region_").upper()


def coverage_for_run(run_dir: Path, scope: str) -> dict[str, set[int]]:
    if scope == "super_subject":
        csv_path = run_dir / "lag_performance.csv"
        return {"super_subject": read_lags(csv_path)} if csv_path.exists() else {}

    prefix = "subject_" if scope == "per_subject" else "region_"
    coverage = {}
    for csv_path in sorted(run_dir.glob(f"{prefix}*/lag_performance.csv")):
        entity = csv_path.parent.name
        if scope == "per_region":
            entity = normalize_region_name(entity)
        coverage[entity] = read_lags(csv_path)
    return coverage


def normalize_run_mode(value) -> str:
    text = str(value or "")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def normalize_task_key(value: str) -> str:
    task_key = str(value).removesuffix("_task")
    return TASK_KEY_ALIASES.get(task_key, task_key)


def scope_matches_run_mode(scope: str, run_mode: str) -> bool:
    valid_modes = {
        "super_subject": {"super_subject", "supersubject"},
        "per_subject": {"per_subject"},
        "per_region": {"per_region"},
    }[scope]
    return run_mode in valid_modes


def get_nested(config: Mapping, keys: Sequence[str]):
    value = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def normalized_for_fingerprint(value):
    if isinstance(value, Mapping):
        return {
            str(key): normalized_for_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if not (str(key) == "training_params")
        }
    if isinstance(value, (list, tuple)):
        return [normalized_for_fingerprint(item) for item in value]
    if hasattr(value, "item"):
        try:
            return normalized_for_fingerprint(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def model_fingerprint(config: Mapping) -> str:
    training_params = config.get("training_params", {})
    cleaned_training = {}
    if isinstance(training_params, Mapping):
        cleaned_training = {
            key: value
            for key, value in training_params.items()
            if key not in LAG_TRAINING_KEYS
        }
    payload = {
        "config_setter_name": config.get("config_setter_name"),
        "model_spec": config.get("model_spec"),
        "training_params": cleaned_training,
        "task_specific_config": get_nested(config, ("task_config", "task_specific_config")),
    }
    return json.dumps(normalized_for_fingerprint(payload), sort_keys=True)


def subject_ids_from_config(config: Mapping) -> tuple[str, ...]:
    ids = get_nested(config, ("task_config", "data_params", "subject_ids")) or []
    return tuple(str(int(subject_id)) for subject_id in ids)


def metadata_for_run(run_dir: Path) -> RunMetadata | None:
    config_path = config_path_for_run(run_dir)
    if not config_path.exists():
        return None
    try:
        config = load_yaml(config_path, unsafe=True)
    except Exception as exc:
        print(f"Skipping {run_dir}: could not read config.yml ({exc})")
        return None
    if not isinstance(config, Mapping):
        return None
    task_name = get_nested(config, ("task_config", "task_name"))
    run_mode = normalize_run_mode(config.get("run_mode"))
    if not task_name or not run_mode:
        return None
    return RunMetadata(
        path=run_dir,
        task_name=str(task_name),
        run_mode=run_mode,
        fingerprint=model_fingerprint(config),
        subject_ids=subject_ids_from_config(config),
        task_key=normalize_task_key(str(task_name)),
    )


def infer_cleaned_metadata(run_dir: Path) -> RunMetadata | None:
    parts = run_dir.parts
    try:
        root_index = parts.index("cleaned-paper-results")
    except ValueError:
        return None

    relative_parts = parts[root_index + 1 :]
    if len(relative_parts) != 3:
        return None

    model_key, task_key, scope = relative_parts
    run_mode = normalize_run_mode(scope)
    if run_mode not in SCOPES:
        return None
    if not coverage_for_run(run_dir, run_mode):
        return None

    return RunMetadata(
        path=run_dir,
        task_name=f"{task_key}_task",
        run_mode=run_mode,
        fingerprint="",
        subject_ids=(),
        model_key=model_key,
        task_key=task_key,
        has_config=False,
    )


def metadata_for_path(run_dir: Path) -> RunMetadata | None:
    metadata = metadata_for_run(run_dir)
    if metadata is not None:
        return metadata
    return infer_cleaned_metadata(run_dir)


def expected_lags(min_lag: int, max_lag: int, step: int) -> set[int]:
    if step <= 0:
        raise ValueError("--step must be positive")
    return set(range(min_lag, max_lag + 1, step))


def baseline_model(model: str, prefix: str) -> bool:
    return model == prefix or model.startswith(f"{prefix}_")


def discover_candidate_metadata(results_root: Path) -> list[RunMetadata]:
    candidates = []
    for config_path in sorted(results_root.rglob("config.yml")):
        metadata = metadata_for_run(config_path.parent)
        if metadata is not None:
            candidates.append(metadata)
    if results_root.name == "cleaned-paper-results":
        for scope_dir in sorted(
            path
            for path in results_root.glob("*/*/*")
            if path.is_dir() and path.name in SCOPES
        ):
            metadata = infer_cleaned_metadata(scope_dir)
            if metadata is not None:
                candidates.append(metadata)
    return candidates


def merge_coverage(target: dict[str, set[int]], update: Mapping[str, set[int]]) -> None:
    for entity, lags in update.items():
        target.setdefault(entity, set()).update(lags)


def configured_paths_for_group(config: Mapping, model: str, task: str, scope: str) -> list[Path]:
    return as_path_list(config.get("results", {}).get(model, {}).get(task, {}).get(scope))


def relative_or_original(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def required_entities(
    scope: str,
    configured_metadata: Sequence[RunMetadata],
    current_coverage: Mapping[str, set[int]],
) -> tuple[str, ...]:
    if scope == "super_subject":
        return ("super_subject",)
    if scope == "per_region":
        return DEFAULT_REGIONS

    subject_ids = sorted(
        {
            f"subject_{subject_id}"
            for metadata in configured_metadata
            for subject_id in metadata.subject_ids
        },
        key=lambda item: int(item.removeprefix("subject_")),
    )
    return tuple(subject_ids or sorted(current_coverage))


def missing_for_entities(
    coverage: Mapping[str, set[int]],
    entities: Iterable[str],
    desired_lags: set[int],
) -> dict[str, set[int]]:
    return {
        entity: desired_lags - coverage.get(entity, set())
        for entity in entities
        if desired_lags - coverage.get(entity, set())
    }


def useful_candidate_lags(
    candidate_coverage: Mapping[str, set[int]],
    current_coverage: Mapping[str, set[int]],
    required: Sequence[str],
    desired_lags: set[int],
) -> tuple[bool, bool, list[str]]:
    contributes = False
    blocked = False
    warnings = []
    required_set = set(required)
    for entity, lags in sorted(candidate_coverage.items()):
        if entity not in required_set:
            continue
        candidate_desired = lags & desired_lags
        if not candidate_desired:
            continue
        current = current_coverage.get(entity, set())
        overlap = candidate_desired & current
        new_lags = candidate_desired - current
        if new_lags:
            contributes = True
        if overlap and new_lags:
            blocked = True
            warnings.append(
                f"{entity}: overlap {format_lags(overlap)}, new {format_lags(new_lags)}"
            )
    return contributes, blocked, warnings


def format_lags(lags: Iterable[int]) -> str:
    values = sorted(lags)
    if len(values) > 12:
        return f"{values[0]}..{values[-1]} ({len(values)} lags)"
    return ", ".join(str(value) for value in values) or "none"


def expand_config(
    config: Mapping,
    config_root: Path,
    results_root: Path,
    desired_lags: set[int],
    model_prefix: str,
) -> dict:
    expanded = copy.deepcopy(config)
    candidates = discover_candidate_metadata(results_root)
    results = config.get("results", {})
    if not isinstance(results, Mapping):
        raise ValueError("Config key 'results' must be a mapping")

    for model, tasks in results.items():
        if not baseline_model(str(model), model_prefix) or not isinstance(tasks, Mapping):
            continue
        for task, scopes in tasks.items():
            if not isinstance(scopes, Mapping):
                continue
            for scope in SCOPES:
                raw_paths = configured_paths_for_group(config, model, task, scope)
                if not raw_paths:
                    continue

                resolved_paths = [normalize_path(path, config_root) for path in raw_paths]
                configured_metadata = [
                    metadata
                    for run_dir in resolved_paths
                    if (metadata := metadata_for_path(run_dir)) is not None
                ]
                task_keys = {
                    metadata.task_key or normalize_task_key(metadata.task_name)
                    for metadata in configured_metadata
                } or {str(task)}
                fingerprints = {
                    metadata.fingerprint
                    for metadata in configured_metadata
                    if metadata.has_config and metadata.fingerprint
                }
                run_modes = {metadata.run_mode for metadata in configured_metadata} or {scope}
                current_coverage: dict[str, set[int]] = {}
                for run_dir in resolved_paths:
                    merge_coverage(current_coverage, coverage_for_run(run_dir, scope))

                matching_candidates = []
                for candidate in candidates:
                    if candidate.model_key is not None and candidate.model_key != str(model):
                        continue
                    candidate_task_key = candidate.task_key or normalize_task_key(candidate.task_name)
                    if candidate_task_key not in task_keys and candidate_task_key != str(task):
                        continue
                    if (
                        fingerprints
                        and candidate.has_config
                        and candidate.fingerprint not in fingerprints
                    ):
                        continue
                    if candidate.run_mode not in run_modes or not scope_matches_run_mode(scope, candidate.run_mode):
                        continue
                    matching_candidates.append(candidate)

                required = required_entities(scope, configured_metadata, current_coverage)
                if not required and scope == "per_subject":
                    subject_entities = set(current_coverage)
                    for candidate in matching_candidates:
                        subject_entities.update(coverage_for_run(candidate.path, scope))
                    required = tuple(
                        sorted(
                            subject_entities,
                            key=lambda item: int(item.removeprefix("subject_")),
                        )
                    )
                if not missing_for_entities(current_coverage, required, desired_lags):
                    continue

                existing_resolved = {path.resolve() for path in resolved_paths}
                additions: list[Path] = []
                for candidate in matching_candidates:
                    if candidate.path.resolve() in existing_resolved:
                        continue

                    candidate_coverage = coverage_for_run(candidate.path, scope)
                    contributes, blocked, warnings = useful_candidate_lags(
                        candidate_coverage,
                        current_coverage,
                        required,
                        desired_lags,
                    )
                    if not contributes:
                        continue
                    if blocked:
                        for warning in warnings:
                            print(
                                "Partial overlap blocks merge for "
                                f"{model}/{task}/{scope}: {candidate.path} | {warning}"
                            )
                        continue

                    additions.append(candidate.path)
                    existing_resolved.add(candidate.path.resolve())
                    merge_coverage(current_coverage, candidate_coverage)
                    if not missing_for_entities(current_coverage, required, desired_lags):
                        break

                if additions:
                    output_paths = [str(path) for path in raw_paths]
                    output_paths.extend(relative_or_original(path, config_root) for path in additions)
                    expanded["results"][model][task][scope] = output_paths
                    print(
                        f"Added {len(additions)} path(s) for {model}/{task}/{scope}: "
                        + ", ".join(relative_or_original(path, config_root) for path in additions)
                    )

    return expanded


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    expanded = expand_config(
        config,
        Path.cwd(),
        normalize_path(args.results_root, Path.cwd()),
        expected_lags(args.min_lag, args.max_lag, args.step),
        args.model_prefix,
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w") as f:
        yaml.safe_dump(expanded, f, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
