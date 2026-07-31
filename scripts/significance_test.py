#!/usr/bin/env python
"""Run paired prediction-artifact permutation significance tests."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import metrics  # noqa: F401,E402 - import registers metric callables
from core.registry import metric_registry  # noqa: E402
from utils.significance_utils import (  # noqa: E402
    PredictionArtifact,
    align_lag_predictions,
    apply_holm,
    common_sample_ids,
    compute_metric,
    load_prediction_artifact,
    paired_best_lag_test,
    paired_lag_test,
    validate_artifact_tasks,
)


MODE_BEST_LAG = "best_lag"
MODE_BASELINE_LAGS = "baseline_lags"
VALID_MODES = {MODE_BEST_LAG, MODE_BASELINE_LAGS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired block-permutation tests over test_predictions.h5 artifacts."
    )
    parser.add_argument("config", type=Path, help="Analysis YAML configuration")
    return parser.parse_args(argv)


def _mapping(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _resolve_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _load_result_specs(
    raw_specs, config_dir: Path, field: str
) -> tuple[list[PredictionArtifact], list[dict[str, str]]]:
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"{field} must be a non-empty list of named result specs")
    artifacts = []
    normalized = []
    names: set[str] = set()
    for index, raw_spec in enumerate(raw_specs):
        spec = _mapping(raw_spec, f"{field}[{index}]")
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError(f"{field}[{index}].name must be non-empty")
        if name in names:
            raise ValueError(f"Duplicate result name {name!r}")
        if "path" not in spec:
            raise ValueError(f"{field}[{index}].path is required")
        path = _resolve_path(spec["path"], config_dir)
        artifacts.append(load_prediction_artifact(name, path))
        normalized.append({"name": name, "path": str(path)})
        names.add(name)
    return artifacts, normalized


def load_analysis_config(path: Path) -> tuple[dict, dict]:
    config_path = path.resolve()
    with config_path.open("r") as stream:
        raw = yaml.safe_load(stream) or {}
    raw = _mapping(raw, "config")
    mode = str(raw.get("mode", "")).strip()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")

    metric_spec = _mapping(raw.get("metric"), "metric")
    metric_name = str(metric_spec.get("name", "")).strip()
    if metric_name not in metric_registry:
        raise ValueError(
            f"Unknown metric {metric_name!r}; available metrics: "
            f"{sorted(metric_registry)}"
        )
    if not isinstance(metric_spec.get("higher_is_better"), bool):
        raise ValueError("metric.higher_is_better must be true or false")

    config_dir = config_path.parent
    results, normalized_results = _load_result_specs(
        raw.get("results"), config_dir, "results"
    )
    if len({artifact.result_dir for artifact in results}) != len(results):
        raise ValueError("Each result spec must point to a different run directory")
    baseline = None
    normalized_baseline = None
    if mode == MODE_BEST_LAG:
        if len(results) < 2:
            raise ValueError("best_lag mode requires at least two results")
    else:
        baseline_values, normalized_values = _load_result_specs(
            [raw.get("baseline")], config_dir, "baseline"
        )
        baseline = baseline_values[0]
        normalized_baseline = normalized_values[0]
        if baseline.name in {artifact.name for artifact in results}:
            raise ValueError("The baseline and comparison result names must differ")
        if baseline.result_dir in {artifact.result_dir for artifact in results}:
            raise ValueError("The baseline and comparison result paths must differ")

    valid_lags_raw = raw.get("valid_lags")
    if valid_lags_raw is not None:
        if not isinstance(valid_lags_raw, list) or not valid_lags_raw:
            raise ValueError("valid_lags must be null or a non-empty list")
        valid_lags = sorted({int(lag) for lag in valid_lags_raw})
    else:
        valid_lags = None

    block_size = int(raw.get("block_size", 1))
    n_permutations = int(raw.get("n_permutations", 10_000))
    random_seed = int(raw.get("random_seed", 42))
    alpha = float(raw.get("alpha", 0.05))
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if "output_dir" not in raw:
        raise ValueError("output_dir is required")
    output_dir = _resolve_path(raw["output_dir"], config_dir)
    input_dirs = {artifact.result_dir for artifact in results}
    if baseline is not None:
        input_dirs.add(baseline.result_dir)
    if output_dir in input_dirs:
        raise ValueError("output_dir must not overwrite an input result directory")

    runtime = {
        "mode": mode,
        "metric_name": metric_name,
        "metric_fn": metric_registry[metric_name],
        "higher_is_better": metric_spec["higher_is_better"],
        "results": results,
        "baseline": baseline,
        "valid_lags": valid_lags,
        "block_size": block_size,
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "alpha": alpha,
        "output_dir": output_dir,
    }
    normalized = {
        "mode": mode,
        "metric": {
            "name": metric_name,
            "higher_is_better": metric_spec["higher_is_better"],
        },
        "results": normalized_results,
        "baseline": normalized_baseline,
        "valid_lags": valid_lags,
        "block_size": block_size,
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "alpha": alpha,
        "multiple_comparisons": "holm",
        "output_dir": str(output_dir),
    }
    return runtime, normalized


def _candidate_lags(
    artifacts: Sequence[PredictionArtifact], valid_lags: Sequence[int] | None
) -> list[int]:
    available = set(artifacts[0].lags)
    for artifact in artifacts[1:]:
        available &= set(artifact.lags)
    if valid_lags is not None:
        available &= set(valid_lags)
    if not available:
        raise ValueError("No candidate lags are available across the requested results")
    return sorted(available)


def run_best_lag(config: Mapping) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = config["results"]
    validate_artifact_tasks(artifacts)
    lags = _candidate_lags(artifacts, config["valid_lags"])
    shared_ids = common_sample_ids(artifacts, lags)

    predictions: dict[str, dict[int, np.ndarray]] = {
        artifact.name: {} for artifact in artifacts
    }
    targets: dict[int, np.ndarray] = {}
    reference_ids = None
    reference_target = None
    for lag in lags:
        ordered_ids, target, aligned = align_lag_predictions(
            artifacts, lag, shared_ids
        )
        if reference_ids is not None and not np.array_equal(
            reference_ids, ordered_ids
        ):
            raise ValueError(
                "Sample onset order differs across candidate lags on the common set"
            )
        if reference_target is not None and not np.array_equal(
            reference_target, target
        ):
            raise ValueError(
                "Targets differ across candidate lags on the common sample set"
            )
        reference_ids = ordered_ids
        reference_target = target
        targets[lag] = target
        for artifact, values in zip(artifacts, aligned):
            predictions[artifact.name][lag] = values

    direction = 1.0 if config["higher_is_better"] else -1.0
    observed_best: dict[str, tuple[int, float]] = {}
    for artifact in artifacts:
        scores = {
            lag: compute_metric(
                config["metric_fn"], predictions[artifact.name][lag], targets[lag]
            )
            for lag in lags
        }
        best_lag = max(lags, key=lambda lag: (direction * scores[lag], -lag))
        observed_best[artifact.name] = (best_lag, scores[best_lag])
    winner_index = max(
        range(len(artifacts)),
        key=lambda index: (
            direction * observed_best[artifacts[index].name][1],
            -index,
        ),
    )
    winner = artifacts[winner_index].name

    ordered_pairs = [
        (artifact_a, artifact_b)
        for artifact_a in artifacts
        for artifact_b in artifacts
        if artifact_a.name != artifact_b.name
    ]
    seed_sequences = np.random.SeedSequence(config["random_seed"]).spawn(
        len(ordered_pairs)
    )
    rows = []
    for (artifact_a, artifact_b), seed_sequence in zip(ordered_pairs, seed_sequences):
        result = paired_best_lag_test(
            predictions[artifact_a.name],
            predictions[artifact_b.name],
            targets,
            config["metric_fn"],
            config["higher_is_better"],
            config["block_size"],
            config["n_permutations"],
            np.random.default_rng(seed_sequence),
        )
        rows.append(
            {
                "result_a": artifact_a.name,
                "result_b": artifact_b.name,
                "best_lag_a": result["best_lag_a"],
                "best_lag_b": result["best_lag_b"],
                "score_a": result["score_a"],
                "score_b": result["score_b"],
                "effect": result["effect"],
                "p_value": result["p_value"],
                "sample_count": result["sample_count"],
                "block_count": result["block_count"],
                "n_permutations": config["n_permutations"],
            }
        )
    all_pairs = apply_holm(rows, config["alpha"])
    winner_rows = all_pairs[all_pairs["result_a"] == winner].reset_index(drop=True)
    return winner_rows, all_pairs


def run_baseline_lags(config: Mapping) -> pd.DataFrame:
    baseline = config["baseline"]
    results = config["results"]
    validate_artifact_tasks([baseline, *results])
    comparisons: list[tuple[PredictionArtifact, int]] = []
    for result in results:
        lags = _candidate_lags([baseline, result], config["valid_lags"])
        comparisons.extend((result, lag) for lag in lags)

    seed_sequences = np.random.SeedSequence(config["random_seed"]).spawn(
        len(comparisons)
    )
    rows = []
    for (result, lag), seed_sequence in zip(comparisons, seed_sequences):
        _, target, predictions = align_lag_predictions([result, baseline], lag)
        test = paired_lag_test(
            predictions[0],
            predictions[1],
            target,
            config["metric_fn"],
            config["higher_is_better"],
            config["block_size"],
            config["n_permutations"],
            np.random.default_rng(seed_sequence),
        )
        rows.append(
            {
                "result": result.name,
                "baseline": baseline.name,
                "lag": lag,
                "result_score": test["score_a"],
                "baseline_score": test["score_b"],
                "effect": test["effect"],
                "p_value": test["p_value"],
                "sample_count": test["sample_count"],
                "block_count": test["block_count"],
                "n_permutations": config["n_permutations"],
            }
        )
    return apply_holm(rows, config["alpha"])


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(values: Mapping, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w") as stream:
            yaml.safe_dump(dict(values), stream, sort_keys=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(config_path: Path) -> Path:
    config, normalized = load_analysis_config(config_path)
    if config["mode"] == MODE_BEST_LAG:
        p_values, all_pairs = run_best_lag(config)
    else:
        p_values = run_baseline_lags(config)
        all_pairs = None

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(p_values, output_dir / "p_values.csv")
    if all_pairs is not None:
        _atomic_csv(all_pairs, output_dir / "all_pairwise_tests.csv")
    _atomic_yaml(normalized, output_dir / "config.yml")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = run(args.config)
    print(f"Wrote significance results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
