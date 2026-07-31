"""Paired permutation tests over saved out-of-fold prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class LagPredictions:
    """Pooled out-of-fold predictions for one result and lag."""

    sample_id: np.ndarray
    start: np.ndarray
    prediction: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class PredictionArtifact:
    """Prediction artifact loaded from one result run unit."""

    name: str
    result_dir: Path
    task_name: str
    lags: Mapping[int, LagPredictions]


def _fold_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 0, name


def load_prediction_artifact(name: str, result_dir: Path) -> PredictionArtifact:
    """Load and validate ``test_predictions.h5`` from a single run unit."""
    result_dir = Path(result_dir)
    artifact_path = result_dir / "test_predictions.h5"
    if not artifact_path.is_file():
        raise ValueError(
            f"Result {name!r} does not contain {artifact_path.name}: {result_dir}"
        )

    loaded_lags: dict[int, LagPredictions] = {}
    with h5py.File(artifact_path, "r") as artifact:
        schema_version = int(artifact.attrs.get("schema_version", -1))
        if schema_version != 1:
            raise ValueError(
                f"Result {name!r} has unsupported prediction schema {schema_version}"
            )
        raw_task_name = artifact.attrs.get("task_name", "")
        task_name = (
            raw_task_name.decode("utf-8")
            if isinstance(raw_task_name, bytes)
            else str(raw_task_name)
        )
        if not task_name:
            raise ValueError(f"Result {name!r} has no task_name artifact attribute")

        for group_name in sorted(artifact):
            if not group_name.startswith("lag_"):
                continue
            lag_group = artifact[group_name]
            lag = int(lag_group.attrs.get("lag_ms", group_name.removeprefix("lag_")))
            pieces: dict[str, list[np.ndarray]] = {
                "sample_id": [],
                "start": [],
                "prediction": [],
                "target": [],
            }
            fold_names = sorted(
                (key for key in lag_group if key.startswith("fold_")),
                key=_fold_sort_key,
            )
            if not fold_names:
                raise ValueError(f"Result {name!r}, lag {lag} has no fold groups")
            for fold_name in fold_names:
                fold = lag_group[fold_name]
                missing = [key for key in pieces if key not in fold]
                if missing:
                    raise ValueError(
                        f"Result {name!r}, lag {lag}, {fold_name} is missing {missing}"
                    )
                pieces["sample_id"].append(fold["sample_id"].asstr()[:])
                for key in ("start", "prediction", "target"):
                    pieces[key].append(fold[key][:])

            values = {
                key: np.concatenate(parts, axis=0) for key, parts in pieces.items()
            }
            row_count = len(values["sample_id"])
            if any(len(values[key]) != row_count for key in values):
                raise ValueError(f"Result {name!r}, lag {lag} has misaligned datasets")
            ids = values["sample_id"].astype(str)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Result {name!r}, lag {lag} has duplicate sample IDs")
            loaded_lags[lag] = LagPredictions(
                sample_id=ids,
                start=np.asarray(values["start"], dtype=np.float64),
                prediction=np.asarray(values["prediction"]),
                target=np.asarray(values["target"]),
            )

    if not loaded_lags:
        raise ValueError(f"Result {name!r} has no completed lag groups")
    return PredictionArtifact(name, result_dir.resolve(), task_name, loaded_lags)


def validate_artifact_tasks(artifacts: Sequence[PredictionArtifact]) -> str:
    task_names = {artifact.task_name for artifact in artifacts}
    if len(task_names) != 1:
        details = ", ".join(
            f"{artifact.name}={artifact.task_name}" for artifact in artifacts
        )
        raise ValueError(f"Prediction artifacts have different tasks: {details}")
    return next(iter(task_names))


def _index_by_id(data: LagPredictions) -> dict[str, int]:
    return {sample_id: idx for idx, sample_id in enumerate(data.sample_id)}


def align_lag_predictions(
    artifacts: Sequence[PredictionArtifact],
    lag: int,
    required_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Strictly align one lag across results, optionally selecting common IDs."""
    if not artifacts:
        raise ValueError("At least one prediction artifact is required")
    lag_data = [artifact.lags[lag] for artifact in artifacts]
    id_sets = [set(data.sample_id.tolist()) for data in lag_data]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        counts = ", ".join(
            f"{artifact.name}={len(ids)}" for artifact, ids in zip(artifacts, id_sets)
        )
        raise ValueError(f"Sample IDs differ between results at lag {lag}: {counts}")
    selected_ids = id_sets[0] if required_ids is None else required_ids
    if not selected_ids <= id_sets[0]:
        raise ValueError(f"Required sample IDs are unavailable at lag {lag}")

    reference_index = _index_by_id(lag_data[0])
    ordered_ids = np.array(
        sorted(
            selected_ids,
            key=lambda item: (lag_data[0].start[reference_index[item]], item),
        ),
        dtype=str,
    )
    aligned_targets: list[np.ndarray] = []
    aligned_starts: list[np.ndarray] = []
    aligned_predictions: list[np.ndarray] = []
    for data in lag_data:
        by_id = _index_by_id(data)
        indices = np.array([by_id[item] for item in ordered_ids], dtype=np.int64)
        aligned_targets.append(data.target[indices])
        aligned_starts.append(data.start[indices])
        aligned_predictions.append(data.prediction[indices])

    for index, (artifact, target, start) in enumerate(
        zip(artifacts[1:], aligned_targets[1:], aligned_starts[1:]), start=1
    ):
        if not np.array_equal(aligned_targets[0], target):
            raise ValueError(
                f"Targets differ between {artifacts[0].name!r} and {artifact.name!r} "
                f"at lag {lag}"
            )
        if not np.array_equal(aligned_starts[0], start):
            raise ValueError(
                f"Onsets differ between {artifacts[0].name!r} and {artifact.name!r} "
                f"at lag {lag}"
            )
        if aligned_predictions[0].shape != aligned_predictions[index].shape:
            raise ValueError(
                f"Prediction shapes differ between results at lag {lag}: "
                f"{[values.shape for values in aligned_predictions]}"
            )
    return ordered_ids, aligned_targets[0], aligned_predictions


def common_sample_ids(
    artifacts: Sequence[PredictionArtifact], lags: Sequence[int]
) -> set[str]:
    """Return the samples present for every candidate lag after strict pair alignment."""
    common: set[str] | None = None
    for lag in lags:
        align_lag_predictions(artifacts, lag)
        ids = set(artifacts[0].lags[lag].sample_id.tolist())
        common = ids if common is None else common & ids
    if not common:
        raise ValueError("No samples are shared across all candidate lags")
    return common


def compute_metric(
    metric_fn: Callable, prediction: np.ndarray, target: np.ndarray
) -> float:
    """Evaluate a registered metric over pooled out-of-fold predictions."""
    value = metric_fn(
        torch.as_tensor(prediction, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
    )
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Significance metrics must return one scalar value")
        result = float(value.detach().cpu().item())
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError("Significance metrics must return one scalar value")
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("Significance metric returned a non-finite value")
    return result


def block_swap_mask(
    sample_count: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one paired swap decision for each contiguous event block."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    block_count = (sample_count + block_size - 1) // block_size
    decisions = rng.integers(0, 2, size=block_count, dtype=np.int8).astype(bool)
    return np.repeat(decisions, block_size)[:sample_count]


def _swap_predictions(
    prediction_a: np.ndarray, prediction_b: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expanded_mask = mask.reshape((len(mask),) + (1,) * (prediction_a.ndim - 1))
    return (
        np.where(expanded_mask, prediction_b, prediction_a),
        np.where(expanded_mask, prediction_a, prediction_b),
    )


def paired_lag_test(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    target: np.ndarray,
    metric_fn: Callable,
    higher_is_better: bool,
    block_size: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Test whether A is better than B at one fixed lag."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    direction = 1.0 if higher_is_better else -1.0
    score_a = compute_metric(metric_fn, prediction_a, target)
    score_b = compute_metric(metric_fn, prediction_b, target)
    statistic = direction * (score_a - score_b)
    extreme = 0
    for _ in range(n_permutations):
        mask = block_swap_mask(len(target), block_size, rng)
        perm_a, perm_b = _swap_predictions(prediction_a, prediction_b, mask)
        perm_statistic = direction * (
            compute_metric(metric_fn, perm_a, target)
            - compute_metric(metric_fn, perm_b, target)
        )
        extreme += perm_statistic >= statistic - 1e-12
    return {
        "score_a": score_a,
        "score_b": score_b,
        "effect": statistic,
        "p_value": (extreme + 1) / (n_permutations + 1),
        "sample_count": len(target),
        "block_count": (len(target) + block_size - 1) // block_size,
    }


def paired_best_lag_test(
    predictions_a: Mapping[int, np.ndarray],
    predictions_b: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    metric_fn: Callable,
    higher_is_better: bool,
    block_size: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Test A > B while reselecting both best lags under every permutation."""
    lags = sorted(targets)
    if set(predictions_a) != set(lags) or set(predictions_b) != set(lags):
        raise ValueError("Both results must provide every candidate lag")
    sample_counts = {len(targets[lag]) for lag in lags}
    if len(sample_counts) != 1:
        raise ValueError("Best-lag testing requires one common sample set")
    sample_count = next(iter(sample_counts))
    direction = 1.0 if higher_is_better else -1.0

    scores_a = {
        lag: compute_metric(metric_fn, predictions_a[lag], targets[lag]) for lag in lags
    }
    scores_b = {
        lag: compute_metric(metric_fn, predictions_b[lag], targets[lag]) for lag in lags
    }
    best_lag_a = max(lags, key=lambda lag: (direction * scores_a[lag], -lag))
    best_lag_b = max(lags, key=lambda lag: (direction * scores_b[lag], -lag))
    statistic = direction * (scores_a[best_lag_a] - scores_b[best_lag_b])

    extreme = 0
    for _ in range(n_permutations):
        mask = block_swap_mask(sample_count, block_size, rng)
        perm_scores_a: list[float] = []
        perm_scores_b: list[float] = []
        for lag in lags:
            perm_a, perm_b = _swap_predictions(
                predictions_a[lag], predictions_b[lag], mask
            )
            perm_scores_a.append(compute_metric(metric_fn, perm_a, targets[lag]))
            perm_scores_b.append(compute_metric(metric_fn, perm_b, targets[lag]))
        perm_best_a = max(direction * score for score in perm_scores_a)
        perm_best_b = max(direction * score for score in perm_scores_b)
        extreme += perm_best_a - perm_best_b >= statistic - 1e-12

    return {
        "best_lag_a": best_lag_a,
        "best_lag_b": best_lag_b,
        "score_a": scores_a[best_lag_a],
        "score_b": scores_b[best_lag_b],
        "effect": statistic,
        "p_value": (extreme + 1) / (n_permutations + 1),
        "sample_count": sample_count,
        "block_count": (sample_count + block_size - 1) // block_size,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down familywise-error adjusted p-values."""
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running_max = 0.0
    test_count = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (test_count - rank) * p_value))
        adjusted[original_index] = running_max
    return adjusted


def apply_holm(rows: list[dict], alpha: float) -> pd.DataFrame:
    """Attach the sole first-version multiplicity correction to result rows."""
    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    output = []
    for row, corrected in zip(rows, adjusted):
        output.append(
            {
                **row,
                "p_value_holm": corrected,
                "significant": bool(corrected <= alpha),
            }
        )
    return pd.DataFrame(output)
