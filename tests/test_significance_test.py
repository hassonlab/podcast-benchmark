from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from metrics.regression_metrics import mse_metric
from scripts.significance_test import run
from utils import significance_utils
from utils.significance_utils import (
    LagPredictions,
    PredictionArtifact,
    align_lag_predictions,
    block_swap_mask,
    common_sample_ids,
    compute_metric,
    holm_adjust,
    load_prediction_artifact,
    paired_best_lag_test,
    paired_lag_test,
)


def _write_artifact(
    result_dir: Path,
    lag_predictions: dict[int, np.ndarray],
    *,
    targets: np.ndarray | None = None,
    ids_by_lag: dict[int, list[str]] | None = None,
    task_name: str = "test_task",
) -> None:
    result_dir.mkdir(parents=True)
    sample_count = len(next(iter(lag_predictions.values())))
    if targets is None:
        targets = np.arange(sample_count, dtype=np.float32)[:, None]
    with h5py.File(result_dir / "test_predictions.h5", "w") as artifact:
        artifact.attrs["schema_version"] = 1
        artifact.attrs["task_name"] = task_name
        for lag, predictions in lag_predictions.items():
            ids = (
                ids_by_lag[lag]
                if ids_by_lag is not None
                else [f"sample_{index}" for index in range(sample_count)]
            )
            group = artifact.create_group(f"lag_{lag}")
            group.attrs["lag_ms"] = lag
            split = sample_count // 2
            for fold, indices in enumerate(
                (np.arange(split), np.arange(split, sample_count)), start=1
            ):
                fold_group = group.create_group(f"fold_{fold}")
                fold_group.create_dataset(
                    "sample_id",
                    data=np.asarray(ids, dtype=object)[indices],
                    dtype=h5py.string_dtype("utf-8"),
                )
                fold_group.create_dataset(
                    "start", data=np.arange(sample_count, dtype=float)[indices]
                )
                fold_group.create_dataset("prediction", data=predictions[indices])
                fold_group.create_dataset("target", data=targets[indices])


def test_load_and_align_prediction_artifacts_by_sample_id(tmp_path):
    targets = np.arange(6, dtype=np.float32)[:, None]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_artifact(first_dir, {0: targets.copy()}, targets=targets)
    _write_artifact(second_dir, {0: targets[::-1].copy()}, targets=targets)

    first = load_prediction_artifact("first", first_dir)
    second = load_prediction_artifact("second", second_dir)
    ids, aligned_targets, predictions = align_lag_predictions([first, second], 0)

    assert ids.tolist() == [f"sample_{index}" for index in range(6)]
    np.testing.assert_array_equal(aligned_targets, targets)
    np.testing.assert_array_equal(predictions[0], targets)
    np.testing.assert_array_equal(predictions[1], targets[::-1])


def test_alignment_strictly_rejects_different_sample_sets(tmp_path):
    targets = np.arange(4, dtype=np.float32)[:, None]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_artifact(first_dir, {0: targets}, targets=targets)
    _write_artifact(
        second_dir,
        {0: targets},
        targets=targets,
        ids_by_lag={0: ["sample_0", "sample_1", "sample_2", "different"]},
    )

    with pytest.raises(ValueError, match="Sample IDs differ"):
        align_lag_predictions(
            [
                load_prediction_artifact("first", first_dir),
                load_prediction_artifact("second", second_dir),
            ],
            0,
        )


def test_alignment_strictly_rejects_different_targets(tmp_path):
    targets = np.arange(4, dtype=np.float32)[:, None]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_artifact(first_dir, {0: targets}, targets=targets)
    _write_artifact(second_dir, {0: targets}, targets=targets + 1)

    with pytest.raises(ValueError, match="Targets differ"):
        align_lag_predictions(
            [
                load_prediction_artifact("first", first_dir),
                load_prediction_artifact("second", second_dir),
            ],
            0,
        )


def test_significance_metric_must_return_a_finite_scalar():
    values = np.arange(3, dtype=np.float32)

    with pytest.raises(ValueError, match="one scalar"):
        compute_metric(lambda prediction, target: prediction, values, values)
    with pytest.raises(ValueError, match="non-finite"):
        compute_metric(lambda prediction, target: float("nan"), values, values)


def test_best_lag_sample_population_is_intersected_across_lags(tmp_path):
    def lag(ids):
        values = np.arange(len(ids), dtype=np.float32)[:, None]
        return LagPredictions(
            sample_id=np.asarray(ids),
            start=np.arange(len(ids), dtype=float),
            prediction=values,
            target=values,
        )

    first = PredictionArtifact(
        "first",
        tmp_path / "first",
        "task",
        {0: lag(["a", "b", "c"]), 100: lag(["b", "c", "d"])},
    )
    second = PredictionArtifact(
        "second",
        tmp_path / "second",
        "task",
        {0: lag(["a", "b", "c"]), 100: lag(["b", "c", "d"])},
    )

    assert common_sample_ids([first, second], [0, 100]) == {"b", "c"}


def test_block_swap_mask_is_constant_within_event_blocks():
    mask = block_swap_mask(10, 3, np.random.default_rng(4))

    assert len(mask) == 10
    assert all(np.unique(mask[start : start + 3]).size == 1 for start in (0, 3, 6))


def test_best_lag_test_draws_one_shared_mask_per_permutation(monkeypatch):
    targets = {0: np.arange(6, dtype=np.float32)[:, None], 1: np.arange(6, dtype=np.float32)[:, None]}
    predictions_a = {0: targets[0].copy(), 1: targets[1] + 2}
    predictions_b = {0: targets[0] + 3, 1: targets[1] + 1}
    calls = []
    original = significance_utils.block_swap_mask

    def recording_mask(sample_count, block_size, rng):
        calls.append((sample_count, block_size))
        return original(sample_count, block_size, rng)

    monkeypatch.setattr(significance_utils, "block_swap_mask", recording_mask)
    result = paired_best_lag_test(
        predictions_a,
        predictions_b,
        targets,
        mse_metric,
        False,
        block_size=2,
        n_permutations=7,
        rng=np.random.default_rng(8),
    )

    assert result["best_lag_a"] == 0
    assert result["best_lag_b"] == 1
    assert calls == [(6, 2)] * 7


def test_lower_is_better_metric_has_positive_improvement_effect():
    target = np.arange(8, dtype=np.float32)[:, None]
    result = paired_lag_test(
        target,
        target + 2,
        target,
        mse_metric,
        higher_is_better=False,
        block_size=1,
        n_permutations=15,
        rng=np.random.default_rng(3),
    )

    assert result["score_a"] == pytest.approx(0.0)
    assert result["score_b"] == pytest.approx(4.0)
    assert result["effect"] == pytest.approx(4.0)
    assert result["p_value"] >= 1 / 16


def test_permutation_test_is_reproducible_for_fixed_seed():
    target = np.arange(8, dtype=np.float32)[:, None]
    arguments = (target, target + 2, target, mse_metric, False, 2, 15)

    first = paired_lag_test(*arguments, np.random.default_rng(19))
    second = paired_lag_test(*arguments, np.random.default_rng(19))

    assert first == second


def test_holm_adjustment_is_monotone_in_ranked_order():
    assert holm_adjust([0.03, 0.01, 0.04]) == pytest.approx([0.06, 0.03, 0.06])


def test_baseline_mode_writes_one_row_per_result_and_lag(tmp_path):
    targets = np.arange(8, dtype=np.float32)[:, None]
    baseline_dir = tmp_path / "baseline"
    result_dir = tmp_path / "result"
    _write_artifact(
        baseline_dir, {0: targets + 3, 100: targets + 2}, targets=targets
    )
    _write_artifact(result_dir, {0: targets, 100: targets + 0.5}, targets=targets)
    output_dir = tmp_path / "output"
    config_path = tmp_path / "analysis.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "baseline_lags",
                "metric": {"name": "mse", "higher_is_better": False},
                "baseline": {"name": "baseline", "path": str(baseline_dir)},
                "results": [{"name": "model", "path": str(result_dir)}],
                "block_size": 2,
                "n_permutations": 15,
                "random_seed": 7,
                "alpha": 0.05,
                "output_dir": str(output_dir),
            }
        )
    )

    assert run(config_path) == output_dir
    frame = pd.read_csv(output_dir / "p_values.csv")
    saved_config = yaml.safe_load((output_dir / "config.yml").read_text())

    assert frame["lag"].tolist() == [0, 100]
    assert (frame["effect"] > 0).all()
    assert "p_value_holm" in frame
    assert saved_config["multiple_comparisons"] == "holm"


def test_best_lag_mode_reports_winner_and_full_ordered_family(tmp_path):
    targets = np.arange(8, dtype=np.float32)[:, None]
    specs = []
    for name, offset in (("winner", 0.0), ("middle", 1.0), ("low", 2.0)):
        result_dir = tmp_path / name
        _write_artifact(
            result_dir,
            {0: targets + offset, 100: targets + offset + 0.5},
            targets=targets,
        )
        specs.append({"name": name, "path": str(result_dir)})
    output_dir = tmp_path / "best_output"
    config_path = tmp_path / "best.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "best_lag",
                "metric": {"name": "mse", "higher_is_better": False},
                "results": specs,
                "valid_lags": [0, 100, 200],
                "block_size": 1,
                "n_permutations": 7,
                "random_seed": 11,
                "alpha": 0.05,
                "output_dir": str(output_dir),
            }
        )
    )

    run(config_path)
    reported = pd.read_csv(output_dir / "p_values.csv")
    full_family = pd.read_csv(output_dir / "all_pairwise_tests.csv")

    assert reported["result_a"].tolist() == ["winner", "winner"]
    assert set(reported["result_b"]) == {"middle", "low"}
    assert len(full_family) == 6
    assert set(full_family["best_lag_a"]) == {0}
