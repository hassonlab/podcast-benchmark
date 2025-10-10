import numpy as np
import pandas as pd
import pytest

from config import DataParams, ExperimentConfig, TrainingParams
from volume_level_ridge import align_for_lag, ridge_r2_by_lag, run_volume_level_ridge_from_config


def test_align_for_lag_positive_lag():
    sr = 100.0
    audio = np.arange(10, dtype=float)
    neural = np.vstack([np.arange(10, dtype=float), np.arange(10, dtype=float) ** 2])

    lag_ms = 20.0  # 2 samples at 100 Hz
    X, y = align_for_lag(audio, neural, lag_ms, sr)

    expected_X = neural[:, 2:].T
    expected_y = audio[:-2]

    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)


def test_align_for_lag_negative_lag():
    sr = 200.0
    audio = np.arange(8, dtype=float)
    neural = np.vstack([np.linspace(0.0, 1.0, 8), np.linspace(1.0, 0.0, 8)])

    lag_ms = -5.0  # -1 sample at 200 Hz
    X, y = align_for_lag(audio, neural, lag_ms, sr)

    expected_X = neural[:, :-1].T
    expected_y = audio[1:]

    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)


def test_align_for_lag_raises_on_large_shift():
    sr = 100.0
    audio = np.arange(5, dtype=float)
    neural = np.vstack([np.arange(5, dtype=float)])

    with pytest.raises(ValueError):
        align_for_lag(audio, neural, 100.0, sr)


def test_ridge_r2_by_lag_identifies_true_lag():
    rng = np.random.default_rng(42)
    samples = 250
    n_features = 6
    sr = 100.0
    shift_samples = 3
    lag_true_ms = shift_samples * 1000.0 / sr

    neural = rng.standard_normal((n_features, samples)).astype(np.float32)
    weights = rng.standard_normal(n_features).astype(np.float32)

    audio = np.zeros(samples, dtype=np.float32)
    aligned_neural = neural[:, shift_samples:].T @ weights
    audio[: aligned_neural.shape[0]] = aligned_neural
    audio += 0.05 * rng.standard_normal(samples).astype(np.float32)

    lags = np.arange(-40.0, 41.0, 10.0, dtype=float)
    results = ridge_r2_by_lag(
        audio,
        neural,
        sr,
        lags,
        alphas=np.logspace(-3, 1, 5),
        cv_splits=5,
        device="cpu",
        random_state=0,
        verbose=False,
    )

    assert results["lag_ms"].size > 0
    best_idx = int(np.argmax(results["r2"]))
    best_lag = results["lag_ms"][best_idx]

    assert pytest.approx(best_lag, rel=0.1, abs=5.0) == lag_true_ms

    for key in ["r2", "alpha", "coef_norm", "n_samples", "n_features", "train_r2"]:
        assert results[key].shape == results["lag_ms"].shape


class _DummyRaw:
    def __init__(self, data: np.ndarray, sfreq: float):
        self._data = np.asarray(data, dtype=np.float32)
        self.info = {"sfreq": float(sfreq)}

    def get_data(self) -> np.ndarray:
        return self._data

    def copy(self) -> "_DummyRaw":
        return _DummyRaw(self._data.copy(), self.info["sfreq"])

    def resample(self, _target_sr: float) -> None:  # pragma: no cover - not used in tests
        raise NotImplementedError


def test_run_volume_level_ridge_all_modes(tmp_path):
    sr = 100.0
    samples = 120
    time = np.arange(samples) / sr

    data_subject_1 = np.vstack(
        [np.sin(2 * np.pi * 0.5 * time), np.cos(2 * np.pi * 0.25 * time)]
    ).astype(np.float32)
    data_subject_2 = np.vstack(
        [np.cos(2 * np.pi * 0.5 * time), np.sin(2 * np.pi * 0.25 * time)]
    ).astype(np.float32)

    raws = [_DummyRaw(data_subject_1, sr), _DummyRaw(data_subject_2, sr)]

    audio = data_subject_1[0] + data_subject_2[0]
    df_targets = pd.DataFrame({
        "start": time,
        "target": audio.astype(np.float32),
    })

    config = ExperimentConfig(
        model_constructor_name="volume_level_ridge",
        task_name="volume_level_encoding_task",
        trial_name="test",
        output_dir=str(tmp_path),
        model_params={
            "analysis_modes": ["pooled_electrodes", "per_subject", "average"],
            "cv_splits": 2,
            "plot": False,
            "allow_neural_resample": False,
        },
        training_params=TrainingParams(
            min_lag=0,
            max_lag=10,
            lag_step_size=10,
            lag=None,
            n_folds=1,
        ),
        data_params=DataParams(
            subject_ids=[1, 2],
            task_params={"target_sr": sr},
        ),
    )

    results = run_volume_level_ridge_from_config(
        config,
        raws,
        df_targets,
        str(tmp_path),
    )

    assert set(results["analysis_modes"]) == {
        "average",
        "per_subject",
        "pooled_electrodes",
    }
    assert "pooled_electrodes" in results
    assert "per_subject" in results
    assert set(results["per_subject"].keys()) == {1, 2}

    average_block = results["average"]
    assert "curve" in average_block and "per_subject" in average_block
    assert not average_block["curve"].empty
    assert {"lag_ms", "r2_mean"}.issubset(set(average_block["curve"].columns))
