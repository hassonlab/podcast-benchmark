import numpy as np
import pytest

from volume_level_ridge import align_for_lag, ridge_r2_by_lag


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
