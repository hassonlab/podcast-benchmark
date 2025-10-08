import numpy as np
import pytest
import torch

from ridge_utils import (
    SlidingWindowMetadata,
    align_for_lag,
    apply_sliding_window_rms,
    ridge_r2_by_lag,
    rms_sliding_window,
)


class TestSlidingWindowRMS:
    def test_rms_sliding_window_matches_manual(self):
        arr = np.array([[0.0, 3.0, 4.0, 0.0]], dtype=np.float32)
        window = 2
        hop = 1

        rms = rms_sliding_window(arr, window, hop)
        expected = np.array(
            [
                [
                    np.sqrt((0.0**2 + 3.0**2) / 2),
                    np.sqrt((3.0**2 + 4.0**2) / 2),
                    np.sqrt((4.0**2 + 0.0**2) / 2),
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rms, expected, atol=1e-6)

    def test_apply_sliding_window_rms_effective_sr(self):
        sr = 1000.0
        audio = np.ones(2000, dtype=np.float32)
        neural = np.ones((2, 2000), dtype=np.float32)
        window_ms = 50.0
        hop_ms = 25.0

        neural_win, audio_win, effective_sr, metadata = apply_sliding_window_rms(
            audio,
            neural,
            sr,
            window_ms,
            hop_ms,
        )

        window_samples = int(round(window_ms * sr / 1000.0))
        hop_samples = int(round(hop_ms * sr / 1000.0))
        expected_windows = int(np.floor((2000 - window_samples) / hop_samples) + 1)

        assert neural_win.shape == (2, expected_windows)
        assert audio_win.shape == (expected_windows,)
        assert pytest.approx(effective_sr, rel=1e-6) == sr / hop_samples
        assert isinstance(metadata, SlidingWindowMetadata)
        assert metadata.window_samples == window_samples
        assert metadata.hop_samples == hop_samples


class TestAlignForLag:
    def test_align_for_lag_positive_shift(self):
        audio = np.arange(10, dtype=np.float32)
        neural = np.vstack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)])
        sr = 1000.0
        lag_ms = 2.0  # -> 2 samples

        X, y = align_for_lag(audio, neural, lag_ms, sr)

        assert X.shape == (8, 2)
        assert y.shape == (8,)
        np.testing.assert_array_equal(y, audio[:-2])
        np.testing.assert_array_equal(X[:, 0], neural[0, 2:])

    def test_align_for_lag_negative_shift(self):
        audio = np.arange(10, dtype=np.float32)
        neural = np.vstack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)])
        sr = 1000.0
        lag_ms = -2.0

        X, y = align_for_lag(audio, neural, lag_ms, sr)

        assert X.shape == (8, 2)
        assert y.shape == (8,)
        np.testing.assert_array_equal(y, audio[2:])
        np.testing.assert_array_equal(X[:, 0], neural[0, :-2])


class TestRidgeR2ByLag:
    @pytest.mark.parametrize("device", ["cpu"])
    def test_ridge_r2_by_lag_prefers_zero_shift(self, device):
        rng = np.random.default_rng(42)
        sr = 100.0
        duration = 5.0
        n_samples = int(sr * duration)
        t = np.arange(n_samples, dtype=np.float32) / sr
        base = np.sin(2 * np.pi * 3 * t)
        audio = base.astype(np.float32)
        neural = np.vstack(
            [
                base + 0.01 * rng.standard_normal(n_samples),
                0.5 * base + 0.01 * rng.standard_normal(n_samples),
            ]
        ).astype(np.float32)

        lags = [-200.0, -100.0, 0.0, 100.0, 200.0]
        results = ridge_r2_by_lag(
            audio,
            neural,
            sr,
            lags,
            cv_splits=5,
            device=device,
            verbose=False,
        )

        assert results["lag_ms"].shape[0] > 0
        best_idx = int(np.argmax(results["r2"]))
        assert np.isclose(results["lag_ms"][best_idx], 0.0)
        assert results["r2"][best_idx] > 0.9
        assert np.all(results["n_samples"] > 0)
        assert np.all(results["n_features"] > 0)
        assert torch.all(torch.as_tensor(results["alpha"]) > 0)