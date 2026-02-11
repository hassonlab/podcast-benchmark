import pytest

import numpy as np

from models.shared_preprocessors import (
    window_rms_preprocessor,
    log_transform_preprocessor,
    zscore_preprocessor,
)


class TestWindowRMSPreprocessor:
    """Validate the RMS neural window preprocessor."""

    def test_window_rms_outputs_expected_values(self):
        data = np.array(
            [
                [[0.0, 3.0, 4.0, 0.0], [1.0, -1.0, 1.0, -1.0]],
                [[2.0, 2.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        rms = window_rms_preprocessor(data)

        expected = np.array(
            [
                [2.5, 1.0],
                [np.sqrt(2.5), 0.0],
            ],
            dtype=np.float32,
        )

        assert rms.shape == (2, 2)
        assert rms.dtype == np.float32
        np.testing.assert_allclose(rms, expected, rtol=1e-6)

    def test_window_rms_rejects_invalid_shape(self):
        bad_input = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="expects data with shape"):
            window_rms_preprocessor(bad_input)


class TestLogTransformPreprocessor:
    """Ensure the log-transform preprocessor matches expected behaviour."""

    def test_log_transform_matches_numpy(self):
        data = np.array(
            [[0.0, 1.0, 10.0], [5.0, 50.0, 500.0]],
            dtype=np.float32,
        )[None, :, :]

        params = {"epsilon": 1e-3, "scale": 20.0}
        transformed = log_transform_preprocessor(data, params)

        expected = 20.0 * np.log10(np.clip(data, 0.0, None) + 1e-3)

        np.testing.assert_allclose(
            transformed, expected.astype(np.float32), rtol=1e-5, atol=5e-6
        )
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float32

    def test_log_transform_supports_natural_log(self):
        data = np.array([[0.1, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        params = {"log_base": "e", "epsilon": 1e-4}

        transformed = log_transform_preprocessor(data, params)
        expected = np.log(np.clip(data, 0.0, None) + 1e-4)

        np.testing.assert_allclose(
            transformed, expected.astype(np.float32), rtol=1e-5, atol=1e-7
        )

    def test_log_transform_rejects_nonpositive_epsilon(self):
        data = np.ones((1, 2, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            log_transform_preprocessor(data, {"epsilon": 0.0})


class TestZScorePreprocessor:
    """Validate channel-wise z-scoring."""

    def test_zscore_computes_channel_stats(self):
        data = np.array(
            [
                [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]],
                [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
            ],
            dtype=np.float32,
        )

        params: dict = {}
        zscored = zscore_preprocessor(data, params)

        channel0 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        channel1 = np.array([2, 2, 2, 1, 2, 3], dtype=np.float64)

        expected = np.empty_like(zscored)
        expected[:, 0, :] = (data[:, 0, :] - channel0.mean()) / channel0.std(ddof=0)
        expected[:, 1, :] = (data[:, 1, :] - channel1.mean()) / channel1.std(ddof=0)

        np.testing.assert_allclose(zscored, expected.astype(np.float32), rtol=1e-6)
        assert "channel_means" in params and "channel_stds" in params
        assert params["channel_means"].shape == (2,)

    def test_zscore_reuses_provided_stats(self):
        data = np.array(
            [
                [[1.0, 2.0], [10.0, 10.0]],
                [[3.0, 4.0], [10.0, 10.0]],
            ],
            dtype=np.float32,
        )

        means = np.array([2.5, 10.0], dtype=np.float32)
        stds = np.array([1.118034, 1.0], dtype=np.float32)

        params = {"channel_means": means, "channel_stds": stds}
        zscored = zscore_preprocessor(data, params)

        expected = np.empty_like(zscored)
        expected[:, 0, :] = (data[:, 0, :] - means[0]) / stds[0]
        expected[:, 1, :] = (data[:, 1, :] - means[1]) / stds[1]

        np.testing.assert_allclose(zscored, expected.astype(np.float32), rtol=1e-6)

    def test_zscore_raises_on_bad_stats(self):
        data = np.ones((1, 2, 3), dtype=np.float32)
        params = {"channel_means": np.array([0.0]), "channel_stds": np.array([1.0])}

        with pytest.raises(ValueError):
            zscore_preprocessor(data, params)
