import pytest

import numpy as np

import core.registry as registry
from core.config import ModelSpec
from models.shared_preprocessors import (
    _build_cache_identity,
    disk_cache_preprocessor,
    window_rms_preprocessor,
    log_transform_preprocessor,
    zscore_preprocessor,
)


def _cache_context(starts=None, channel_names=None, counts=None, lag=0):
    import pandas as pd

    return {
        "selected_rows": pd.DataFrame({"start": starts or [1.0, 2.0, 3.0]}),
        "selected_rows_df": pd.DataFrame({"start": starts or [1.0, 2.0, 3.0]}),
        "lag": lag,
        "subject_channel_names": channel_names or [["A1", "A2"], ["B1"]],
        "subject_channel_counts": counts or [2, 1],
    }


class TestDiskCachePreprocessor:
    def test_single_wrapped_preprocessor_cache_miss_then_hit(
        self, tmp_path, monkeypatch
    ):
        calls = {"count": 0}

        def add_offset(data, params):
            calls["count"] += 1
            return data + params["offset"]

        monkeypatch.setitem(
            registry.data_preprocessor_registry, "cache_test_add_offset", add_offset
        )
        data = np.ones((3, 3, 2), dtype=np.float32)
        params = {
            "base_preprocessing_fn_name": "cache_test_add_offset",
            "base_preprocessor_params": {"offset": 2.0},
            "cache_dir": str(tmp_path),
        }

        first = disk_cache_preprocessor(data, params, **_cache_context())
        second = disk_cache_preprocessor(data, params, **_cache_context())

        np.testing.assert_allclose(first, data + 2.0)
        np.testing.assert_allclose(second, first)
        assert calls["count"] == 1

    def test_wrapped_preprocessor_chain_cache_miss_then_hit(
        self, tmp_path, monkeypatch
    ):
        calls = {"add": 0, "scale": 0}

        def add_offset(data, params):
            calls["add"] += 1
            return data + params["offset"]

        def scale(data, params):
            calls["scale"] += 1
            return data * params["factor"]

        monkeypatch.setitem(
            registry.data_preprocessor_registry, "cache_test_chain_add", add_offset
        )
        monkeypatch.setitem(
            registry.data_preprocessor_registry, "cache_test_chain_scale", scale
        )
        data = np.ones((3, 3, 2), dtype=np.float32)
        params = {
            "base_preprocessing_fn_name": [
                "cache_test_chain_add",
                "cache_test_chain_scale",
            ],
            "base_preprocessor_params": [{"offset": 1.0}, {"factor": 4.0}],
            "cache_dir": str(tmp_path),
        }

        first = disk_cache_preprocessor(data, params, **_cache_context())
        second = disk_cache_preprocessor(data, params, **_cache_context())

        np.testing.assert_allclose(first, (data + 1.0) * 4.0)
        np.testing.assert_allclose(second, first)
        assert calls == {"add": 1, "scale": 1}

    def test_second_call_does_not_execute_any_wrapped_preprocessors_on_hit(
        self, tmp_path, monkeypatch
    ):
        calls = {"count": 0}

        def fail_after_first(data, params):
            calls["count"] += 1
            if calls["count"] > 1:
                raise AssertionError("cache hit executed wrapped preprocessor")
            return data + 3.0

        monkeypatch.setitem(
            registry.data_preprocessor_registry,
            "cache_test_fail_after_first",
            fail_after_first,
        )
        data = np.ones((3, 3, 2), dtype=np.float32)
        params = {
            "base_preprocessing_fn_name": "cache_test_fail_after_first",
            "cache_dir": str(tmp_path),
        }

        disk_cache_preprocessor(data, params, **_cache_context())
        result = disk_cache_preprocessor(data, params, **_cache_context())

        np.testing.assert_allclose(result, data + 3.0)
        assert calls["count"] == 1

    @pytest.mark.parametrize(
        "first_context,second_context",
        [
            (_cache_context(lag=0), _cache_context(lag=100)),
            (
                _cache_context(starts=[1.0, 2.0, 3.0]),
                _cache_context(starts=[1.0, 2.5, 3.0]),
            ),
            (
                _cache_context(channel_names=[["A1", "A2"], ["B1"]]),
                _cache_context(channel_names=[["A2", "A1"], ["B1"]]),
            ),
        ],
    )
    def test_context_identity_changes_cause_miss(
        self, tmp_path, monkeypatch, first_context, second_context
    ):
        calls = {"count": 0}

        def add_call_count(data, params):
            calls["count"] += 1
            return data + calls["count"]

        monkeypatch.setitem(
            registry.data_preprocessor_registry,
            "cache_test_context_identity",
            add_call_count,
        )
        data = np.ones((3, 3, 2), dtype=np.float32)
        params = {
            "base_preprocessing_fn_name": "cache_test_context_identity",
            "cache_dir": str(tmp_path),
        }

        disk_cache_preprocessor(data, params, **first_context)
        result = disk_cache_preprocessor(data, params, **second_context)

        np.testing.assert_allclose(result, data + 2.0)
        assert calls["count"] == 2

    def test_changing_wrapped_preprocessor_params_causes_miss(
        self, tmp_path, monkeypatch
    ):
        calls = {"count": 0}

        def add_offset(data, params):
            calls["count"] += 1
            return data + params["offset"]

        monkeypatch.setitem(
            registry.data_preprocessor_registry, "cache_test_param_change", add_offset
        )
        data = np.ones((3, 3, 2), dtype=np.float32)

        result_a = disk_cache_preprocessor(
            data,
            {
                "base_preprocessing_fn_name": "cache_test_param_change",
                "base_preprocessor_params": {"offset": 1.0},
                "cache_dir": str(tmp_path),
            },
            **_cache_context(),
        )
        result_b = disk_cache_preprocessor(
            data,
            {
                "base_preprocessing_fn_name": "cache_test_param_change",
                "base_preprocessor_params": {"offset": 2.0},
                "cache_dir": str(tmp_path),
            },
            **_cache_context(),
        )

        np.testing.assert_allclose(result_a, data + 1.0)
        np.testing.assert_allclose(result_b, data + 2.0)
        assert calls["count"] == 2

    def test_changing_wrapped_preprocessor_order_causes_miss(
        self, tmp_path, monkeypatch
    ):
        calls = {"add": 0, "scale": 0}

        def add_one(data, params):
            calls["add"] += 1
            return data + 1.0

        def scale_by_two(data, params):
            calls["scale"] += 1
            return data * 2.0

        monkeypatch.setitem(
            registry.data_preprocessor_registry, "cache_test_add_one", add_one
        )
        monkeypatch.setitem(
            registry.data_preprocessor_registry, "cache_test_scale_two", scale_by_two
        )
        data = np.ones((3, 3, 2), dtype=np.float32)

        first = disk_cache_preprocessor(
            data,
            {
                "base_preprocessing_fn_name": [
                    "cache_test_add_one",
                    "cache_test_scale_two",
                ],
                "base_preprocessor_params": [None, None],
                "cache_dir": str(tmp_path),
            },
            **_cache_context(),
        )
        second = disk_cache_preprocessor(
            data,
            {
                "base_preprocessing_fn_name": [
                    "cache_test_scale_two",
                    "cache_test_add_one",
                ],
                "base_preprocessor_params": [None, None],
                "cache_dir": str(tmp_path),
            },
            **_cache_context(),
        )

        np.testing.assert_allclose(first, (data + 1.0) * 2.0)
        np.testing.assert_allclose(second, (data * 2.0) + 1.0)
        assert calls == {"add": 2, "scale": 2}

    def test_model_spec_params_normalize_for_deterministic_identity(self):
        spec = ModelSpec(
            constructor_name="fake_foundation",
            params={"feature_cache": True, "model_dim": 128},
            feature_cache=True,
        )

        identity_a = _build_cache_identity(
            names=["foundation_feature_cache"],
            wrapped_fns=[lambda data, params: data],
            wrapped_params=[{"foundation_model_spec": spec}],
            mode="normal",
            context=_cache_context(),
        )
        identity_b = _build_cache_identity(
            names=["foundation_feature_cache"],
            wrapped_fns=[lambda data, params: data],
            wrapped_params=[
                {
                    "foundation_model_spec": {
                        "constructor_name": "fake_foundation",
                        "params": {"model_dim": 128, "feature_cache": True},
                        "feature_cache": True,
                        "per_subject_feature_concat": False,
                        "sub_models": {},
                        "checkpoint_path": None,
                        "model_data_getter": None,
                    }
                }
            ],
            mode="normal",
            context=_cache_context(),
        )

        assert identity_a["pipeline"][0]["params"] == identity_b["pipeline"][0]["params"]

    def test_selected_row_start_identity_uses_millisecond_precision(self):
        identity_a = _build_cache_identity(
            names=["cache_test"],
            wrapped_fns=[lambda data, params: data],
            wrapped_params=[None],
            mode="normal",
            context=_cache_context(starts=[1.2349, 2.0009]),
        )
        identity_b = _build_cache_identity(
            names=["cache_test"],
            wrapped_fns=[lambda data, params: data],
            wrapped_params=[None],
            mode="normal",
            context=_cache_context(starts=[1.2341, 2.0001]),
        )
        identity_c = _build_cache_identity(
            names=["cache_test"],
            wrapped_fns=[lambda data, params: data],
            wrapped_params=[None],
            mode="normal",
            context=_cache_context(starts=[1.2354, 2.0014]),
        )

        assert identity_a["selected_rows_start"] == [1.234, 2.0]
        assert identity_a["selected_rows_start"] == identity_b["selected_rows_start"]
        assert identity_a["selected_rows_start"] != identity_c["selected_rows_start"]

    def test_unknown_wrapped_preprocessor_raises_clear_value_error(self, tmp_path):
        data = np.ones((3, 3, 2), dtype=np.float32)

        with pytest.raises(
            ValueError, match="Unknown wrapped preprocessor 'missing_preprocessor'"
        ):
            disk_cache_preprocessor(
                data,
                {
                    "base_preprocessing_fn_name": "missing_preprocessor",
                    "cache_dir": str(tmp_path),
                },
                **_cache_context(),
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
