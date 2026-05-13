import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core import registry
from core.config import DataParams, ExperimentConfig, ModelSpec, TaskConfig
from core.registry import register_model_constructor
from models.foundation_feature_preprocessor import (
    foundation_feature_cache,
    set_foundation_feature_cache_config,
)
from utils.dataset import _apply_preprocessing

FAKE_CACHE_STATS = {"instances": []}


class FakeFoundationModel(nn.Module):
    def __init__(self, feature_scale=1.0):
        super().__init__()
        self.feature_scale = feature_scale
        self.calls = []
        FAKE_CACHE_STATS["instances"].append(self)

    def encode_features(self, x, **kwargs):
        self.calls.append(
            {
                "x": x.detach().cpu().clone(),
                "xyz_id": (
                    kwargs.get("xyz_id", None).detach().cpu().clone()
                    if torch.is_tensor(kwargs.get("xyz_id", None))
                    else None
                ),
                "lip_coords": (
                    kwargs.get("lip_coords", None).detach().cpu().clone()
                    if torch.is_tensor(kwargs.get("lip_coords", None))
                    else None
                ),
            }
        )
        pooled = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        return pooled * self.feature_scale


@register_model_constructor("fake_foundation_cache_model")
def build_fake_foundation_cache_model(model_params):
    return FakeFoundationModel(model_params.get("feature_scale", 1.0))


def test_foundation_feature_cache_normal_mode_uses_encode_features():
    FAKE_CACHE_STATS["instances"].clear()
    data = np.arange(24, dtype=np.float32).reshape(3, 4, 2)

    features = foundation_feature_cache(
        data,
        {
            "foundation_model_spec": ModelSpec(
                constructor_name="fake_foundation_cache_model",
                params={"feature_scale": 2.0},
            ),
            "batch_size": 2,
        },
    )

    expected = data.mean(axis=-1) * 2.0
    np.testing.assert_allclose(features, expected)
    assert [call["x"].shape for call in FAKE_CACHE_STATS["instances"][0].calls] == [
        torch.Size([2, 4, 2]),
        torch.Size([1, 4, 2]),
    ]


def test_foundation_feature_cache_per_subject_concat_splits_coordinates_in_order():
    FAKE_CACHE_STATS["instances"].clear()
    data = np.arange(40, dtype=np.float32).reshape(2, 5, 4)
    xyz = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
    lip = np.arange(30, dtype=np.int64).reshape(2, 5, 3) + 100
    rows = pd.DataFrame(
        {
            "xyz_id": [xyz[0], xyz[1]],
            "lip_coords": [lip[0], lip[1]],
        }
    )

    features = foundation_feature_cache(
        data,
        {
            "foundation_model_spec": {
                "constructor_name": "fake_foundation_cache_model",
                "params": {},
            },
            "mode": "per_subject_feature_concat",
            "batch_size": 2,
        },
        selected_rows=rows,
        subject_channel_counts=[2, 3],
    )

    expected = np.concatenate(
        [data[:, :2].mean(axis=-1), data[:, 2:].mean(axis=-1)], axis=-1
    )
    np.testing.assert_allclose(features, expected)

    calls = FAKE_CACHE_STATS["instances"][0].calls
    assert [call["x"].shape for call in calls] == [
        torch.Size([2, 2, 4]),
        torch.Size([2, 3, 4]),
    ]
    np.testing.assert_array_equal(calls[0]["xyz_id"].numpy(), xyz[:, :2])
    np.testing.assert_array_equal(calls[1]["xyz_id"].numpy(), xyz[:, 2:])
    np.testing.assert_array_equal(calls[0]["lip_coords"].numpy(), lip[:, :2])
    np.testing.assert_array_equal(calls[1]["lip_coords"].numpy(), lip[:, 2:])


def test_legacy_two_argument_preprocessors_still_work_with_context():
    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    def legacy_preprocessor(x, params):
        return x + params["offset"]

    result = _apply_preprocessing(
        data,
        [legacy_preprocessor],
        [{"offset": 3}],
        selected_rows=pd.DataFrame({"unused": [1]}),
        subject_channel_counts=[2],
    )

    np.testing.assert_array_equal(result, data + 3)


def test_foundation_feature_cache_config_setter_updates_direct_preprocessor(
    monkeypatch,
):
    def fake_foundation_setter(config, raws, task_df):
        assert config.model_spec.feature_cache is True
        assert config.model_spec.params["feature_cache"] is True
        config.model_spec.params["output_dim"] = 7
        config.model_spec.params["embedding_dim"] = 7
        config.model_spec.params["output_activation"] = "sigmoid"
        config.model_spec.params["dropout"] = 0.2
        return config

    monkeypatch.setitem(
        registry.config_setter_registry,
        "fake_foundation_config_setter",
        fake_foundation_setter,
    )

    config = ExperimentConfig(
        model_spec=ModelSpec(
            constructor_name="probe_model",
            params={},
        ),
        task_config=TaskConfig(
            task_name="fake_task",
            data_params=DataParams(
                data_root="data",
                preprocessing_fn_name=[
                    "stft_preprocessing",
                    "foundation_feature_cache",
                ],
                preprocessor_params=[
                    {
                        "freq_channel_cutoff": 40,
                    },
                    {
                        "mode": "normal",
                        "foundation_config_setter_name": "fake_foundation_config_setter",
                        "foundation_model_spec": {
                            "constructor_name": "fake_foundation_cache_model",
                            "params": {},
                        },
                    }
                ],
            ),
        ),
    )

    result = set_foundation_feature_cache_config(config, raws=[], task_df=None)
    stft_params = result.task_config.data_params.preprocessor_params[0]
    foundation_params = result.task_config.data_params.preprocessor_params[1]

    assert result.task_config.data_params.preprocessing_fn_name == [
        "stft_preprocessing",
        "foundation_feature_cache",
    ]
    assert stft_params["freq_channel_cutoff"] == 40
    nested_spec = foundation_params["foundation_model_spec"]
    assert nested_spec.feature_cache is True
    assert nested_spec.params["feature_cache"] is True
    assert "output_dim" not in nested_spec.params
    assert "embedding_dim" not in nested_spec.params
    assert "output_activation" not in nested_spec.params
    assert "dropout" not in nested_spec.params
    assert foundation_params["input_fields"] == []
    assert "output_dim" not in result.model_spec.params
    assert "embedding_dim" not in result.model_spec.params


def test_foundation_feature_cache_config_setter_allows_missing_nested_output_dim(
    monkeypatch,
):
    def fake_foundation_setter(config, raws, task_df):
        assert config.model_spec.params.get("output_dim") is None
        config.model_spec.params["input_channels"] = 40
        return config

    monkeypatch.setitem(
        registry.config_setter_registry,
        "fake_cache_only_config_setter",
        fake_foundation_setter,
    )

    config = ExperimentConfig(
        model_spec=ModelSpec(constructor_name="probe_model", params={"output_dim": 1}),
        task_config=TaskConfig(
            task_name="fake_task",
            data_params=DataParams(
                preprocessing_fn_name=["foundation_feature_cache"],
                preprocessor_params=[
                    {
                        "mode": "normal",
                        "foundation_config_setter_name": "fake_cache_only_config_setter",
                        "foundation_model_spec": {
                            "constructor_name": "fake_foundation_cache_model",
                            "params": {"output_dim": None},
                        },
                    }
                ],
            ),
        ),
    )

    result = set_foundation_feature_cache_config(config, raws=[], task_df=None)
    foundation_params = result.task_config.data_params.preprocessor_params[0]
    nested_spec = foundation_params["foundation_model_spec"]

    assert nested_spec.feature_cache is True
    assert nested_spec.params == {"feature_cache": True, "input_channels": 40}
    assert result.model_spec.params["output_dim"] == 1
