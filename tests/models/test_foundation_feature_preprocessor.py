import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.config import ModelSpec
from core.registry import register_model_constructor
from models.foundation_feature_preprocessor import foundation_feature_cache
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
                "xyz_id": kwargs.get("xyz_id", None).detach().cpu().clone()
                if torch.is_tensor(kwargs.get("xyz_id", None))
                else None,
                "lip_coords": kwargs.get("lip_coords", None).detach().cpu().clone()
                if torch.is_tensor(kwargs.get("lip_coords", None))
                else None,
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
