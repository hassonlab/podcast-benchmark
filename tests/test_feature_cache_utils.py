"""Tests for generic feature-cache helpers."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pytest

from core.config import ModelSpec, TaskConfig, TrainingParams
from core.registry import register_model_constructor
from utils.dataset import NeuralDictDataset
from utils.decoding_utils import (
    CachedFeatureModel,
    _build_cached_fold_loaders,
    _normalize_fold_targets,
    extract_features_for_caching,
    train_decoding_model,
)


TOY_CACHE_STATS = {"encode_calls": 0, "instances": []}
TOY_PERSUBJECT_STATS = {"encode_calls": 0, "instances": []}


class CacheableToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)
        self.encode_calls = 0
        self.forward_from_features_calls = 0
        self.forward_calls = 0

    def encode_features(self, x, **kwargs):
        self.encode_calls += 1
        return self.encoder(x)

    def forward_from_features(self, features, **kwargs):
        self.forward_from_features_calls += 1
        return self.head(features)

    def forward(self, x, **kwargs):
        self.forward_calls += 1
        return self.forward_from_features(self.encode_features(x, **kwargs), **kwargs)


def test_extract_features_for_caching_uses_encode_features_not_forward():
    model = CacheableToyModel()
    x = torch.randn(5, 4)
    target = torch.arange(5)
    ds = NeuralDictDataset(x, {"extra": torch.arange(5)}, target)
    loader = DataLoader(ds, batch_size=2)

    features, input_dicts, y = extract_features_for_caching(
        model, loader, torch.device("cpu")
    )

    assert features.shape == (5, 3)
    assert torch.equal(y, target)
    assert len(input_dicts) == 3
    assert model.encode_calls == 3
    assert model.forward_calls == 0


def test_cached_feature_model_uses_forward_from_features():
    model = CacheableToyModel()
    wrapper = CachedFeatureModel(model)
    features = torch.randn(4, 3)

    out = wrapper(features)

    assert out.shape == (4, 2)
    assert model.forward_from_features_calls == 1
    assert model.forward_calls == 0


class TrainableCacheModel(nn.Module):
    def __init__(self, input_dim=4, feature_dim=3, output_dim=1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.head = nn.Linear(feature_dim, output_dim)
        self.output_dim = output_dim
        TOY_CACHE_STATS["instances"].append(self)

    def encode_features(self, x, **kwargs):
        TOY_CACHE_STATS["encode_calls"] += 1
        return self.encoder(x)

    def forward_from_features(self, features, **kwargs):
        return self.head(features).squeeze(-1)

    def forward(self, x, **kwargs):
        return self.forward_from_features(self.encode_features(x, **kwargs), **kwargs)


@register_model_constructor("trainable_cache_test_model")
def build_trainable_cache_test_model(model_params):
    model_params = dict(model_params)
    model_params.pop("feature_cache", None)
    return TrainableCacheModel(**model_params)


class PerSubjectCacheModel(nn.Module):
    def __init__(self, feature_dim=2, output_dim=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.output_dim = output_dim
        TOY_PERSUBJECT_STATS["instances"].append(self)

    def encode_features(self, x, **kwargs):
        TOY_PERSUBJECT_STATS["encode_calls"] += 1
        pooled = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        return pooled * self.scale

    def forward(self, x, **kwargs):
        return self.encode_features(x, **kwargs).mean(dim=-1)


@register_model_constructor("per_subject_cache_test_model")
def build_per_subject_cache_test_model(model_params):
    model_params = dict(model_params)
    model_params.pop("feature_cache", None)
    return PerSubjectCacheModel(**model_params)


def _tiny_training_params(**overrides):
    params = TrainingParams(
        batch_size=2,
        epochs=1,
        n_folds=2,
        losses=["mse"],
        metrics=[],
        early_stopping_metric="mse",
        smaller_is_better=True,
        normalize_targets=False,
        tensorboard_logging=False,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_train_decoding_model_feature_cache_extracts_full_lag_once(tmp_path):
    TOY_CACHE_STATS["encode_calls"] = 0
    TOY_CACHE_STATS["instances"].clear()
    neural_data = torch.randn(6, 4)
    targets = torch.randn(6)
    data_df = pd.DataFrame({"row": range(6)})
    model_spec = ModelSpec(
        constructor_name="trainable_cache_test_model",
        params={"input_dim": 4, "feature_dim": 3, "output_dim": 1},
        feature_cache=True,
    )

    train_decoding_model(
        neural_data,
        targets,
        data_df,
        model_spec,
        "toy_task",
        TaskConfig(),
        lag=0,
        training_params=_tiny_training_params(),
        checkpoint_dir=str(tmp_path),
    )

    assert TOY_CACHE_STATS["encode_calls"] == 3
    assert len(TOY_CACHE_STATS["instances"]) == 3


def test_cached_fold_loaders_slice_features_and_keep_normalized_targets():
    features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    extra_inputs = {"row": torch.arange(6)}
    targets = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    tr_idx = torch.tensor([0, 2, 4])
    va_idx = torch.tensor([1])
    te_idx = torch.tensor([3, 5])
    training_params = _tiny_training_params(normalize_targets=True)
    target_splits = _normalize_fold_targets(
        targets, tr_idx, va_idx, te_idx, training_params
    )

    loaders = _build_cached_fold_loaders(
        features,
        extra_inputs,
        {"train": tr_idx, "val": va_idx, "test": te_idx},
        target_splits,
        training_params,
    )

    val_ds = loaders["val"].dataset
    test_ds = loaders["test"].dataset
    assert torch.equal(val_ds.neural_data, features[va_idx])
    assert torch.equal(test_ds.input_dict["row"], extra_inputs["row"][te_idx])
    expected_val_target = (targets[va_idx] - targets[tr_idx].mean()) / targets[
        tr_idx
    ].std()
    assert torch.allclose(val_ds.target, expected_val_target)


def test_per_subject_concat_uses_centralized_cache_path(monkeypatch, tmp_path):
    TOY_PERSUBJECT_STATS["encode_calls"] = 0
    TOY_PERSUBJECT_STATS["instances"].clear()

    def fail_per_fold_extraction(*args, **kwargs):
        raise AssertionError("per-fold per-subject extraction should not be called")

    monkeypatch.setattr(
        "utils.decoding_utils.extract_per_subject_concat_features",
        fail_per_fold_extraction,
    )
    neural_data = torch.randn(6, 4, 2)
    targets = torch.randn(6)
    data_df = pd.DataFrame({"row": range(6)})
    model_spec = ModelSpec(
        constructor_name="per_subject_cache_test_model",
        params={"feature_dim": 2, "output_dim": 1},
        per_subject_feature_concat=True,
    )

    train_decoding_model(
        neural_data,
        targets,
        data_df,
        model_spec,
        "toy_task",
        TaskConfig(),
        lag=0,
        training_params=_tiny_training_params(),
        checkpoint_dir=str(tmp_path),
        subject_channel_counts=[2, 2],
    )

    assert TOY_PERSUBJECT_STATS["encode_calls"] == 6


def test_lag_level_cache_rejects_fold_specific_checkpoint_template(tmp_path):
    neural_data = torch.randn(6, 4)
    targets = torch.randn(6)
    data_df = pd.DataFrame({"row": range(6)})
    model_spec = ModelSpec(
        constructor_name="trainable_cache_test_model",
        params={"input_dim": 4, "feature_dim": 3, "output_dim": 1},
        feature_cache=True,
        checkpoint_path="checkpoints/lag_{lag}/best_model_fold{fold}.pt",
    )

    with pytest.raises(ValueError, match="fold-specific encoders"):
        train_decoding_model(
            neural_data,
            targets,
            data_df,
            model_spec,
            "toy_task",
            TaskConfig(),
            lag=0,
            training_params=_tiny_training_params(),
            checkpoint_dir=str(tmp_path),
        )
