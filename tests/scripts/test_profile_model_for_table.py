from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from core.config import BaseTaskConfig
from core.registry import register_model_constructor, register_task_data_getter
from scripts import profile_model_for_table as profiler


@dataclass
class ProfileToyTaskConfig(BaseTaskConfig):
    pass


def _toy_task_df(_task_config):
    return pd.DataFrame(
        {
            "start": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "target": np.linspace(0.0, 1.0, 6, dtype=np.float32),
        }
    )


register_task_data_getter(
    "profile_toy_task", config_type=ProfileToyTaskConfig
)(_toy_task_df)


class ProfileToyModel(nn.Module):
    def __init__(self, input_dim=6, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        out = self.linear(x.reshape(x.shape[0], -1))
        return out.squeeze(-1)


class ProfileCacheToyModel(nn.Module):
    def __init__(self, input_dim=6, feature_dim=3, output_dim=1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.head = nn.Linear(feature_dim, output_dim)
        self.output_dim = output_dim

    def encode_features(self, x, **kwargs):
        return self.encoder(x.reshape(x.shape[0], -1))

    def forward_from_features(self, features, **kwargs):
        return self.head(features).squeeze(-1)

    def forward(self, x, **kwargs):
        return self.forward_from_features(self.encode_features(x, **kwargs), **kwargs)


class ProfilePerSubjectToyModel(nn.Module):
    def __init__(self, feature_dim=2, output_dim=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.output_dim = output_dim

    def encode_features(self, x, **kwargs):
        pooled = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        return pooled[:, : self.scale.numel()] * self.scale

    def forward(self, x, **kwargs):
        return self.encode_features(x, **kwargs).mean(dim=-1)


@register_model_constructor("profile_toy_model")
def _build_profile_toy_model(model_params):
    params = dict(model_params)
    params.pop("feature_cache", None)
    return ProfileToyModel(**params)


@register_model_constructor("profile_cache_toy_model")
def _build_profile_cache_toy_model(model_params):
    params = dict(model_params)
    params.pop("feature_cache", None)
    return ProfileCacheToyModel(**params)


@register_model_constructor("profile_persubject_toy_model")
def _build_profile_persubject_toy_model(model_params):
    params = dict(model_params)
    params.pop("feature_cache", None)
    return ProfilePerSubjectToyModel(**params)


@pytest.fixture
def toy_raws(monkeypatch):
    sfreq = 1000
    n_samples = 1000
    raws = []
    for prefix in ("A", "B"):
        info = mne.create_info(
            ch_names=[f"{prefix}1", f"{prefix}2"], sfreq=sfreq, ch_types="seeg"
        )
        data = np.random.randn(2, n_samples)
        raws.append(mne.io.RawArray(data, info, verbose=False))
    monkeypatch.setattr(profiler.data_utils, "load_raws", lambda _params: raws)
    return raws


def _write_config(tmp_path, constructor_name, *, feature_cache=False, per_subject=False):
    config_path = tmp_path / f"{constructor_name}.yml"
    params = (
        "    feature_dim: 2\n    output_dim: 1"
        if per_subject
        else (
            "    input_dim: 12\n    feature_dim: 3\n    output_dim: 1"
            if feature_cache
            else "    input_dim: 12\n    output_dim: 1"
        )
    )
    config_path.write_text(
        f"""
model_spec:
  constructor_name: {constructor_name}
  feature_cache: {str(feature_cache).lower()}
  per_subject_feature_concat: {str(per_subject).lower()}
  params:
{params}
task_config:
  task_name: profile_toy_task
  data_params:
    window_width: 0.004
    subject_ids: [1, 2]
training_params:
  batch_size: 2
  epochs: 1
  n_folds: 2
  lag: 0
  losses: [mse]
  metrics: []
  early_stopping_metric: mse
  smaller_is_better: true
  tensorboard_logging: false
trial_name: profile_test
"""
    )
    return config_path


def test_profiler_outputs_csv_and_latex_with_separate_timing_columns(tmp_path, toy_raws):
    config_path = _write_config(tmp_path, "profile_cache_toy_model", feature_cache=True)

    row = profiler.profile_config(
        str(config_path), device_name="cpu", warmup_iters=1, timing_iters=2
    )
    csv_text = profiler.row_to_csv(row)
    latex_text = profiler.row_to_latex(row)

    assert "cache_build_seconds" in csv_text
    assert "finetune_epoch_seconds" in csv_text
    assert "finetune_ms_per_step" in csv_text
    assert "inference_ms_per_sample" in csv_text
    assert "cache\\_build\\_seconds" in latex_text
    assert row["feature_cache"] is True
    assert row["cached_feature_shape"] != "n/a"


def test_profiler_runs_non_cache_and_per_subject_cache_specs(tmp_path, toy_raws):
    non_cache_path = _write_config(tmp_path, "profile_toy_model")
    per_subject_path = _write_config(
        tmp_path, "profile_persubject_toy_model", per_subject=True
    )

    non_cache_row = profiler.profile_config(
        str(non_cache_path), device_name="cpu", warmup_iters=0, timing_iters=1
    )
    per_subject_row = profiler.profile_config(
        str(per_subject_path), device_name="cpu", warmup_iters=0, timing_iters=1
    )

    assert non_cache_row["cache_build_seconds"] == "n/a"
    assert non_cache_row["cached_feature_shape"] == "n/a"
    assert per_subject_row["per_subject_feature_concat"] is True
    assert per_subject_row["num_subjects"] == 2
    assert per_subject_row["cached_feature_shape"] != "n/a"
    assert per_subject_row["total_params"] == 7
    assert per_subject_row["trainable_params"] == 5
    assert per_subject_row["frozen_params"] == 2


def test_profiler_uses_first_task_for_multi_task_configs(tmp_path, toy_raws):
    config_path = tmp_path / "multi.yml"
    config_path.write_text(
        """
tasks:
  - model_spec:
      constructor_name: profile_toy_model
      params:
        input_dim: 12
        output_dim: 1
    task_config:
      task_name: profile_toy_task
      data_params:
        window_width: 0.004
        subject_ids: [1, 2]
    training_params:
      batch_size: 2
      n_folds: 2
      lag: 0
      losses: [mse]
      metrics: []
      early_stopping_metric: mse
      smaller_is_better: true
    trial_name: first_task
  - model_spec:
      constructor_name: profile_cache_toy_model
      params:
        input_dim: 12
        feature_dim: 3
        output_dim: 1
    task_config:
      task_name: profile_toy_task
      data_params:
        window_width: 0.004
        subject_ids: [1, 2]
    trial_name: second_task
"""
    )

    row = profiler.profile_config(
        str(config_path), device_name="cpu", warmup_iters=0, timing_iters=1
    )

    assert row["trial_name"] == "first_task"
    assert row["constructor_name"] == "profile_toy_model"


def test_profiler_rejects_legacy_flat_configs_without_model_spec(tmp_path, toy_raws):
    config_path = tmp_path / "legacy.yml"
    config_path.write_text(
        """
task_config:
  task_name: profile_toy_task
  data_params:
    window_width: 0.004
    subject_ids: [1, 2]
trial_name: legacy
"""
    )

    with pytest.raises(ValueError, match="model_spec.constructor_name"):
        profiler.profile_config(str(config_path), device_name="cpu")
