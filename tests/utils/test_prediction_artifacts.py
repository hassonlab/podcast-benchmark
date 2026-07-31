import numpy as np
import pytest
import torch

from core.config import DataParams, ModelSpec, TrainingParams
from utils.decoding_utils import (
    _maybe_shuffle_training_targets,
    _preprocessor_params_for_null_seed,
    _validate_null_repetitions,
)
from utils.fold_utils import get_zero_shot_folds


def test_shuffle_targets_changes_training_split_only_and_is_reproducible():
    splits = {
        "train": torch.arange(20, dtype=torch.float32),
        "val": torch.tensor([20.0, 21.0]),
        "test": torch.tensor([22.0, 23.0]),
    }
    params = TrainingParams(shuffle_targets=True, random_seed=7)

    first = _maybe_shuffle_training_targets(splits, params, fold=2)
    second = _maybe_shuffle_training_targets(splits, params, fold=2)

    assert torch.equal(first["train"], second["train"])
    assert not torch.equal(first["train"], splits["train"])
    assert torch.equal(first["train"].sort().values, splits["train"])
    assert torch.equal(first["val"], splits["val"])
    assert torch.equal(first["test"], splits["test"])


def test_shuffle_targets_changes_between_null_repetition_seeds():
    splits = {
        "train": torch.arange(20, dtype=torch.float32),
        "val": torch.tensor([20.0]),
        "test": torch.tensor([21.0]),
    }
    first = _maybe_shuffle_training_targets(
        splits, TrainingParams(shuffle_targets=True, random_seed=7), fold=1
    )
    second = _maybe_shuffle_training_targets(
        splits, TrainingParams(shuffle_targets=True, random_seed=8), fold=1
    )
    assert not torch.equal(first["train"], second["train"])


def test_null_repetitions_allow_combined_nested_random_init_and_shuffle():
    params = TrainingParams(shuffle_targets=True, num_null_repetitions=3)
    data_params = DataParams(
        preprocessor_params={
            "foundation_model_spec": {
                "constructor_name": "encoder",
                "random_init": True,
            }
        }
    )
    assert _validate_null_repetitions(
        [0], ModelSpec(constructor_name="probe"), params, data_params
    ) == (3, True)

    first_params = _preprocessor_params_for_null_seed(
        data_params.preprocessor_params, 42
    )
    second_params = _preprocessor_params_for_null_seed(
        data_params.preprocessor_params, 43
    )
    assert first_params["_null_repetition_seed"] == 42
    assert second_params["_null_repetition_seed"] == 43


def test_null_repetitions_require_one_lag_and_a_control():
    model_spec = ModelSpec(constructor_name="probe")
    data_params = DataParams()
    with pytest.raises(ValueError, match="exactly one"):
        _validate_null_repetitions(
            [0, 100],
            model_spec,
            TrainingParams(shuffle_targets=True, num_null_repetitions=2),
            data_params,
        )
    with pytest.raises(ValueError, match="requires shuffle_targets"):
        _validate_null_repetitions(
            [0],
            model_spec,
            TrainingParams(num_null_repetitions=2),
            data_params,
        )


def test_zero_shot_folds_use_stable_first_occurrence_order():
    words = np.array(
        [
            "zebra",
            "apple",
            "zebra",
            "moon",
            "apple",
            "kite",
            "river",
            "cloud",
            "stone",
            "field",
        ]
    )

    first = get_zero_shot_folds(words, num_folds=2)
    second = get_zero_shot_folds(words.copy(), num_folds=2)

    for first_split, second_split in zip(first, second):
        for first_indices, second_indices in zip(first_split, second_split):
            assert np.array_equal(first_indices, second_indices)
