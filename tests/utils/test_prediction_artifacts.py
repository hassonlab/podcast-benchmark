import numpy as np
import torch

from core.config import TrainingParams
from utils.decoding_utils import _maybe_shuffle_training_targets
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
