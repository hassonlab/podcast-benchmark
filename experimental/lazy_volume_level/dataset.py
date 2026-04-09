from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class LazyWindowDataset(Dataset):
    """Dataset that stores only timestamps, extra inputs, and targets.

    Neural windows are materialized later in the collate function so the full
    preprocessed dataset never has to exist in memory at once.
    """

    def __init__(
        self,
        starts: np.ndarray,
        input_dict: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> None:
        self.starts = np.asarray(starts, dtype=np.float32)
        self.input_dict = dict(input_dict)
        self.target = target

        lengths = [len(v) for v in self.input_dict.values()]
        if not all(length == len(self.starts) for length in lengths):
            raise ValueError("All extra input tensors must match the number of starts.")
        if len(self.target) != len(self.starts):
            raise ValueError("Target tensor must match the number of starts.")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        item_dict = {key: value[idx] for key, value in self.input_dict.items()}
        return float(self.starts[idx]), item_dict, self.target[idx]
