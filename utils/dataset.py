import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RawNeuralDataset:
    """
    Stores raw electrode arrays and word onset indices to provide fast lag-based
    slicing without mne.Epochs calls and without redundant per-word storage.

    Rather than pre-extracting a wide window per word (which wastes RAM when words
    are densely packed in time), we store each subject's full raw array and compute
    lag windows on the fly by indexing into it.

    Args:
        raws: List of preloaded MNE Raw objects (one per subject).
        task_df: DataFrame with at least 'start' (onset in seconds) and 'target' columns.
        window_width: Width of the analysis window in seconds.
        lags: List of lags in milliseconds to support.
        preprocessing_fns: Optional list of preprocessing functions applied after slicing.
        preprocessor_params: Parameters forwarded to preprocessing functions.
    """

    def __init__(
        self,
        raws: list,
        task_df: pd.DataFrame,
        window_width: float,
        lags: list[int],
        preprocessing_fns=None,
        preprocessor_params=None,
    ):
        min_lag = min(lags)
        max_lag = max(lags)

        tmin_full = min_lag / 1000 - window_width / 2
        tmax_full = max_lag / 1000 + window_width / 2 - 2e-3
        self.window_width = window_width
        self.preprocessing_fns = preprocessing_fns
        self.preprocessor_params = preprocessor_params

        # Single valid_mask covering all raws and all lags
        valid_mask = pd.Series(True, index=task_df.index)
        sfreq = None
        for raw in raws:
            sfreq = raw.info["sfreq"]
            data_duration = raw.times[-1]
            valid_mask = valid_mask & (
                (task_df.start + tmin_full >= 0)
                & (task_df.start + tmax_full <= data_duration)
            )

        if not valid_mask.any():
            raise ValueError("No valid events found within data time bounds for all lags")

        self.sfreq = sfreq
        self.task_df = task_df[valid_mask].reset_index(drop=True)

        targets = self.task_df.target.to_numpy()
        if targets.dtype == object:
            targets = np.stack(targets)
        self.targets_tensor = torch.FloatTensor(targets)

        # Precompute onset sample indices (same for all raws)
        self.onset_samples = np.array(
            [int(round(onset * sfreq)) for onset in self.task_df.start]
        )

        # Store the full raw arrays — one per subject
        self.raw_arrays = [raw.get_data() for raw in raws]

    def get_data_for_lag(
        self, lag: int
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """Return neural data sliced for the given lag, targets, and valid task_df.

        Slices each subject's raw array at onset + lag offset for every word.

        Args:
            lag: Lag in milliseconds.

        Returns:
            Tuple of (neural_tensor, targets_tensor, task_df) where neural_tensor has
            shape [n_words, n_electrodes, n_window_samples].
        """
        from utils.data_utils import _apply_preprocessing

        lag_offset = int(round((lag / 1000 - self.window_width / 2) * self.sfreq))
        n_window_samples = int(round((self.window_width - 2e-3) * self.sfreq)) + 1

        windows_per_raw = []
        for raw_array in self.raw_arrays:
            windows = np.stack([
                raw_array[:, onset + lag_offset : onset + lag_offset + n_window_samples]
                for onset in self.onset_samples
            ])  # [n_words, n_elec, n_window_samples]
            windows_per_raw.append(windows)

        neural = np.concatenate(windows_per_raw, axis=1)  # [n_words, n_total_elec, n_window_samples]

        if self.preprocessing_fns:
            neural = _apply_preprocessing(neural, self.preprocessing_fns, self.preprocessor_params)

        return torch.FloatTensor(neural), self.targets_tensor, self.task_df


class NeuralDictDataset(Dataset):
    """
    A PyTorch Dataset that takes neural data, a dictionary of tensors as input, and a target tensor.

    Args:
        neural_data: Tensor containing neural data inputs.
        input_dict: Dictionary where keys are strings and values are tensors.
                   All tensors must have the same length in dimension 0.
        target: Target tensor with the same length as input tensors in dimension 0.
    """

    def __init__(self, neural_data, input_dict, target):
        self.neural_data = neural_data
        self.input_dict = input_dict
        self.target = target

        # Validate that all tensors have the same length
        lengths = [len(v) for v in input_dict.values()]
        if not all(length == len(target) for length in lengths):
            raise ValueError(
                "All input tensors and target must have the same length in dimension 0"
            )

        self.length = len(target)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a tuple: (dict of ith indexed tensors, ith target)
        item_dict = {key: value[idx] for key, value in self.input_dict.items()}
        return self.neural_data[idx], item_dict, self.target[idx]
