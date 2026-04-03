import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import mne


class RawNeuralDataset:
    """
    Loads neural data from MNE Raw objects once into a tensor and provides fast
    access to any lag by slicing — no mne.Epochs call per lag.

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
        from utils.data_utils import _apply_preprocessing

        min_lag = min(lags)
        max_lag = max(lags)

        self.tmin_full = min_lag / 1000 - window_width / 2
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
                (task_df.start + self.tmin_full >= 0)
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

        n_full_samples = int(round((tmax_full - self.tmin_full) * sfreq)) + 1
        self.n_full_samples = n_full_samples

        # Extract windows for each raw and concatenate along electrode axis
        datas = []
        for raw in raws:
            raw_array = raw.get_data()  # [n_electrodes, n_time]
            windows = []
            for onset in self.task_df.start:
                start_sample = int(round((onset + self.tmin_full) * sfreq))
                windows.append(raw_array[:, start_sample : start_sample + n_full_samples])
            datas.append(np.stack(windows, axis=0))  # [n_words, n_elec, n_samples]

        # [n_words, n_total_electrodes, n_full_samples]
        self.neural_data = torch.FloatTensor(np.concatenate(datas, axis=1))

    def get_data_for_lag(
        self, lag: int
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """Return neural data sliced for the given lag, targets, and valid task_df.

        Args:
            lag: Lag in milliseconds.

        Returns:
            Tuple of (neural_tensor, targets_tensor, task_df) where neural_tensor has
            shape [n_words, n_electrodes, n_window_samples].
        """
        from utils.data_utils import _apply_preprocessing

        lag_tmin = lag / 1000 - self.window_width / 2
        start_sample = int(round((lag_tmin - self.tmin_full) * self.sfreq))
        n_window_samples = int(round((self.window_width - 2e-3) * self.sfreq)) + 1

        neural_slice = self.neural_data[:, :, start_sample : start_sample + n_window_samples]

        if self.preprocessing_fns:
            neural_np = neural_slice.numpy()
            neural_np = _apply_preprocessing(
                neural_np, self.preprocessing_fns, self.preprocessor_params
            )
            neural_slice = torch.FloatTensor(neural_np)

        return neural_slice, self.targets_tensor, self.task_df


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
