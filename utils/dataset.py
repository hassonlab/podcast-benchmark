import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _apply_preprocessing(data, preprocessing_fns, preprocessor_params):
    """Apply a list of preprocessing functions to data."""
    if not preprocessing_fns:
        return data

    for i, preprocessing_fn in enumerate(preprocessing_fns):
        if preprocessor_params and isinstance(preprocessor_params, list):
            params = preprocessor_params[i] if i < len(preprocessor_params) else None
        else:
            params = preprocessor_params
        data = preprocessing_fn(data, params)

    return data


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
        preprocessing_fns=None,
        preprocessor_params=None,
    ):
        self.window_width = window_width
        self.preprocessing_fns = preprocessing_fns
        self.preprocessor_params = preprocessor_params
        self.task_df = task_df
        self.data_durations = [raw.times[-1] for raw in raws]
        self._raw_subject_channel_counts = [len(raw.ch_names) for raw in raws]
        self.subject_channel_counts = list(self._raw_subject_channel_counts)
        self._sfreqs = [raw.info["sfreq"] for raw in raws]

        if len(set(self._sfreqs)) != 1:
            raise ValueError(
                "RawNeuralDataset requires all raws to share the same sampling rate"
            )

        self.sfreq = self._sfreqs[0]
        self.raw_arrays = [
            np.asarray(raw.get_data(), dtype=np.float32) for raw in raws
        ]

    def get_data_for_lag(
        self, lag: int
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, list[int]]:
        """Return neural data sliced for the given lag, targets, rows, and channel counts.

        Slices each subject's raw array at onset + lag offset for every word.

        Args:
            lag: Lag in milliseconds.

        Returns:
            Tuple of `(neural_tensor, targets_tensor, task_df, subject_channel_counts)`
            where `neural_tensor` has shape `[n_words, n_electrodes, n_window_samples]`.
        """
        lag_offset = int(round((lag / 1000 - self.window_width / 2) * self.sfreq))
        n_window_samples = int(round((self.window_width - 2e-3) * self.sfreq)) + 1

        tmin = lag / 1000 - self.window_width / 2
        tmax = lag / 1000 + self.window_width / 2 - 2e-3

        selected_rows_df = None
        subject_channel_counts = []
        per_raw_onsets = []
        total_channel_count = 0

        for raw_array, data_duration, channel_count in zip(
            self.raw_arrays, self.data_durations, self._raw_subject_channel_counts
        ):
            valid_mask = (self.task_df.start + tmin >= 0) & (
                self.task_df.start + tmax <= data_duration
            )
            task_df_valid = self.task_df[valid_mask].reset_index(drop=True)

            if len(task_df_valid) == 0:
                continue

            onset_samples = (task_df_valid.start.to_numpy() * self.sfreq).astype(int)
            selection = np.flatnonzero(
                ~pd.Series(onset_samples).duplicated().to_numpy()
            )
            selected_rows_this_raw = task_df_valid.iloc[selection]

            if selected_rows_df is None:
                selected_rows_df = selected_rows_this_raw
            else:
                assert selected_rows_df.equals(
                    selected_rows_this_raw
                ), "Selected rows differ across raws"

            per_raw_onsets.append(onset_samples[selection])
            subject_channel_counts.append(channel_count)
            total_channel_count += channel_count

        if not per_raw_onsets or selected_rows_df is None:
            raise ValueError("No valid events found within data time bounds")

        neural = np.empty(
            (len(selected_rows_df), total_channel_count, n_window_samples),
            dtype=np.float32,
        )
        channel_start = 0
        for raw_array, onset_samples, channel_count in zip(
            self.raw_arrays, per_raw_onsets, subject_channel_counts
        ):
            channel_stop = channel_start + channel_count
            for row_idx, onset in enumerate(onset_samples):
                neural[
                    row_idx,
                    channel_start:channel_stop,
                    :,
                ] = raw_array[
                    :,
                    onset + lag_offset : onset + lag_offset + n_window_samples,
                ]
            channel_start = channel_stop

        if self.preprocessing_fns:
            neural = _apply_preprocessing(
                neural, self.preprocessing_fns, self.preprocessor_params
            )

        targets = selected_rows_df.target.to_numpy(copy=True)
        if targets.dtype == object:
            targets = np.stack(targets)
        targets_tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))

        assert neural.shape[0] == len(
            selected_rows_df
        ), "Mismatch between neural data and task_df lengths"

        neural_tensor = torch.from_numpy(np.asarray(neural, dtype=np.float32))
        self.subject_channel_counts = subject_channel_counts

        return neural_tensor, targets_tensor, selected_rows_df, subject_channel_counts


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
