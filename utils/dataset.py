import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import data_utils


def _preprocessor_accepts_context(preprocessing_fn):
    signature = inspect.signature(preprocessing_fn)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return True
    return any(
        name in signature.parameters
        for name in (
            "selected_rows",
            "selected_rows_df",
            "subject_channel_counts",
            "subject_channel_names",
            "model_inputs",
            "device",
            "lag",
        )
    )


def _preprocessor_context_kwargs(preprocessing_fn, context):
    signature = inspect.signature(preprocessing_fn)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return context
    return {
        name: value for name, value in context.items() if name in signature.parameters
    }


def _apply_preprocessing(data, preprocessing_fns, preprocessor_params, **context):
    """Apply a list of preprocessing functions to data."""
    if not preprocessing_fns:
        return data

    for i, preprocessing_fn in enumerate(preprocessing_fns):
        if preprocessor_params and isinstance(preprocessor_params, list):
            params = preprocessor_params[i] if i < len(preprocessor_params) else None
        else:
            params = preprocessor_params
        if context and _preprocessor_accepts_context(preprocessing_fn):
            data = preprocessing_fn(
                data,
                params,
                **_preprocessor_context_kwargs(preprocessing_fn, context),
            )
        else:
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
        include_sample_ids: Add stable string sample IDs to selected task rows.
    """

    def __init__(
        self,
        raws: list,
        task_df: pd.DataFrame,
        window_width: float,
        preprocessing_fns=None,
        preprocessor_params=None,
        per_raw_event_times=None,
        include_sample_ids: bool = False,
    ):
        self.window_width = window_width
        self.preprocessing_fns = preprocessing_fns
        self.preprocessor_params = preprocessor_params
        self.task_df = task_df.copy()
        if include_sample_ids:
            if "sample_id" in self.task_df:
                sample_ids = self.task_df["sample_id"]
            elif "event_id" in self.task_df:
                sample_ids = self.task_df["event_id"]
            else:
                sample_ids = pd.Series(
                    np.arange(len(self.task_df), dtype=np.int64),
                    index=self.task_df.index,
                )
            if sample_ids.isna().any() or sample_ids.duplicated().any():
                raise ValueError("Task sample IDs must be present and unique")
            # A single UTF-8 representation gives HDF5 artifacts a stable join
            # key even when tasks use different event-id scalar types.
            self.task_df["sample_id"] = sample_ids.astype(str).to_numpy()
        self.data_durations = [raw.times[-1] for raw in raws]
        self._raw_subject_channel_counts = [len(raw.ch_names) for raw in raws]
        self._raw_subject_channel_names = [list(raw.ch_names) for raw in raws]
        self.subject_channel_counts = list(self._raw_subject_channel_counts)
        self._sfreqs = [raw.info["sfreq"] for raw in raws]

        if per_raw_event_times is None:
            base_onsets = task_df["start"].to_numpy(dtype=float)
            per_raw_event_times = [base_onsets.copy() for _ in raws]
        if len(per_raw_event_times) != len(raws):
            raise ValueError(
                "per_raw_event_times must contain one onset array per Raw"
            )
        self.per_raw_event_times = []
        for onsets in per_raw_event_times:
            values = np.asarray(onsets, dtype=float)
            if values.ndim != 1 or len(values) != len(task_df):
                raise ValueError(
                    "Each per-subject event-time array must be one-dimensional "
                    "and match task_df length"
                )
            self.per_raw_event_times.append(values)

        if len(set(self._sfreqs)) != 1:
            raise ValueError(
                "RawNeuralDataset requires all raws to share the same sampling rate"
            )

        self.sfreq = self._sfreqs[0]
        self.raw_arrays = [np.asarray(raw.get_data(), dtype=np.float32) for raw in raws]

    def _lag_window_params(self, lag: int):
        lag_offset = int(round((lag / 1000 - self.window_width / 2) * self.sfreq))
        n_window_samples = int(round((self.window_width - 2e-3) * self.sfreq)) + 1
        tmin = lag / 1000 - self.window_width / 2
        tmax = lag / 1000 + self.window_width / 2 - 2e-3
        return lag_offset, n_window_samples, tmin, tmax

    def _selection_for_lag(self, lag: int):
        _, _, tmin, tmax = self._lag_window_params(lag)
        common_bounds_mask = np.ones(len(self.task_df), dtype=bool)
        common_nonduplicate_mask = np.ones(len(self.task_df), dtype=bool)
        onset_samples_by_raw = []
        for data_duration, event_times in zip(
            self.data_durations, self.per_raw_event_times
        ):
            finite = np.isfinite(event_times)
            in_bounds = (event_times + tmin >= 0) & (
                event_times + tmax <= data_duration
            )
            safe_event_times = np.where(finite, event_times, 0.0)
            onset_samples = np.rint(safe_event_times * self.sfreq).astype(np.int64)
            nonduplicate = ~pd.Series(onset_samples).duplicated().to_numpy()
            common_bounds_mask &= finite & in_bounds
            common_nonduplicate_mask &= nonduplicate
            onset_samples_by_raw.append(onset_samples)

        common_mask = common_bounds_mask & common_nonduplicate_mask
        selection = np.flatnonzero(common_mask)
        if len(selection) == 0:
            raise ValueError("No valid events found within data time bounds")

        selected_rows_df = self.task_df.iloc[selection].copy()
        # Preserve the legacy MNE Epochs selection index: bounds filtering
        # resets the row index, while duplicate removal can leave gaps.
        bounded_positions = np.flatnonzero(common_bounds_mask)
        selected_rows_df.index = np.searchsorted(bounded_positions, selection)
        per_raw_onsets = [onsets[selection] for onsets in onset_samples_by_raw]
        subject_channel_counts = list(self._raw_subject_channel_counts)
        subject_channel_names = list(self._raw_subject_channel_names)

        return (
            selected_rows_df,
            per_raw_onsets,
            subject_channel_counts,
            subject_channel_names,
        )

    def _slice_lag_windows(
        self,
        lag: int,
        selected_positions,
        per_raw_onsets,
        subject_channel_counts,
    ):
        lag_offset, n_window_samples, _, _ = self._lag_window_params(lag)
        selected_positions = np.asarray(selected_positions, dtype=np.int64)
        total_channel_count = sum(subject_channel_counts)
        neural = np.empty(
            (len(selected_positions), total_channel_count, n_window_samples),
            dtype=np.float32,
        )
        channel_start = 0
        for raw_array, onset_samples, channel_count in zip(
            self.raw_arrays, per_raw_onsets, subject_channel_counts
        ):
            channel_stop = channel_start + channel_count
            selected_onsets = onset_samples[selected_positions]
            for row_idx, onset in enumerate(selected_onsets):
                neural[
                    row_idx,
                    channel_start:channel_stop,
                    :,
                ] = raw_array[
                    :,
                    onset + lag_offset : onset + lag_offset + n_window_samples,
                ]
            channel_start = channel_stop
        return neural

    @staticmethod
    def _targets_to_numpy(selected_rows_df):
        targets = selected_rows_df.target.to_numpy(copy=True)
        if targets.dtype == object:
            targets = np.stack(targets)
        return np.asarray(targets, dtype=np.float32)

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
        (
            selected_rows_df,
            per_raw_onsets,
            subject_channel_counts,
            subject_channel_names,
        ) = self._selection_for_lag(lag)

        selected_positions = np.arange(len(selected_rows_df), dtype=np.int64)
        neural = self._slice_lag_windows(
            lag, selected_positions, per_raw_onsets, subject_channel_counts
        )

        if self.preprocessing_fns:
            neural = _apply_preprocessing(
                neural,
                self.preprocessing_fns,
                self.preprocessor_params,
                selected_rows=selected_rows_df,
                selected_rows_df=selected_rows_df,
                lag=lag,
                subject_channel_names=subject_channel_names,
                subject_channel_counts=subject_channel_counts,
            )

        targets_tensor = torch.from_numpy(self._targets_to_numpy(selected_rows_df))

        assert neural.shape[0] == len(
            selected_rows_df
        ), "Mismatch between neural data and task_df lengths"

        neural_tensor = torch.from_numpy(np.asarray(neural, dtype=np.float32))
        self.subject_channel_counts = subject_channel_counts

        return neural_tensor, targets_tensor, selected_rows_df, subject_channel_counts

    def build_preprocessed_chunks(self, lag: int, num_chunks: int, cache_dir: str):
        (
            selected_rows_df,
            per_raw_onsets,
            subject_channel_counts,
            subject_channel_names,
        ) = self._selection_for_lag(lag)

        num_chunks = max(1, int(num_chunks))
        chunk_size = max(1, math.ceil(len(selected_rows_df) / num_chunks))
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        chunk_row_positions = []
        all_targets = self._targets_to_numpy(selected_rows_df)
        try:
            for chunk_idx, start in enumerate(
                range(0, len(selected_rows_df), chunk_size)
            ):
                stop = min(start + chunk_size, len(selected_rows_df))
                row_positions = np.arange(start, stop, dtype=np.int64)
                chunk_rows_df = selected_rows_df.iloc[row_positions]
                neural = self._slice_lag_windows(
                    lag, row_positions, per_raw_onsets, subject_channel_counts
                )

                if self.preprocessing_fns:
                    neural = _apply_preprocessing(
                        neural,
                        self.preprocessing_fns,
                        self.preprocessor_params,
                        selected_rows=chunk_rows_df,
                        selected_rows_df=chunk_rows_df,
                        lag=lag,
                        subject_channel_names=subject_channel_names,
                        subject_channel_counts=subject_channel_counts,
                    )

                chunk_path = cache_path / (
                    f"lag_{lag}_chunk_{chunk_idx}_{os.getpid()}.npz"
                )
                temp_chunk_path = cache_path / (
                    f".lag_{lag}_chunk_{chunk_idx}_{os.getpid()}.tmp.npz"
                )
                chunk_paths.append(temp_chunk_path)
                np.savez(
                    temp_chunk_path,
                    data=np.asarray(neural, dtype=np.float32),
                    targets=all_targets[row_positions],
                    row_positions=row_positions,
                )
                os.replace(temp_chunk_path, chunk_path)
                chunk_paths[-1] = chunk_path
                chunk_row_positions.append(row_positions)
                del neural
        except Exception:
            for chunk_path in chunk_paths:
                try:
                    chunk_path.unlink()
                except FileNotFoundError:
                    pass
            for temp_chunk_path in cache_path.glob(
                f".lag_{lag}_chunk_*_{os.getpid()}.tmp.npz"
            ):
                try:
                    temp_chunk_path.unlink()
                except FileNotFoundError:
                    pass
            raise

        self.subject_channel_counts = subject_channel_counts
        return PreprocessedChunkStore(
            chunk_paths=chunk_paths,
            chunk_row_positions=chunk_row_positions,
            data_df=selected_rows_df,
            targets=torch.from_numpy(all_targets),
            subject_channel_counts=subject_channel_counts,
        )


@dataclass
class PreprocessedChunkStore:
    chunk_paths: list[Path]
    chunk_row_positions: list[np.ndarray]
    data_df: pd.DataFrame
    targets: torch.Tensor
    subject_channel_counts: list[int]

    def cleanup(self):
        for path in self.chunk_paths:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    def get_loader(
        self,
        indices,
        task_config,
        targets,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        return ChunkedPreprocessedLoader(
            self.chunk_paths,
            self.data_df,
            indices,
            task_config,
            targets,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            chunk_row_positions=self.chunk_row_positions,
        )


class ChunkedPreprocessedLoader:
    def __init__(
        self,
        chunk_paths,
        data_df,
        indices,
        task_config,
        targets,
        batch_size,
        shuffle=False,
        seed=0,
        chunk_row_positions=None,
    ):
        self.chunk_paths = list(chunk_paths)
        if chunk_row_positions is None:
            chunk_row_positions = self._load_chunk_row_positions(self.chunk_paths)
        self.chunk_row_positions = [
            np.asarray(row_positions, dtype=np.int64)
            for row_positions in chunk_row_positions
        ]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.index_set = set(self.indices.tolist())
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.input_fields = task_config.task_specific_config.input_fields
        self.extra_inputs = data_utils.df_columns_to_tensors(
            data_df,
            self.input_fields,
            self.indices,
        )
        self.targets_for_split = targets[
            torch.as_tensor(self.indices, dtype=torch.long)
        ]
        self.absolute_to_split_position = {
            int(abs_pos): split_pos for split_pos, abs_pos in enumerate(self.indices)
        }
        self.chunk_plans = self._build_chunk_plans()
        self._length = sum(len(plan["batches"]) for plan in self.chunk_plans)

    @staticmethod
    def _load_chunk_row_positions(chunk_paths):
        row_positions = []
        for chunk_path in chunk_paths:
            with np.load(chunk_path) as loaded:
                row_positions.append(
                    loaded["row_positions"].astype(np.int64, copy=False)
                )
        return row_positions

    def _build_chunk_plans(self):
        plans = []
        for chunk_idx, row_positions in enumerate(self.chunk_row_positions):
            mask = np.fromiter(
                (int(pos) in self.index_set for pos in row_positions),
                dtype=bool,
                count=len(row_positions),
            )
            if not np.any(mask):
                continue

            # absolute = selected_rows_df row, local = row within this chunk,
            # split = row within this loader's train/val/test tensors.
            local_positions = np.flatnonzero(mask)
            absolute_positions = row_positions[local_positions]
            split_positions = np.asarray(
                [
                    self.absolute_to_split_position[int(abs_pos)]
                    for abs_pos in absolute_positions
                ],
                dtype=np.int64,
            )
            batches = [
                (
                    local_positions[start : start + self.batch_size],
                    split_positions[start : start + self.batch_size],
                )
                for start in range(0, len(local_positions), self.batch_size)
            ]
            plans.append(
                {
                    "chunk_idx": chunk_idx,
                    "local_positions": local_positions,
                    "absolute_positions": absolute_positions,
                    "split_positions": split_positions,
                    "batches": batches,
                }
            )
        return plans

    def ordered_indices(self):
        """Return absolute row positions in non-shuffled iteration order."""
        if not self.chunk_plans:
            return np.array([], dtype=np.int64)
        return np.concatenate(
            [plan["absolute_positions"] for plan in self.chunk_plans]
        ).astype(np.int64, copy=False)

    def __len__(self):
        return self._length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        plan_order = np.arange(len(self.chunk_plans))
        if self.shuffle:
            rng.shuffle(plan_order)
        self.epoch += 1

        for plan_idx in plan_order:
            plan = self.chunk_plans[int(plan_idx)]
            local_positions = plan["local_positions"]
            split_positions = plan["split_positions"]
            if self.shuffle:
                order = np.arange(len(local_positions))
                rng.shuffle(order)
                local_positions = local_positions[order]
                split_positions = split_positions[order]
                batches = [
                    (
                        local_positions[start : start + self.batch_size],
                        split_positions[start : start + self.batch_size],
                    )
                    for start in range(0, len(local_positions), self.batch_size)
                ]
            else:
                batches = plan["batches"]

            with np.load(self.chunk_paths[plan["chunk_idx"]]) as loaded:
                data = loaded["data"]

                for batch_local, batch_split in batches:
                    neural_batch = torch.from_numpy(
                        np.asarray(data[batch_local], dtype=np.float32)
                    )
                    inputs_dict = {
                        name: tensor[torch.as_tensor(batch_split, dtype=torch.long)]
                        for name, tensor in self.extra_inputs.items()
                    }
                    target_batch = self.targets_for_split[
                        torch.as_tensor(batch_split, dtype=torch.long)
                    ]
                    yield neural_batch, inputs_dict, target_batch


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
