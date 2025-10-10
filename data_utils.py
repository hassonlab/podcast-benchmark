import os
from math import gcd

from typing import Optional

import numpy as np
import mne
from mne_bids import BIDSPath
import pandas as pd

from scipy.signal import resample_poly

from config import DataParams
import registry


def _as_float_array(data: np.ndarray) -> np.ndarray:
    """Validate neural data array and ensure float32 dtype."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(
            "Neural preprocessors expect data with at least two dimensions (examples, channels, ...)."
        )
    return arr


def _get_channel_axis(data: np.ndarray) -> int:
    """Return the axis index corresponding to channels for neural arrays."""

    if data.ndim == 2:
        return 1
    return 1  # shape presumed (examples, channels, samples, ...)


def load_raws(data_params: DataParams):
    """
    Loads raw iEEG data for multiple subjects based on specified parameters.

    This function:
    1. Iterates over subject IDs provided in the configuration.
    2. Constructs BIDS-compliant file paths to locate preprocessed high-gamma iEEG data.
    3. Loads each subject's data using MNE's `read_raw_fif` function.
    4. Optionally filters channels using a regular expression (e.g., for selecting specific electrode groups).
    5. Collects and returns a list of raw MNE objects.

    Args:
        data_params (DataParams): Configuration object containing subject IDs, data paths,
        and optional channel filtering settings.

    Returns:
        List[mne.io.Raw]: A list of raw iEEG recordings for the specified subjects.
    """
    raws = []
    for sub_id in data_params.subject_ids:
        file_path = BIDSPath(
            root=os.path.join(data_params.data_root, "derivatives/ecogprep"),
            subject=f"{sub_id:02}",
            task="podcast",
            datatype="ieeg",
            description="highgamma",
            suffix="ieeg",
            extension=".fif",
        )

        raw = mne.io.read_raw_fif(file_path, verbose=False)
        if data_params.per_subject_electrodes:
            subject_electrode_names = data_params.per_subject_electrodes[sub_id]
            picks = mne.pick_channels(raw.ch_names, subject_electrode_names)
            raw = raw.pick(picks)
        elif data_params.channel_reg_ex:
            picks = mne.pick_channels_regexp(raw.ch_names, data_params.channel_reg_ex)
            raw = raw.pick(picks)
        raws.append(raw)

    return raws


def read_subject_mapping(participant_map_file: str, delimiter="\t"):
    """
    Read and parse a participant mapping file to create subject ID mappings.

    This function reads a tab-delimited or custom-delimited file containing participant
    information and creates a mapping from NYU subject IDs to converted participant IDs.
    The participant IDs are expected to be in the format "sub-XX" where XX is extracted
    as an integer.

    Args:
        participant_map_file (str): Path to the participant mapping CSV/TSV file.
                                   Expected to contain columns 'nyu_id' and 'participant_id'.
        delimiter (str, optional): Delimiter used in the file. Defaults to "\t" (tab).

    Returns:
        dict[int, int]: A dictionary mapping NYU subject IDs to converted participant IDs.
                       Keys are nyu_id values, values are integers extracted from
                       participant_id (e.g., "sub-05" -> 5).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If required columns 'nyu_id' or 'participant_id' are missing.
        ValueError: If participant_id format is invalid (not "sub-XX" format).

    Example:
        >>> # File contains:
        >>> # nyu_id    participant_id
        >>> # 661       sub-05
        >>> # 717       sub-12
        >>> mapping = read_subject_mapping('participants.tsv')
        >>> print(mapping)
        {661: 5, 717: 12}
    """
    # Read the subject mapping from data.
    participant_info_df = pd.read_csv(participant_map_file, delimiter=delimiter)

    def participant_id_to_int(participant_id):
        return int(participant_id.split("-")[1])

    subject_id_map = {
        x[1]["nyu_id"]: participant_id_to_int(x[1]["participant_id"])
        for x in participant_info_df.iterrows()
    }
    return subject_id_map


def read_electrode_file(
    file_path: str, subject_mapping: Optional[dict[int, int]] = None
):
    """
    Parse an electrode mapping CSV file to create a subject-to-electrodes mapping.

    This function reads a CSV file containing electrode information organized by subject
    and returns a dictionary mapping each subject ID to their list of electrode names.
    Each subject can have multiple electrodes, and the electrode order is preserved
    as it appears in the CSV file.

    Args:
        file_path (str): Path to the CSV file containing electrode data.
                        The CSV must have columns 'subject' (int) and 'elec' (str).
        subject_mapping (dict[int, int]): Optional mapping to go from the subject ID in the
                        file to the subject ID of our dataset (i.e. 798 -> 9)

    Returns:
        dict: A dictionary where keys are subject IDs (int) and values are lists
              of electrode names (str) for that subject. For example:
              {1: ['A1', 'A2', 'B1'], 2: ['C1', 'C2']}

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If required columns 'subject' or 'elec' are missing from the CSV.

    Example:
        >>> # CSV file contains:
        >>> # subject,elec
        >>> # 1,A1
        >>> # 1,A2
        >>> # 2,C1
        >>> result = read_electrode_file('electrodes.csv')
        >>> print(result)
        {1: ['A1', 'A2'], 2: ['C1']}
    """
    file_data = pd.read_csv(file_path)
    subjects, electrodes = file_data.subject, file_data.elec

    sub_elec_mapping = {}
    for subject, electrode in zip(subjects, electrodes):
        subject = int(subject)
        if subject_mapping is not None:
            subject = subject_mapping[subject]

        if subject not in sub_elec_mapping.keys():
            sub_elec_mapping[subject] = []

        sub_elec_mapping[subject].append(electrode)

    return sub_elec_mapping


def get_data(
    lag,
    raws: list[mne.io.Raw],
    df_word: pd.DataFrame,
    window_width: float,
    preprocessing_fn=None,
    preprocessor_params: dict = None,
    word_column: Optional[str] = None,
):
    """Gather data for every word in df_word from raw.

    Args:
        lag: the lag relative to each word onset to gather data around
        raws: list of mne.Raw object holding electrode data
        df_word: dataframe containing columns start, end, word, and target
        window_width: the width of the window which is gathered around each word onset + lag
        preprocessing_fn: function to apply to epoch data.
            Should have contract:
                preprocessing_fn(data: np.array of shape [num_words, num_electrodes, timesteps],
                                preprocessor_params)  -> array of shape [num_words, ...]
        word_column: If provided, will return the column of words specified here.
    """
    datas = []
    for raw in raws:
        # Calculate time bounds for filtering
        tmin = lag / 1000 - window_width / 2
        tmax = lag / 1000 + window_width / 2 - 2e-3
        data_duration = raw.times[-1]  # End time of the data

        # Filter out events where the time window falls outside data bounds
        valid_mask = (df_word.start + tmin >= 0) & (
            df_word.start + tmax <= data_duration
        )
        df_word_valid = df_word[valid_mask].reset_index(drop=True)

        if len(df_word_valid) == 0:
            # No valid events for this raw, skip
            continue

        events = np.zeros((len(df_word_valid), 3), dtype=int)
        events[:, 0] = (df_word_valid.start * raw.info["sfreq"]).astype(int)

        epochs = mne.Epochs(
            raw,
            events,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            proj=False,
            event_id=None,
            preload=True,
            on_missing="ignore",
            event_repeated="merge",
            verbose="ERROR",
        )

        data = epochs.get_data(copy=False)
        selected_targets = df_word_valid.target[epochs.selection]

        # TODO: Clean this up so we don't need to pass around this potentially None variable.
        if word_column:
            selected_words = df_word_valid[word_column].to_numpy()[epochs.selection]
        else:
            selected_words = None

        # Make sure the number of samples match
        assert data.shape[0] == selected_targets.shape[0], "Sample counts don't match"
        if selected_words is not None:
            assert data.shape[0] == selected_words.shape[0], "Words don't match"

        datas.append(data)

    if len(datas) == 0:
        raise ValueError("No valid events found within data time bounds")

    datas = np.concatenate(datas, axis=1)

    if preprocessing_fn:
        datas = preprocessing_fn(datas, preprocessor_params)

    return datas, selected_targets, selected_words


@registry.register_data_preprocessor("window_rms")
def window_rms_preprocessor(
    data: np.ndarray, preprocessor_params: Optional[dict] = None
) -> np.ndarray:
    """Reduce each neural window to root-mean-square (RMS) amplitudes.

    Args:
        data (np.ndarray): Neural windows shaped ``(n_examples, n_channels, n_samples)``.
        preprocessor_params (dict, optional): Currently unused but kept for
            parity with other preprocessors.

    Returns:
        np.ndarray: RMS features shaped ``(n_examples, n_channels)`` where each
        value is ``sqrt(mean(window**2))`` for the corresponding channel.

    Raises:
        ValueError: If ``data`` does not have three dimensions.
    """

    if data.ndim != 3:
        raise ValueError(
            "window_rms_preprocessor expects data with shape (examples, channels, samples)."
        )

    # Use float64 during accumulation for numerical stability, then cast back.
    squared = np.square(data, dtype=np.float64)
    mean_sq = squared.mean(axis=-1)
    rms = np.sqrt(np.maximum(mean_sq, 0.0))
    return rms.astype(np.float32, copy=False)


@registry.register_data_preprocessor("log_transform")
def log_transform_preprocessor(
    data: np.ndarray, preprocessor_params: Optional[dict] = None
) -> np.ndarray:
    """Apply a logarithmic compression to neural amplitudes.

    Args:
        data (np.ndarray): Neural data with shape ``(n_examples, n_channels, ...)``.
        preprocessor_params (dict, optional): Supported keys:
            - ``epsilon`` (float): Minimum offset added before the log. Defaults to 1e-6.
            - ``clip_min`` (float): Lower bound applied before the log. Defaults to 0.0.
            - ``log_base`` (float or str): ``10`` (default), ``"e"`` or a custom base > 0 and != 1.
            - ``scale`` (float): Optional scaling multiplier applied after the log. Defaults to 1.0.

    Returns:
        np.ndarray: Log-compressed amplitudes with the same shape as the input.
    """

    arr = _as_float_array(data)
    params = preprocessor_params or {}

    epsilon = float(params.get("epsilon", 1e-6))
    clip_min = float(params.get("clip_min", 0.0))
    scale = float(params.get("scale", 1.0))
    base = params.get("log_base", 10.0)

    if epsilon <= 0:
        raise ValueError("epsilon must be positive for log_transform_preprocessor.")

    arr64 = arr.astype(np.float64, copy=False)
    clipped = np.clip(arr64, clip_min, None)
    shifted = clipped + epsilon

    if isinstance(base, str):
        if base.lower() == "e":
            log_values = np.log(shifted)
        elif base.lower() == "10":
            log_values = np.log10(shifted)
        else:
            raise ValueError("log_base string must be 'e' or '10'.")
    else:
        base = float(base)
        if base <= 0 or np.isclose(base, 1.0):
            raise ValueError("log_base must be > 0 and != 1.")
        log_values = np.log(shifted) / np.log(base)

    if scale != 1.0:
        log_values *= scale

    return log_values.astype(np.float32, copy=False)


@registry.register_data_preprocessor("zscore")
def zscore_preprocessor(
    data: np.ndarray, preprocessor_params: Optional[dict] = None
) -> np.ndarray:
    """Standardize each channel independently across all observations.

    Args:
        data (np.ndarray): Neural data shaped ``(n_examples, n_channels, ...)``.
        preprocessor_params (dict, optional): Supported keys:
            - ``epsilon`` (float): Minimum standard deviation. Defaults to 1e-6.
            If ``channel_means`` and ``channel_stds`` are not provided they are computed and,
            when ``preprocessor_params`` is a mutable dict, stored back into it as float32 arrays.

    Returns:
        np.ndarray: Z-scored data with the same shape as the input.

    Raises:
        ValueError: If provided statistics have mismatched shapes.
    """

    arr = _as_float_array(data)
    params = preprocessor_params if preprocessor_params is not None else {}

    epsilon = float(params.get("epsilon", 1e-6))
    channel_axis = _get_channel_axis(arr)

    channel_first = np.moveaxis(arr, channel_axis, 0)
    flat = channel_first.reshape(channel_first.shape[0], -1)
    flat64 = flat.astype(np.float64, copy=False)

    means = params.get("channel_means")
    stds = params.get("channel_stds")

    if means is not None and stds is not None:
        means = np.asarray(means, dtype=np.float64).reshape(-1, 1)
        stds = np.asarray(stds, dtype=np.float64).reshape(-1, 1)
        if means.shape[0] != flat64.shape[0] or stds.shape[0] != flat64.shape[0]:
            raise ValueError(
                "channel_means and channel_stds must match the number of channels in data."
            )
    else:
        means = np.nanmean(flat64, axis=1, keepdims=True)
        stds = np.nanstd(flat64, axis=1, keepdims=True)
        if preprocessor_params is not None:
            preprocessor_params["channel_means"] = means.squeeze(-1).astype(np.float32)
            preprocessor_params["channel_stds"] = stds.squeeze(-1).astype(np.float32)

    stds = np.where(~np.isfinite(stds) | (stds < epsilon), epsilon, stds)
    normalized = (flat64 - means) / stds
    normalized = normalized.reshape(channel_first.shape)

    normalized = np.moveaxis(normalized, 0, channel_axis)
    return normalized.astype(np.float32, copy=False)


def load_ieeg_edf_files(
    edf_paths: list[str],
    target_sampling_rate: Optional[float] = None,
    preload: bool = True,
    dtype: np.dtype = np.float32,
) -> tuple[list[np.ndarray], list[list[str]], list[float]]:
    """Load iEEG EDF files and optionally resample them to a common sampling rate."""

    data_list: list[np.ndarray] = []
    channel_names: list[list[str]] = []
    sampling_rates: list[float] = []

    effective_target_sr = target_sampling_rate

    for path in edf_paths:
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
        data = raw.get_data().astype(dtype, copy=False)
        channels = list(raw.ch_names)
        sr = float(raw.info["sfreq"])

        if sr <= 0:
            raise ValueError(
                f"EDF file '{path}' reported a non-positive sampling rate: {sr}Hz"
            )

        if effective_target_sr is None:
            effective_target_sr = sr

        sr_int = int(round(sr))
        target_sr_int = int(round(effective_target_sr))

        if sr_int != target_sr_int:
            if target_sr_int <= 0:
                raise ValueError("target_sampling_rate must be positive when provided.")

            factor = gcd(target_sr_int, sr_int) or 1
            up = target_sr_int // factor
            down = sr_int // factor

            data = resample_poly(data, up, down, axis=1).astype(dtype, copy=False)
            sr = float(target_sr_int)

        data_list.append(data)
        channel_names.append(channels)
        sampling_rates.append(sr)

    return data_list, channel_names, sampling_rates

