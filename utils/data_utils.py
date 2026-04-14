import os
from typing import Optional

import numpy as np
import torch
import mne
from mne_bids import BIDSPath
import pandas as pd
import torch

from core.config import DataParams
from core import registry


def load_raws(data_params: DataParams):
    """
    Loads raw iEEG data with YAML-configurable preprocessing.
    All transformations (Resampling, Drop Bad, Scaling) happen here directly on the Raw object.
    """
    raws = []

    for sub_id in data_params.subject_ids:
        if data_params.use_high_gamma:
            root_path = os.path.join(data_params.data_root, "derivatives/ecogprep")
            description = "highgamma"
            print(f"Loading High Gamma data for subject {sub_id}...")
        else:
            root_path = os.path.join(data_params.data_root, "derivatives/ecogprep")
            description = None
            print(f"Loading Raw data for subject {sub_id}...")

        file_path = BIDSPath(
            root=root_path,
            subject=f"{sub_id:02}",
            task="podcast",
            datatype="ieeg",
            description=description,
            suffix="ieeg",
            extension=".fif",
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        if data_params.target_sr is not None:
            if int(raw.info["sfreq"]) != int(data_params.target_sr):
                print(
                    f"  - [Config] Resampling {raw.info['sfreq']}Hz -> {data_params.target_sr}Hz"
                )
                raw.resample(data_params.target_sr)

        if data_params.do_drop_bads:
            try:
                # Build channels.tsv path via BIDSPath (avoid accidental reading of .fif)
                ch_path = file_path.copy()
                ch_path.update(suffix="channels", extension=".tsv")
                tsv_path = str(ch_path)

                # Some datasets omit 'description' on channels.tsv even if present on ieeg.fif
                if not os.path.exists(tsv_path) and getattr(
                    file_path, "description", None
                ):
                    ch_path2 = file_path.copy()
                    ch_path2.update(
                        description=None, suffix="channels", extension=".tsv"
                    )
                    tsv_path = str(ch_path2)

                if not os.path.exists(tsv_path) and "derivatives" in str(file_path):
                    raw_root = data_params.data_root
                    raw_bids_path = BIDSPath(
                        root=raw_root,
                        subject=f"{sub_id:02}",
                        task="podcast",
                        datatype="ieeg",
                        suffix="channels",
                        extension=".tsv",
                    )
                    tsv_path = str(raw_bids_path)

                if os.path.exists(tsv_path):
                    df_ch = pd.read_csv(tsv_path, sep="\t")
                    if "status" in df_ch.columns and "name" in df_ch.columns:
                        bad_channels = df_ch[df_ch["status"] == "bad"]["name"].tolist()
                        if bad_channels:
                            bad_channels = [
                                ch for ch in bad_channels if ch in raw.ch_names
                            ]
                            raw.info["bads"].extend(bad_channels)
                            print(
                                f"  - [Config] Dropping {len(bad_channels)} bad channels"
                            )
                            raw.drop_channels(bad_channels)
            except Exception as e:
                print(f"Warning: Failed to process bad channels: {e}")

        if data_params.signal_unit == "uV":
            print(f"  - [Config] Scaling data: Volt -> MicroVolt (x 1e6)")
            raw.apply_function(lambda x: x * 1e6, channel_wise=False)

        if data_params.per_subject_electrodes:
            subject_electrode_names = data_params.per_subject_electrodes[sub_id]
            picks = [ch for ch in subject_electrode_names if ch in raw.ch_names]
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


def extract_subject_id_from_raw(raw: mne.io.Raw) -> int:
    """
    Extract subject ID from MNE Raw object.

    This function extracts the subject ID from the Raw object's info.
    The subject ID is expected to be in the format "sub-XX" in the info['subject'] field,
    or can be extracted from the file path.

    Args:
        raw: MNE Raw object

    Returns:
        int: Subject ID as integer (e.g., 3 for "sub-03")

    Raises:
        ValueError: If subject ID cannot be extracted
    """
    # Try to get from info['subject']
    if hasattr(raw.info, "subject") and raw.info["subject"] is not None:
        subject_str = str(raw.info["subject"])
        if subject_str.startswith("sub-"):
            return int(subject_str.split("-")[1])

    # Try to get from file path if available
    if hasattr(raw, "filenames") and raw.filenames:
        for filename in raw.filenames:
            # Convert PosixPath to string if needed (mne may store paths as Path objects)
            filename_str = str(filename)
            if "sub-" in filename_str:
                # Extract subject ID from path like ".../sub-03/..."
                parts = filename_str.split("sub-")
                if len(parts) > 1:
                    subject_part = parts[1].split("/")[0]
                    try:
                        return int(subject_part)
                    except ValueError:
                        continue

    raise ValueError(
        f"Could not extract subject ID from Raw object. "
        f"Info subject: {raw.info.get('subject', None)}, "
        f"Filenames: {getattr(raw, 'filenames', None)}"
    )


def get_lip_coordinates(subject_id: int, data_root: str = "data") -> pd.DataFrame:
    """
    Load LIP coordinates from TSV file for a given subject.

    This function loads LIP (Lateral, Inferior, Posterior) coordinates from
    the BIDS-compliant TSV file: data/sub-XX/ieeg/sub-XX_space-LIP_electrodes.tsv

    Args:
        subject_id: Subject ID as integer (e.g., 3 for sub-03)
        data_root: Root directory for data files (default: "data")

    Returns:
        pd.DataFrame: DataFrame with columns ['name', 'x', 'y', 'z', ...]
                     where x, y, z are LIP coordinates as integers

    Raises:
        FileNotFoundError: If the TSV file does not exist
        ValueError: If required columns are missing
    """
    tsv_path = os.path.join(
        data_root,
        f"sub-{subject_id:02}",
        "ieeg",
        f"sub-{subject_id:02}_space-LIP_electrodes.tsv",
    )

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"LIP coordinates file not found: {tsv_path}. "
            f"Make sure the file exists or set return_lip_coords=False."
        )

    df = pd.read_csv(tsv_path, sep="\t")

    # Validate required columns
    required = {"name", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV file missing required columns: {missing}")

    # Clean and convert data types
    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    for c in ["x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove rows with NaN coordinates
    df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

    return df


def get_mni_coordinates(
    subject_id: int, channel_names: list[str], data_root: str = "data"
) -> np.ndarray:
    """
    Load MNI xyz coordinates from TSV file for specified channels.

    This function loads MNI152 coordinates from the BIDS-compliant TSV file:
    data/sub-XX/ieeg/sub-XX_space-MNI152NLin2009aSym_electrodes.tsv

    The coordinates are extracted in the order specified by channel_names,
    matching the channel order in the neural data.

    Args:
        subject_id: Subject ID as integer (e.g., 1 for sub-01)
        channel_names: List of channel names to extract coordinates for
                       (must match the order of channels in the neural data)
        data_root: Root directory for data files (default: "data")

    Returns:
        np.ndarray: MNI xyz coordinates [num_channels, 3] matching channel_names order
                    dtype=np.float32
                    If a channel is not found in the TSV file, NaN coordinates are inserted

    Raises:
        FileNotFoundError: If the TSV file does not exist
        ValueError: If required columns are missing

    Note:
        This function is designed for DIVER model's PositionalEncoding3D,
        which requires xyz_id in data_info_list for each sample.
        Podcast Benchmark uses consistent channel order across all samples,
        so the same xyz_id can be used for all samples in a batch.
    """
    tsv_path = os.path.join(
        data_root,
        f"sub-{subject_id:02}",
        "ieeg",
        f"sub-{subject_id:02}_space-MNI152NLin2009aSym_electrodes.tsv",
    )

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"MNI coordinates file not found: {tsv_path}. "
            f"Make sure the file exists. DIVER model requires MNI coordinates for PositionalEncoding3D."
        )

    df = pd.read_csv(tsv_path, sep="\t")

    # Validate required columns
    required = {"name", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV file missing required columns: {missing}")

    # Clean and convert data types
    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip().str.upper()

    # Normalize channel names for matching
    normalized_channel_names = [ch.strip().upper() for ch in channel_names]

    # Extract xyz coordinates in the order of channel_names
    xyz_list = []
    for ch_name in normalized_channel_names:
        # Find matching channel in TSV
        matching_rows = df[df["name"] == ch_name]
        if len(matching_rows) > 0:
            # Use first match (should be unique)
            xyz = matching_rows[["x", "y", "z"]].iloc[0].values
            xyz_list.append(xyz)
        else:
            # Channel not found - insert NaN coordinates
            print(
                f"Warning: Channel '{ch_name}' not found in MNI coordinates TSV. Inserting NaN coordinates."
            )
            xyz_list.append([np.nan, np.nan, np.nan])

    # Convert to numpy array
    xyz_id = np.array(xyz_list, dtype=np.float32)

    # Check for NaN coordinates
    nan_count = np.isnan(xyz_id).any(axis=1).sum()
    if nan_count > 0:
        print(
            f"Warning: {nan_count} channels have NaN MNI coordinates. "
            f"PositionalEncoding3D will mask these with zeros."
        )

    return xyz_id


def _apply_preprocessing(data, preprocessing_fns, preprocessor_params):
    """Apply a list of preprocessing functions to data.

    Args:
        data: numpy array to preprocess
        preprocessing_fns: list of preprocessing functions to apply in order
        preprocessor_params: parameters to pass to preprocessing functions (dict or list of dicts)

    Returns:
        Preprocessed data array
    """
    if not preprocessing_fns:
        return data

    for i, preprocessing_fn in enumerate(preprocessing_fns):
        if preprocessor_params and isinstance(preprocessor_params, list):
            params = preprocessor_params[i] if i < len(preprocessor_params) else None
        else:
            params = preprocessor_params
        data = preprocessing_fn(data, params)

    return data


def get_data(
    lag,
    raws: list[mne.io.Raw],
    task_df: pd.DataFrame,
    window_width: float,
    preprocessing_fns=None,
    preprocessor_params: dict = None,
):
    """Gather data for every row in task_df from raw.

    Args:
        lag: the lag relative to each word onset to gather data around
        raws: list of mne.Raw object holding electrode data
        task_df: dataframe containing columns start, target, and optionally word_column
        window_width: the width of the window which is gathered around each word onset + lag
        preprocessing_fns: functions to apply to epoch data in order of calling.
            Should have contract:
                preprocessing_fn(data: np.array of shape [num_words, num_electrodes, timesteps],
                                preprocessor_params)  -> array of shape [num_words, ...]
    """
    datas = []
    for raw in raws: #raw for each subject
        # Calculate time bounds for filtering
        tmin = lag / 1000 - window_width / 2
        tmax = lag / 1000 + window_width / 2 - 2e-3
        data_duration = raw.times[-1]  # End time of the data

        # Filter out events where the time window falls outside data bounds
        valid_mask = (task_df.start + tmin >= 0) & (
            task_df.start + tmax <= data_duration
        )
        task_df_valid = task_df[valid_mask].reset_index(drop=True)

        if len(task_df_valid) == 0:
            # No valid events for this raw, skip
            continue

        events = np.zeros((len(task_df_valid), 3), dtype=int)
        events[:, 0] = (task_df_valid.start * raw.info["sfreq"]).astype(int)

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
            event_repeated="merge", #! WARNING: this can drop event if multiple events fall into the same epoch.
            verbose="ERROR",
        )

        data = epochs.get_data(copy=False)  #* shape: [n_words, n_electrodes_this_subject, timesteps]           
        # Should be the same selected rows from all raws.
        # TODO: add an assertion to make sure this always holds.
        selected_rows_df = task_df_valid.iloc[epochs.selection]
        selected_targets = selected_rows_df.target.to_numpy()

        # Make sure the number of samples match
        assert data.shape[0] == selected_rows_df.shape[0], "Sample counts don't match"

        datas.append(data)

    if len(datas) == 0:
        raise ValueError("No valid events found within data time bounds")

    subject_channel_counts = [d.shape[1] for d in datas]
    datas = np.concatenate(datas, axis=1) #* axis=1 (electrode dimension) concat

    datas = _apply_preprocessing(datas, preprocessing_fns, preprocessor_params)

    return datas, selected_targets, selected_rows_df, subject_channel_counts


def df_columns_to_tensors(
    df: pd.DataFrame,
    column_names: Optional[list[str]],
    fold_indices: Optional[np.ndarray] = None,
) -> dict[str, torch.Tensor]:
    """
    Convert specified DataFrame columns to PyTorch tensors.

    Args:
        df: pandas DataFrame containing the data
        column_names: list of column names to convert to tensors
        fold_indices: optional array of row indices to select from the DataFrame.
                     If provided, only rows at these indices will be converted.

    Returns:
        dict: Dictionary where keys are column names and values are PyTorch tensors
              containing the values from those columns

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        >>> result = df_columns_to_tensors(df, ['a', 'b'])
        >>> print(result)
        {'a': tensor([1, 2, 3]), 'b': tensor([4., 5., 6.])}

        >>> # With fold indices
        >>> result = df_columns_to_tensors(df, ['a'], fold_indices=np.array([0, 2]))
        >>> print(result)
        {'a': tensor([1, 3])}
    """
    if column_names is None:
        return {}

    result = {}
    for col_name in column_names:
        column_data = df[col_name].values

        # Apply fold indices if provided
        if fold_indices is not None:
            column_data = column_data[fold_indices]

        # Check if we have an object array (likely containing arrays/lists)
        # If so, stack them into a 2D array
        if column_data.dtype == object:
            column_data = np.stack(column_data)

        result[col_name] = torch.tensor(column_data)
    return result
