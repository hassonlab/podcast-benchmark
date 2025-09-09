import os

import numpy as np
import mne
from mne_bids import BIDSPath
import pandas as pd

from config import DataParams


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
        if data_params.channel_reg_ex:
            picks = mne.pick_channels_regexp(raw.ch_names, data_params.channel_reg_ex)
            raw = raw.pick(picks)
        raws.append(raw)

    return raws


def get_data(
    lag,
    raws: list[mne.io.Raw],
    df_word: pd.DataFrame,
    window_width: float,
    preprocessing_fn=None,
    preprocessor_params: dict = None,
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
    """
    datas = []
    for raw in raws:
        events = np.zeros((len(df_word), 3), dtype=int)
        events[:, 0] = (df_word.start * raw.info["sfreq"]).astype(int)

        epochs = mne.Epochs(
            raw,
            events,
            tmin=lag / 1000 - window_width / 2,
            tmax=lag / 1000 + window_width / 2 - 2e-3,
            baseline=None,
            proj=False,
            event_id=None,
            preload=True,
            on_missing="ignore",
            event_repeated="merge",
            verbose="ERROR",
        )

        data = epochs.get_data(copy=False)
        selected_targets = df_word.target[epochs.selection]

        selected_words = df_word.word.to_numpy()[epochs.selection]

        # Make sure the number of samples match
        assert data.shape[0] == selected_targets.shape[0], "Sample counts don't match"
        assert data.shape[0] == selected_words.shape[0], "Words don't match"

        datas.append(data)

    datas = np.concatenate(datas, axis=1)

    if preprocessing_fn:
        datas = preprocessing_fn(datas, preprocessor_params)

    return datas, selected_targets, selected_words
