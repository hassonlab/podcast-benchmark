"""Podcast MNE Raw loader."""

import os

import mne
import pandas as pd
from mne_bids import BIDSPath

from core import registry

# Subjects whose ecogprep FIF files hold microvolts although the FIF unit says volts. The raw BIDS EDF
# of sub-01 has physical dimension "n/a" on every channel (sub-02..09: "uV"), so MNE kept its uV
# numbers and labelled them V, and ecogprep wrote them unchanged: median |x| ~23 "V" for sub-01 vs
# 2e-5..5e-5 V for the others, in both the broadband and the high-gamma file. Rescaling here makes
# every subject volts, so `signal_unit: uV` gives microvolts for all of them.
STORED_IN_MICROVOLTS = {1}


def _to_volts(raw, subject_id):
    if subject_id in STORED_IN_MICROVOLTS:
        raw.apply_function(lambda x: x * 1e-6, channel_wise=False)
    return raw


def _channels_tsv_candidates(file_path, data_root, subject_id):
    if not hasattr(file_path, "copy"):
        return []
    paths = []
    described = file_path.copy().update(suffix="channels", extension=".tsv")
    paths.append(str(described))
    if file_path.description:
        paths.append(
            str(
                file_path.copy().update(
                    description=None, suffix="channels", extension=".tsv"
                )
            )
        )
    paths.append(
        str(
            BIDSPath(
                root=data_root,
                subject=f"{subject_id:02}",
                task="podcast",
                datatype="ieeg",
                suffix="channels",
                extension=".tsv",
            )
        )
    )
    return paths


@registry.register_dataset("podcast")
def load_podcast_raw(subject_id, data_params):
    description = "highgamma" if data_params.use_high_gamma else None
    file_path = BIDSPath(
        root=os.path.join(data_params.data_root, "derivatives/ecogprep"),
        subject=f"{subject_id:02}",
        task="podcast",
        datatype="ieeg",
        description=description,
        suffix="ieeg",
        extension=".fif",
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    for path in _channels_tsv_candidates(file_path, data_params.data_root, subject_id):
        if not os.path.exists(path):
            continue
        channels = pd.read_csv(path, sep="\t")
        if {"status", "name"} <= set(channels):
            bads = channels.loc[channels["status"] == "bad", "name"].astype(str)
            raw.info["bads"] = list(dict.fromkeys(
                [*raw.info["bads"], *(ch for ch in bads if ch in raw.ch_names)]
            ))
        break
    return _to_volts(raw, subject_id)
