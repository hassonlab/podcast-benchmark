"""Podcast MNE Raw loader."""

import os

import mne
import pandas as pd
from mne_bids import BIDSPath

from core import registry


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
    return raw
