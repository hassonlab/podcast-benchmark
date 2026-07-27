"""Brain Treebank artifact loading, source conversion, and event alignment."""

from __future__ import annotations

import glob
import json
import os
import re
import warnings
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from core import registry


SOURCE_SFREQ = 2048.0
SOURCE_UNIT_SCALE_TO_VOLTS = 1e-6


def movie_slug(movie: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(movie)).strip("-").lower()
    if not slug:
        raise ValueError("Brain Treebank movie must contain a usable identifier")
    return slug


def metadata_records(data_root: str, subject_id: int) -> list[dict]:
    pattern = os.path.join(
        data_root, "subject_metadata", f"sub_{subject_id}_trial*_metadata.json"
    )
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as stream:
            metadata = json.load(stream)
        trial = int(Path(path).name.split("_trial", 1)[1].split("_", 1)[0])
        records.append(
            {
                "subject_id": subject_id,
                "trial_id": trial,
                "movie": metadata.get("filename"),
                "recording_path": os.path.join(
                    data_root, "all_subject_data", f"sub_{subject_id}_trial{trial:03d}.h5"
                ),
                "timings_path": os.path.join(
                    data_root,
                    "subject_timings",
                    f"sub_{subject_id}_trial{trial:03d}_timings.csv",
                ),
                "electrode_labels_path": os.path.join(
                    data_root,
                    "electrode_labels",
                    f"sub_{subject_id}",
                    "electrode_labels.json",
                ),
            }
        )
    return records


def resolve_movie_record(data_root: str, subject_id: int, movie: str) -> dict:
    matches = [
        record
        for record in metadata_records(data_root, subject_id)
        if record["movie"] == movie
        or (
            record["movie"]
            and movie_slug(record["movie"]) == movie_slug(movie)
        )
    ]
    if not matches:
        available = sorted(
            record["movie"] for record in metadata_records(data_root, subject_id)
            if record["movie"]
        )
        raise ValueError(
            f"Subject {subject_id} does not have Brain Treebank movie {movie!r}; "
            f"available={available}"
        )
    return matches[0]


def artifact_paths(data_root: str, subject_id: int, movie: str, representation: str):
    directory = Path(data_root) / "derivatives" / "podcast_benchmark" / f"sub-{subject_id}"
    directory /= "ieeg"
    stem = f"sub-{subject_id}_task-{movie_slug(movie)}"
    return (
        directory / f"{stem}_desc-{representation}_ieeg.fif",
        directory / f"{stem}_sync.tsv",
    )


def _electrode_labels(path: str) -> dict[int, str]:
    with open(path) as stream:
        labels = json.load(stream)
    if isinstance(labels, list):
        return {index: str(label) for index, label in enumerate(labels)}
    parsed = {}
    for key, value in labels.items():
        match = re.search(r"(\d+)$", str(key))
        if match:
            parsed[int(match.group(1))] = str(value)
    return parsed


def load_source_raw(record: dict):
    """Load one source HDF5 trial into an MNE RawArray in volts."""
    recording_path = record["recording_path"]
    if not os.path.exists(recording_path):
        raise FileNotFoundError(f"Brain Treebank recording not found: {recording_path}")
    labels = _electrode_labels(record["electrode_labels_path"])
    arrays, names = [], []
    with h5py.File(recording_path, "r") as stream:
        if "data" not in stream:
            raise ValueError(f"Recording has no 'data' group: {recording_path}")
        keys = []
        for key in stream["data"]:
            match = re.fullmatch(r"electrode_(\d+)", key)
            if match:
                keys.append((int(match.group(1)), key))
        if not keys:
            raise ValueError(f"Recording has no electrode_<n> arrays: {recording_path}")
        for electrode_num, key in sorted(keys):
            values = np.asarray(stream["data"][key], dtype=np.float32).squeeze()
            if values.ndim != 1:
                raise ValueError(f"Expected one-dimensional {key}, got {values.shape}")
            arrays.append(values)
            names.append(labels.get(electrode_num, key))
    data = np.stack(arrays) * SOURCE_UNIT_SCALE_TO_VOLTS
    info = mne.create_info(names, SOURCE_SFREQ, ch_types="seeg")
    info["subject_info"] = {"his_id": f"sub-{record['subject_id']}"}
    return mne.io.RawArray(data, info, verbose=False)


def mark_corrupted_channels(raw, data_root: str, subject_id: int):
    path = Path(data_root) / "corrupted_elec.json"
    if not path.exists():
        raise FileNotFoundError(f"Corrupted-electrode metadata not found: {path}")
    with path.open() as stream:
        by_subject = json.load(stream)
    key = f"sub_{subject_id}"
    if key not in by_subject:
        raise KeyError(f"Corrupted-electrode metadata has no {key!r} entry")
    raw.info["bads"] = [str(ch) for ch in by_subject[key] if str(ch) in raw.ch_names]
    return raw


def attach_mni_coordinates(raw, data_root: str, subject_id: int):
    path = Path(data_root) / "localization" / "elec_coords_full.csv"
    if not path.exists():
        warnings.warn(f"Brain Treebank localization not found: {path}", RuntimeWarning)
        return raw
    table = pd.read_csv(path)
    required = {"Subject", "Electrode", "X", "Y", "Z"}
    if not required <= set(table):
        raise ValueError(f"Localization file is missing {sorted(required - set(table))}")
    rows = table.loc[table["Subject"].astype(str) == f"sub_{subject_id}"]
    lookup = {}
    for row in rows.to_dict("records"):
        coords = np.asarray([row["X"], row["Y"], row["Z"]], dtype=float)
        if np.isfinite(coords).all():
            lookup[re.sub(r"[#*]", "", str(row["Electrode"]))] = coords / 1000.0
    positions = {
        name: lookup[re.sub(r"[#*]", "", name)]
        for name in raw.ch_names
        if re.sub(r"[#*]", "", name) in lookup
    }
    if positions:
        montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame="mni_tal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


def build_sync_table(timings_path: str) -> pd.DataFrame:
    timings = pd.read_csv(timings_path)
    required = {"movie_time", "index"}
    if not required <= set(timings):
        raise ValueError(f"Timing file is missing {sorted(required - set(timings))}")
    result = pd.DataFrame(
        {
            "movie_time": pd.to_numeric(timings["movie_time"], errors="coerce"),
            "raw_time": pd.to_numeric(timings["index"], errors="coerce") / SOURCE_SFREQ,
        }
    ).dropna()
    if result.empty:
        raise ValueError(f"Timing file contains no usable rows: {timings_path}")
    return result.sort_values("movie_time").reset_index(drop=True)


def _brain_treebank_event_times(task_df, subject_id, data_params):
    movie = data_params.dataset_params.get("movie")
    if not movie:
        raise ValueError("Brain Treebank requires data_params.dataset_params.movie")
    _, sync_path = artifact_paths(
        data_params.data_root, subject_id, movie, "highgamma" if data_params.use_high_gamma else "broadband"
    )
    if not sync_path.exists():
        raise FileNotFoundError(f"Brain Treebank sync artifact not found: {sync_path}")
    sync = pd.read_csv(sync_path, sep="\t")
    movie_times = sync["movie_time"].to_numpy(dtype=float)
    raw_times = sync["raw_time"].to_numpy(dtype=float)
    starts = task_df["start"].to_numpy(dtype=float)
    nearest = np.abs(starts[:, None] - movie_times[None, :]).argmin(axis=1)
    return raw_times[nearest] + starts - movie_times[nearest]


@registry.register_dataset(
    "brain_treebank", event_time_getter=_brain_treebank_event_times
)
def load_brain_treebank_raw(subject_id, data_params):
    movie = data_params.dataset_params.get("movie")
    if not movie:
        raise ValueError("Brain Treebank requires data_params.dataset_params.movie")
    resolve_movie_record(data_params.data_root, subject_id, movie)
    representation = "highgamma" if data_params.use_high_gamma else "broadband"
    raw_path, _ = artifact_paths(
        data_params.data_root, subject_id, movie, representation
    )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Brain Treebank {representation} artifact not found: {raw_path}. "
            "Run scripts/preprocess_brain_treebank.py first."
        )
    return mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
