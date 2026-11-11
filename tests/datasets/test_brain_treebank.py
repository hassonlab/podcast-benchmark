import json

import h5py
import mne
import numpy as np
import pandas as pd

import datasets  # noqa: F401
from core.config import DataParams
from datasets.brain_treebank import artifact_paths, build_sync_table, load_source_raw
from utils.data_utils import get_event_times, load_raws


def test_sync_table_and_subject_event_mapping(tmp_path):
    timings = tmp_path / "timings.csv"
    pd.DataFrame({"movie_time": [0.0, 10.0], "index": [2048, 22528]}).to_csv(
        timings, index=False
    )
    sync = build_sync_table(str(timings))
    assert sync["raw_time"].tolist() == [1.0, 11.0]

    raw_path, sync_path = artifact_paths(
        str(tmp_path), subject_id=3, movie="Cars-2", representation="broadband"
    )
    raw_path.parent.mkdir(parents=True)
    raw = mne.io.RawArray(
        np.zeros((2, 2000)),
        mne.create_info(["A", "B"], 100.0, ch_types="seeg"),
        verbose=False,
    )
    raw.save(raw_path, overwrite=True, verbose=False)
    sync.to_csv(sync_path, sep="\t", index=False)

    metadata_dir = tmp_path / "subject_metadata"
    metadata_dir.mkdir()
    (metadata_dir / "sub_3_trial001_metadata.json").write_text(
        json.dumps({"filename": "Cars-2"})
    )
    params = DataParams(
        dataset_name="brain_treebank",
        dataset_params={"movie": "Cars-2"},
        data_root=str(tmp_path),
        subject_ids=[3],
        use_high_gamma=False,
    )
    loaded = load_raws(params)
    event_times = get_event_times(
        pd.DataFrame({"start": [2.0, 9.0], "target": [0, 1]}), params
    )

    assert loaded[0].ch_names == ["A", "B"]
    np.testing.assert_allclose(event_times[0], [3.0, 10.0])


def test_load_source_hdf5_uses_labels_and_converts_microvolts(tmp_path):
    recording = tmp_path / "recording.h5"
    with h5py.File(recording, "w") as stream:
        data = stream.create_group("data")
        data.create_dataset("electrode_0", data=np.array([1.0, 2.0]))
        data.create_dataset("electrode_1", data=np.array([3.0, 4.0]))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["A1", "A2"]))

    raw = load_source_raw(
        {
            "recording_path": str(recording),
            "electrode_labels_path": str(labels),
            "subject_id": 8,
        }
    )

    assert raw.ch_names == ["A1", "A2"]
    assert raw.info["subject_info"]["his_id"] == "sub-8"
    np.testing.assert_allclose(
        raw.get_data(), np.array([[1.0, 2.0], [3.0, 4.0]]) * 1e-6
    )


def test_load_source_hdf5_can_resample_while_reading(tmp_path):
    recording = tmp_path / "recording.h5"
    samples = np.sin(2 * np.pi * 10 * np.arange(2048) / 2048)
    with h5py.File(recording, "w") as stream:
        data = stream.create_group("data")
        data.create_dataset("electrode_0", data=samples)
        data.create_dataset("electrode_1", data=samples * 2)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["A1", "A2"]))

    raw = load_source_raw(
        {
            "recording_path": str(recording),
            "electrode_labels_path": str(labels),
            "subject_id": 8,
        },
        target_sfreq=512,
    )

    assert raw.info["sfreq"] == 512
    assert raw.n_times == 512
