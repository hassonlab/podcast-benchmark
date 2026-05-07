"""
Tests for utils/dataset.py.

Tests dataset utilities for handling lagged raw slicing and dictionary inputs.
"""

import mne
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from utils.dataset import NeuralDictDataset, RawNeuralDataset, _apply_preprocessing


def reference_get_data(
    lag,
    raws: list[mne.io.Raw],
    task_df: pd.DataFrame,
    window_width: float,
    preprocessing_fns=None,
    preprocessor_params: dict = None,
    return_subject_channel_counts: bool = False,
):
    """Reference copy of the pre-refactor get_data implementation."""
    datas = []
    selected_rows_df = None
    selected_targets = None

    for raw in raws:
        tmin = lag / 1000 - window_width / 2
        tmax = lag / 1000 + window_width / 2 - 2e-3
        data_duration = raw.times[-1]

        valid_mask = (task_df.start + tmin >= 0) & (
            task_df.start + tmax <= data_duration
        )
        task_df_valid = task_df[valid_mask].reset_index(drop=True)

        if len(task_df_valid) == 0:
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
            event_repeated="merge",
            verbose="ERROR",
        )

        data = epochs.get_data(copy=False)
        selected_rows_df = task_df_valid.iloc[epochs.selection]
        selected_targets = selected_rows_df.target.to_numpy()

        assert data.shape[0] == selected_rows_df.shape[0], "Sample counts don't match"
        datas.append(data)

    if len(datas) == 0:
        raise ValueError("No valid events found within data time bounds")

    subject_channel_counts = [d.shape[1] for d in datas]
    datas = np.concatenate(datas, axis=1)
    datas = _apply_preprocessing(datas, preprocessing_fns, preprocessor_params)

    if return_subject_channel_counts:
        return datas, selected_targets, selected_rows_df, subject_channel_counts

    return datas, selected_targets, selected_rows_df


@pytest.fixture
def mock_raw():
    """Create a mock MNE Raw object with 10 channels and 10 seconds of data at 1000Hz."""
    n_channels = 10
    sfreq = 1000
    duration = 10
    n_samples = int(sfreq * duration)

    data = np.random.randn(n_channels, n_samples)
    ch_names = [f"CH{i:02d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")

    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture
def mock_raw_pair():
    """Create two raws with different channel counts but aligned time bases."""
    sfreq = 1000
    duration = 10
    n_samples = int(sfreq * duration)

    info_a = mne.create_info(ch_names=["A1", "A2", "A3"], sfreq=sfreq, ch_types="seeg")
    info_b = mne.create_info(
        ch_names=["B1", "B2", "B3", "B4"], sfreq=sfreq, ch_types="seeg"
    )
    raw_a = mne.io.RawArray(np.random.randn(3, n_samples), info_a, verbose=False)
    raw_b = mne.io.RawArray(np.random.randn(4, n_samples), info_b, verbose=False)
    return [raw_a, raw_b]


@pytest.fixture
def task_df_in_bounds():
    """Create a DataFrame with word events that fall within the data bounds."""
    return pd.DataFrame(
        {
            "start": [1.0, 2.0, 3.0],
            "end": [1.5, 2.5, 3.5],
            "word": ["hello", "world", "test"],
            "target": [0, 1, 2],
        }
    )


@pytest.fixture
def task_df_out_of_bounds():
    """Create a DataFrame with word events that fall outside the data bounds."""
    return pd.DataFrame(
        {
            "start": [9.8, 9.9, 10.1],
            "end": [9.9, 10.0, 10.2],
            "word": ["late", "very_late", "too_late"],
            "target": [0, 1, 2],
        }
    )


class TestApplyPreprocessing:
    def test_returns_input_when_no_preprocessing_fns(self):
        data = np.arange(6, dtype=float).reshape(1, 2, 3)

        result = _apply_preprocessing(data, None, None)

        np.testing.assert_array_equal(result, data)

    def test_applies_multiple_preprocessors_with_list_params(self):
        data = np.arange(6, dtype=float).reshape(1, 2, 3)

        def add_offset(x, params):
            return x + params["offset"]

        def scale(x, params):
            return x * params["factor"]

        result = _apply_preprocessing(
            data,
            [add_offset, scale],
            [{"offset": 1.5}, {"factor": 2.0}],
        )

        np.testing.assert_allclose(result, (data + 1.5) * 2.0)


class TestRawNeuralDataset:
    """Test RawNeuralDataset for correct window filtering and lag slicing."""

    def test_in_bounds_baseline(self, mock_raw, task_df_in_bounds):
        """Dataset keeps all events when they are in bounds for the given lag."""
        window_width = 0.5
        ds = RawNeuralDataset([mock_raw], task_df_in_bounds, window_width)
        neural, targets, df, _ = ds.get_data_for_lag(0)

        assert neural.shape[0] == len(task_df_in_bounds)
        assert targets.shape[0] == len(task_df_in_bounds)
        assert len(df) == len(task_df_in_bounds)

    def test_correct_window_shape(self, mock_raw, task_df_in_bounds):
        """get_data_for_lag returns the expected time-axis length."""
        sfreq = mock_raw.info["sfreq"]
        window_width = 0.5
        ds = RawNeuralDataset([mock_raw], task_df_in_bounds, window_width)
        neural, _, _, _ = ds.get_data_for_lag(0)

        expected_samples = int(round((window_width - 2e-3) * sfreq)) + 1
        assert neural.shape[2] == expected_samples

    def test_out_of_bounds_raises(self, mock_raw, task_df_out_of_bounds):
        """Dataset raises ValueError when no events are valid for the window."""
        window_width = 1.0
        ds = RawNeuralDataset([mock_raw], task_df_out_of_bounds, window_width)
        with pytest.raises(
            ValueError, match="No valid events found within data time bounds"
        ):
            ds.get_data_for_lag(500)

    def test_extreme_out_of_bounds_raises(self, mock_raw):
        """Dataset raises ValueError for completely out-of-range events."""
        task_df_extreme = pd.DataFrame(
            {
                "start": [-3.0, 20.0, 25.0],
                "end": [15.5, 20.5, 25.5],
                "word": ["way", "too", "late"],
                "target": [0, 1, 2],
            }
        )
        ds = RawNeuralDataset([mock_raw], task_df_extreme, 0.5)
        with pytest.raises(
            ValueError, match="No valid events found within data time bounds"
        ):
            ds.get_data_for_lag(0)

    def test_mixed_bounds_filters_correctly(self, mock_raw):
        """Only in-bounds events are retained when the DataFrame has a mix."""
        task_df_mixed = pd.DataFrame(
            {
                "start": [2.0, 9.8, 4.0, 10.1],
                "end": [2.5, 9.9, 4.5, 10.2],
                "word": ["good", "late", "okay", "too_late"],
                "target": [0, 1, 2, 3],
            }
        )
        window_width = 1.0
        ds = RawNeuralDataset([mock_raw], task_df_mixed, window_width)
        neural, targets, df, _ = ds.get_data_for_lag(500)

        expected_valid = 2
        assert neural.shape[0] == expected_valid
        assert targets.shape[0] == expected_valid
        assert len(df) == expected_valid

    def test_negative_time_bounds_raises(self, mock_raw):
        """Events whose window goes before t=0 are filtered; raises if none remain."""
        task_df_early = pd.DataFrame(
            {
                "start": [0.1, 0.2, 0.3],
                "end": [0.2, 0.3, 0.4],
                "word": ["early1", "early2", "early3"],
                "target": [0, 1, 2],
            }
        )
        ds = RawNeuralDataset([mock_raw], task_df_early, 0.5)
        with pytest.raises(
            ValueError, match="No valid events found within data time bounds"
        ):
            ds.get_data_for_lag(-1000)

    def test_mixed_negative_bounds_filters(self, mock_raw):
        """Mix of valid and early-start events: only valid rows are kept."""
        task_df_mixed_neg = pd.DataFrame(
            {
                "start": [0.7, 2.0, 3.0, 0.2],
                "end": [1.1, 2.1, 3.1, 0.3],
                "word": ["valid1", "valid2", "valid3", "early"],
                "target": [0, 1, 2, 3],
            }
        )
        window_width = 0.4
        ds = RawNeuralDataset([mock_raw], task_df_mixed_neg, window_width)
        neural, targets, df, _ = ds.get_data_for_lag(-500)

        expected_valid = 3
        assert neural.shape[0] == expected_valid
        assert targets.shape[0] == expected_valid
        assert len(df) == expected_valid

    def test_multiple_lags_consistent_rows(self, mock_raw, task_df_in_bounds):
        """All lags return the same number of rows when they stay in bounds."""
        lags = [-200, 0, 200]
        window_width = 0.5
        ds = RawNeuralDataset([mock_raw], task_df_in_bounds, window_width)

        n_words = None
        for lag in lags:
            neural, targets, df, _ = ds.get_data_for_lag(lag)
            if n_words is None:
                n_words = neural.shape[0]
            assert neural.shape[0] == n_words
            assert targets.shape[0] == n_words
            assert len(df) == n_words

    def test_returns_float_tensors(self, mock_raw, task_df_in_bounds):
        """get_data_for_lag returns torch.FloatTensor objects."""
        ds = RawNeuralDataset([mock_raw], task_df_in_bounds, 0.5)
        neural, targets, _, _ = ds.get_data_for_lag(0)
        assert neural.dtype == torch.float32
        assert targets.dtype == torch.float32

    def test_matches_reference_get_data_single_subject(
        self, mock_raw, task_df_in_bounds
    ):
        """The fast dataset path matches the reference get_data output."""
        lag = 0
        window_width = 0.5

        expected_neural, expected_targets, expected_df = reference_get_data(
            lag, [mock_raw], task_df_in_bounds, window_width
        )
        ds = RawNeuralDataset([mock_raw], task_df_in_bounds, window_width)
        actual_neural, actual_targets, actual_df, _ = ds.get_data_for_lag(lag)

        np.testing.assert_allclose(actual_neural.numpy(), expected_neural)
        np.testing.assert_allclose(
            actual_targets.numpy(), expected_targets.astype(np.float32)
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_matches_reference_get_data_multiple_subjects_and_channel_counts(
        self, mock_raw_pair, task_df_in_bounds
    ):
        """Multi-subject concatenation matches the reference function."""
        lag = 200
        window_width = 0.4

        expected_neural, expected_targets, expected_df, expected_counts = (
            reference_get_data(
                lag,
                mock_raw_pair,
                task_df_in_bounds,
                window_width,
                return_subject_channel_counts=True,
            )
        )
        ds = RawNeuralDataset(mock_raw_pair, task_df_in_bounds, window_width)
        actual_neural, actual_targets, actual_df, actual_counts = ds.get_data_for_lag(
            lag
        )

        np.testing.assert_allclose(actual_neural.numpy(), expected_neural)
        np.testing.assert_allclose(
            actual_targets.numpy(), expected_targets.astype(np.float32)
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)
        assert actual_counts == expected_counts
        assert ds.subject_channel_counts == expected_counts

    def test_matches_reference_selected_rows_with_duplicate_onsets(self, mock_raw):
        """Duplicate onsets should produce the same selected rows as the reference path."""
        task_df = pd.DataFrame(
            {
                "start": [1.0, 1.0, 2.0],
                "end": [1.1, 1.1, 2.1],
                "word": ["dup_a", "dup_b", "later"],
                "target": [10, 11, 12],
            }
        )
        lag = 0
        window_width = 0.2

        expected_neural, expected_targets, expected_df = reference_get_data(
            lag, [mock_raw], task_df, window_width
        )
        ds = RawNeuralDataset([mock_raw], task_df, window_width)
        actual_neural, actual_targets, actual_df, _ = ds.get_data_for_lag(lag)

        np.testing.assert_allclose(actual_neural.numpy(), expected_neural)
        np.testing.assert_allclose(
            actual_targets.numpy(), expected_targets.astype(np.float32)
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_matches_reference_row_filtering_per_lag(self, mock_raw):
        """Each lag should filter rows the same way as the reference function."""
        task_df = pd.DataFrame(
            {
                "start": [0.3, 1.0, 9.6],
                "end": [0.4, 1.1, 9.7],
                "word": ["early", "middle", "late"],
                "target": [1, 2, 3],
            }
        )
        lags = [-200, 200]
        window_width = 0.4
        ds = RawNeuralDataset([mock_raw], task_df, window_width)

        for lag in lags:
            expected_neural, expected_targets, expected_df = reference_get_data(
                lag, [mock_raw], task_df, window_width
            )
            actual_neural, actual_targets, actual_df, _ = ds.get_data_for_lag(lag)

            np.testing.assert_allclose(actual_neural.numpy(), expected_neural)
            np.testing.assert_allclose(
                actual_targets.numpy(), expected_targets.astype(np.float32)
            )
            pd.testing.assert_frame_equal(actual_df, expected_df)


@pytest.fixture
def basic_dataset():
    """Create a basic dataset with two features."""
    neural_data = torch.randn(10, 15)
    input_dict = {
        "feature1": torch.randn(10, 5),
        "feature2": torch.randn(10, 3),
    }
    target = torch.randn(10, 2)
    return NeuralDictDataset(neural_data, input_dict, target)


@pytest.fixture
def scalar_target_dataset():
    """Create a dataset with scalar targets."""
    neural_data = torch.randn(10, 15)
    input_dict = {
        "x": torch.randn(5, 10),
        "y": torch.randn(5, 20),
    }
    target = torch.randn(5)
    return NeuralDictDataset(neural_data, input_dict, target)


def test_dataset_length(basic_dataset):
    """Test that dataset returns correct length."""
    assert len(basic_dataset) == 10


def test_getitem_returns_tuple(basic_dataset):
    """Test that __getitem__ returns a tuple of (neural_data, dict, target)."""
    item = basic_dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 3
    assert isinstance(item[0], torch.Tensor)  # neural_data
    assert isinstance(item[1], dict)  # input_dict
    assert isinstance(item[2], torch.Tensor)  # target


def test_getitem_dict_keys(basic_dataset):
    """Test that returned dict contains all expected keys."""
    _, item_dict, _ = basic_dataset[0]
    assert "feature1" in item_dict
    assert "feature2" in item_dict
    assert len(item_dict) == 2


def test_getitem_tensor_shapes(basic_dataset):
    """Test that indexed tensors have correct shapes."""
    neural_data, item_dict, label = basic_dataset[0]
    assert neural_data.shape == (15,)
    assert item_dict["feature1"].shape == (5,)
    assert item_dict["feature2"].shape == (3,)
    assert label.shape == (2,)


def test_getitem_different_indices(basic_dataset):
    """Test that different indices return different values."""
    neural_data_1, item1_dict, label1 = basic_dataset[0]
    neural_data_2, item2_dict, label2 = basic_dataset[5]

    # Values should be different (with high probability for random data)
    assert not torch.allclose(neural_data_1, neural_data_2)
    assert not torch.allclose(item1_dict["feature1"], item2_dict["feature1"])
    assert not torch.allclose(label1, label2)


def test_scalar_target(scalar_target_dataset):
    """Test dataset with scalar targets."""
    _, item_dict, label = scalar_target_dataset[2]
    assert label.shape == ()  # Scalar tensor
    assert item_dict["x"].shape == (10,)
    assert item_dict["y"].shape == (20,)


def test_mismatched_input_lengths_raises_error():
    """Test that mismatched input tensor lengths raise ValueError."""
    neural_data = torch.randn(10, 15)
    bad_input = {
        "feature1": torch.randn(10, 5),
        "feature2": torch.randn(8, 3),  # Wrong length
    }
    target = torch.randn(10)

    with pytest.raises(ValueError, match="same length"):
        NeuralDictDataset(neural_data, bad_input, target)


def test_mismatched_target_length_raises_error():
    """Test that mismatched target length raises ValueError."""
    neural_data = torch.randn(10, 15)
    input_dict = {
        "feature1": torch.randn(10, 5),
    }
    target = torch.randn(8)  # Wrong length

    with pytest.raises(ValueError, match="same length"):
        NeuralDictDataset(neural_data, input_dict, target)


def test_dataloader_compatibility(basic_dataset):
    """Test that dataset works with PyTorch DataLoader."""
    dataloader = DataLoader(basic_dataset, batch_size=3, shuffle=True)

    batch_neural_data, batch_dict, batch_target = next(iter(dataloader))

    assert isinstance(batch_dict, dict)
    assert batch_neural_data.shape[0] <= 3  # Batch size
    assert batch_neural_data.shape == (batch_neural_data.shape[0], 15)
    assert batch_dict["feature1"].shape[0] <= 3  # Batch size
    assert batch_dict["feature1"].shape == (batch_dict["feature1"].shape[0], 5)
    assert batch_dict["feature2"].shape == (batch_dict["feature2"].shape[0], 3)
    assert batch_target.shape == (batch_dict["feature1"].shape[0], 2)


def test_dataloader_multiple_batches(basic_dataset):
    """Test that DataLoader iterates over all samples."""
    dataloader = DataLoader(basic_dataset, batch_size=3, shuffle=False)

    total_samples = 0
    for batch_neural_data, batch_dict, batch_target in dataloader:
        total_samples += batch_dict["feature1"].shape[0]
        # Verify each batch has correct structure
        assert batch_neural_data.shape[0] == batch_dict["feature1"].shape[0]
        assert isinstance(batch_dict, dict)
        assert "feature1" in batch_dict
        assert "feature2" in batch_dict

    assert total_samples == 10


def test_dataloader_last_batch(basic_dataset):
    """Test that DataLoader handles the last incomplete batch correctly."""
    dataloader = DataLoader(basic_dataset, batch_size=3, shuffle=False)

    batches = list(dataloader)
    assert len(batches) == 4  # 10 samples / 3 batch_size = 4 batches (3+3+3+1)

    # Last batch should have 1 sample
    last_batch_neural_data, last_batch_dict, last_batch_target = batches[-1]
    assert last_batch_neural_data.shape[0] == 1
    assert last_batch_dict["feature1"].shape[0] == 1
    assert last_batch_target.shape[0] == 1


def test_empty_input_dict():
    """Test behavior with empty input dictionary."""
    neural_data = torch.randn(7, 15)
    input_dict = {}
    target = torch.randn(5)

    dataset = NeuralDictDataset(neural_data, input_dict, target)
    assert len(dataset) == 5

    ret_neural_data, item_dict, label = dataset[0]
    assert ret_neural_data.shape == (15,)
    assert isinstance(item_dict, dict)
    assert len(item_dict) == 0
    assert label.shape == ()


def test_single_feature():
    """Test dataset with only one feature."""
    neural_data = torch.randn(7, 15)
    input_dict = {
        "single_feature": torch.randn(7, 4),
    }
    target = torch.randn(7, 1)

    dataset = NeuralDictDataset(neural_data, input_dict, target)
    ret_neural_data, item_dict, label = dataset[3]

    assert len(item_dict) == 1
    assert ret_neural_data.shape == (15,)
    assert item_dict["single_feature"].shape == (4,)
    assert label.shape == (1,)
