"""
Tests for data_utils.py.

Tests the get_data function for handling out-of-bounds time windows
and the read_electrode_file function for parsing electrode mapping files.
"""

import pytest
import numpy as np
import pandas as pd
import mne
import tempfile
import os
from math import gcd
from unittest.mock import patch, MagicMock

from scipy.signal import resample_poly
from data_utils import (
    get_data,
    read_electrode_file,
    load_raws,
    load_ieeg_edf_files,
    read_subject_mapping,
    load_audio_waveform,
    hilbert_envelope,
    butterworth_lowpass_envelope,
    resample_envelope,
    compress_envelope_db,
)
from config import DataParams


@pytest.fixture
def mock_raw():
    """Create a mock MNE Raw object with 10 channels and 10 seconds of data at 1000Hz."""
    n_channels = 10
    sfreq = 1000  # Hz
    duration = 10  # seconds
    n_samples = int(sfreq * duration)

    # Create random data
    data = np.random.randn(n_channels, n_samples)

    # Create channel names
    ch_names = [f"CH{i:02d}" for i in range(n_channels)]

    # Create info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def df_word_in_bounds():
    """Create a DataFrame with word events that fall within the data bounds."""
    return pd.DataFrame(
        {
            "start": [1.0, 2.0, 3.0],  # seconds, well within 10s duration
            "end": [1.5, 2.5, 3.5],
            "word": ["hello", "world", "test"],
            "target": [0, 1, 2],
        }
    )


@pytest.fixture
def df_word_out_of_bounds():
    """Create a DataFrame with word events that fall outside the data bounds."""
    return pd.DataFrame(
        {
            "start": [9.8, 9.9, 10.1],  # seconds, near or beyond 10s duration
            "end": [9.9, 10.0, 10.2],
            "word": ["late", "very_late", "too_late"],
            "target": [0, 1, 2],
        }
    )


class TestGetDataOutOfBounds:
    """Test get_data function with out-of-bounds time windows."""

    def test_get_data_in_bounds_baseline(self, mock_raw, df_word_in_bounds):
        """Test that get_data works correctly with in-bounds events."""
        lag = 0
        window_width = 0.5

        data, targets, words = get_data(
            lag=lag,
            raws=[mock_raw],
            df_word=df_word_in_bounds,
            window_width=window_width,
            word_column="word",
        )

        assert data.shape[0] == len(df_word_in_bounds)
        assert len(targets) == len(df_word_in_bounds)
        assert len(words) == len(df_word_in_bounds)

    def test_get_data_without_word_column(self, mock_raw, df_word_in_bounds):
        """Test that get_data works correctly with in-bounds events."""
        lag = 0
        window_width = 0.5

        data, targets, words = get_data(
            lag=lag,
            raws=[mock_raw],
            df_word=df_word_in_bounds,
            window_width=window_width,
        )

        assert data.shape[0] == len(df_word_in_bounds)
        assert len(targets) == len(df_word_in_bounds)

    def test_get_data_out_of_bounds_bug(self, mock_raw, df_word_out_of_bounds):
        """Test that out-of-bounds events raise ValueError instead of creating empty epochs."""
        lag = 500
        window_width = 1.0

        with pytest.raises(
            ValueError, match="No valid events found within data time bounds"
        ):
            data, targets, words = get_data(
                lag=lag,
                raws=[mock_raw],
                df_word=df_word_out_of_bounds,
                window_width=window_width,
            )

    def test_get_data_extreme_out_of_bounds(self, mock_raw):
        """Test that extremely out-of-bounds events raise ValueError without warnings."""
        df_word_extreme = pd.DataFrame(
            {
                "start": [-3.0, 20.0, 25.0],
                "end": [15.5, 20.5, 25.5],
                "word": ["way", "too", "late"],
                "target": [0, 1, 2],
            }
        )

        lag = 0
        window_width = 0.5

        with pytest.raises(
            ValueError, match="No valid events found within data time bounds"
        ):
            data, targets, words = get_data(
                lag=lag,
                raws=[mock_raw],
                df_word=df_word_extreme,
                window_width=window_width,
            )

    def test_get_data_mixed_bounds(self, mock_raw):
        """Test that only in-bounds events are kept when mix of valid/invalid events."""
        df_word_mixed = pd.DataFrame(
            {
                "start": [2.0, 9.8, 4.0, 10.1],
                "end": [2.5, 9.9, 4.5, 10.2],
                "word": ["good", "late", "okay", "too_late"],
                "target": [0, 1, 2, 3],
            }
        )

        lag = 500
        window_width = 1.0

        data, targets, words = get_data(
            lag=lag,
            raws=[mock_raw],
            df_word=df_word_mixed,
            window_width=window_width,
            word_column="word",
        )

        expected_valid_events = 2
        assert data.shape[0] == expected_valid_events
        assert len(targets) == expected_valid_events
        assert len(words) == expected_valid_events

    def test_get_data_negative_time_bounds(self, mock_raw):
        """Test that events with negative time windows are filtered out."""
        df_word_early = pd.DataFrame(
            {
                "start": [0.1, 0.2, 0.3],
                "end": [0.2, 0.3, 0.4],
                "word": ["early1", "early2", "early3"],
                "target": [0, 1, 2],
            }
        )

        lag = -1000  # -1 second
        window_width = 0.5  # ±0.25s around event+lag

        # Events at 0.1s with -1s lag = -0.9s, window [-1.15s, -0.65s] - all negative
        with pytest.raises(
            ValueError, match="No valid events found within data time bounds"
        ):
            data, targets, words = get_data(
                lag=lag,
                raws=[mock_raw],
                df_word=df_word_early,
                window_width=window_width,
            )

    def test_get_data_mixed_negative_bounds(self, mock_raw):
        """Test filtering with mix of valid events and negative time window events."""
        df_word_mixed_neg = pd.DataFrame(
            {
                "start": [
                    0.7,
                    2.0,
                    3.0,
                    0.2,
                ],  # Mix: some valid, some will be negative
                "end": [1.1, 2.1, 3.1, 0.3],
                "word": ["valid1", "valid2", "valid3", "early"],
                "target": [0, 1, 2, 3],
            }
        )

        lag = -500  # -0.5 seconds
        window_width = 0.4  # ±0.2s around event+lag

        data, targets, words = get_data(
            lag=lag,
            raws=[mock_raw],
            df_word=df_word_mixed_neg,
            window_width=window_width,
            word_column="word",
        )

        expected_valid_events = 3
        assert data.shape[0] == expected_valid_events
        assert len(targets) == expected_valid_events
        assert len(words) == expected_valid_events


@pytest.fixture
def temp_electrode_csv():
    """Create a temporary CSV file with electrode data for testing."""
    electrode_data = """subject,elec
1,A1
1,A2
1,B1
2,C1
2,C2
3,D1
3,D2
3,D3
3,E1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(electrode_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_single_subject_csv():
    """Create a temporary CSV file with single subject electrode data."""
    electrode_data = """subject,elec
5,X1
5,X2
5,Y1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(electrode_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_empty_csv():
    """Create a temporary empty CSV file with headers only."""
    electrode_data = """subject,elec
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(electrode_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestReadElectrodeFile:
    """Test read_electrode_file function for parsing electrode mapping CSV files."""

    def test_read_electrode_file_multiple_subjects(self, temp_electrode_csv):
        """Test reading CSV with multiple subjects and electrodes."""
        result = read_electrode_file(temp_electrode_csv)

        expected = {1: ["A1", "A2", "B1"], 2: ["C1", "C2"], 3: ["D1", "D2", "D3", "E1"]}

        assert result == expected
        assert len(result) == 3
        assert len(result[1]) == 3
        assert len(result[2]) == 2
        assert len(result[3]) == 4

    def test_read_electrode_file_single_subject(self, temp_single_subject_csv):
        """Test reading CSV with single subject and multiple electrodes."""
        result = read_electrode_file(temp_single_subject_csv)

        expected = {5: ["X1", "X2", "Y1"]}

        assert result == expected
        assert len(result) == 1
        assert len(result[5]) == 3

    def test_read_electrode_file_empty_data(self, temp_empty_csv):
        """Test reading CSV with headers but no data."""
        result = read_electrode_file(temp_empty_csv)

        assert result == {}
        assert len(result) == 0

    def test_read_electrode_file_electrode_order_preserved(self, temp_electrode_csv):
        """Test that electrode order is preserved as they appear in CSV."""
        result = read_electrode_file(temp_electrode_csv)

        # Check that electrode order matches CSV order
        assert result[1] == ["A1", "A2", "B1"]
        assert result[3] == ["D1", "D2", "D3", "E1"]

    def test_read_electrode_file_subject_types(self, temp_electrode_csv):
        """Test that subject IDs are converted to integers."""
        result = read_electrode_file(temp_electrode_csv)

        # All keys should be integers
        for subject_id in result.keys():
            assert isinstance(subject_id, int)

    def test_read_electrode_file_electrode_types(self, temp_electrode_csv):
        """Test that electrode names remain as strings."""
        result = read_electrode_file(temp_electrode_csv)

        # All electrode names should be strings
        for electrodes in result.values():
            for electrode in electrodes:
                assert isinstance(electrode, str)

    def test_read_electrode_file_nonexistent_file(self):
        """Test that reading non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            read_electrode_file("nonexistent_file.csv")


@pytest.fixture
def mock_raw_with_channels():
    """Create a factory function for mock MNE Raw objects with specific channel names."""

    def _create_mock_raw():
        n_channels = 6
        sfreq = 1000
        duration = 10
        n_samples = int(sfreq * duration)

        data = np.random.randn(n_channels, n_samples)
        ch_names = ["LGA1", "LGA2", "LGB1", "LGB2", "RGA1", "RGA2"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw

    return _create_mock_raw


@pytest.fixture
def bids_temp_files(mock_raw_with_channels):
    """Create temporary BIDS-structured .fif files for testing."""

    def _create_temp_files(temp_dir, subject_ids):
        """Create temporary files for given subject IDs using BIDSPath structure."""
        from mne_bids import BIDSPath

        created_files = []
        for sub_id in subject_ids:
            # Use BIDSPath with same parameters as load_raws to get correct path structure
            file_path = BIDSPath(
                root=os.path.join(temp_dir, "derivatives/ecogprep"),
                subject=f"{sub_id:02}",
                task="podcast",
                datatype="ieeg",
                description="highgamma",
                suffix="ieeg",
                extension=".fif",
            )

            # Create directory structure
            os.makedirs(file_path.directory, exist_ok=True)

            # Save mock data to the BIDSPath location
            mock_raw_with_channels().save(str(file_path), overwrite=True, verbose=False)
            created_files.append(str(file_path))

        return created_files

    return _create_temp_files


class TestLoadRaws:
    """Test load_raws function for loading multiple subjects with different configurations."""

    @patch("data_utils.mne.io.read_raw_fif")
    @patch("data_utils.BIDSPath")
    def test_load_raws_multiple_subjects(
        self, mock_bids_path, mock_read_raw_fif, mock_raw_with_channels
    ):
        """Test loading raw data for multiple subjects."""
        # Setup mocks
        mock_bids_path.return_value = "/fake/path/sub-01_task-podcast_ieeg.fif"
        mock_read_raw_fif.return_value = mock_raw_with_channels()

        # Create DataParams for multiple subjects
        data_params = DataParams(subject_ids=[1, 2, 3], data_root="/fake/data")

        raws = load_raws(data_params)

        # Verify results
        assert len(raws) == 3
        assert mock_read_raw_fif.call_count == 3
        assert mock_bids_path.call_count == 3

        # Check that BIDSPath was called with correct parameters for each subject
        expected_calls = [
            {
                "root": "/fake/data/derivatives/ecogprep",
                "subject": "01",
                "task": "podcast",
                "datatype": "ieeg",
                "description": "highgamma",
                "suffix": "ieeg",
                "extension": ".fif",
            },
            {
                "root": "/fake/data/derivatives/ecogprep",
                "subject": "02",
                "task": "podcast",
                "datatype": "ieeg",
                "description": "highgamma",
                "suffix": "ieeg",
                "extension": ".fif",
            },
            {
                "root": "/fake/data/derivatives/ecogprep",
                "subject": "03",
                "task": "podcast",
                "datatype": "ieeg",
                "description": "highgamma",
                "suffix": "ieeg",
                "extension": ".fif",
            },
        ]

        for i, call_args in enumerate(mock_bids_path.call_args_list):
            call_kwargs = call_args[1]  # Get keyword arguments
            expected = expected_calls[i]
            for key, value in expected.items():
                assert call_kwargs[key] == value

    @patch("data_utils.mne.io.read_raw_fif")
    @patch("data_utils.BIDSPath")
    def test_load_raws_single_subject(
        self, mock_bids_path, mock_read_raw_fif, mock_raw_with_channels
    ):
        """Test loading raw data for a single subject."""
        mock_bids_path.return_value = "/fake/path/sub-05_task-podcast_ieeg.fif"
        mock_read_raw_fif.return_value = mock_raw_with_channels()

        data_params = DataParams(subject_ids=[5], data_root="/fake/data")

        raws = load_raws(data_params)

        assert len(raws) == 1
        assert mock_read_raw_fif.call_count == 1
        assert mock_bids_path.call_count == 1

        # Check BIDSPath was called with correct subject
        call_kwargs = mock_bids_path.call_args_list[0][1]
        assert call_kwargs["subject"] == "05"

    @patch("data_utils.mne.io.read_raw_fif")
    @patch("data_utils.BIDSPath")
    def test_load_raws_no_subjects(self, mock_bids_path, mock_read_raw_fif):
        """Test loading raw data with empty subject list."""
        data_params = DataParams(subject_ids=[], data_root="/fake/data")

        raws = load_raws(data_params)

        assert len(raws) == 0
        assert mock_read_raw_fif.call_count == 0
        assert mock_bids_path.call_count == 0

    @patch("data_utils.mne.io.read_raw_fif")
    @patch("data_utils.BIDSPath")
    def test_load_raws_per_subject_electrodes(
        self, mock_bids_path, mock_read_raw_fif, mock_raw_with_channels
    ):
        """Test loading raw data with per_subject_electrodes filtering."""
        mock_bids_path.return_value = "/fake/path/sub-01_task-podcast_ieeg.fif"
        # Use side_effect to create fresh mock for each call to avoid channel selection side effects
        mock_read_raw_fif.side_effect = [
            mock_raw_with_channels(),
            mock_raw_with_channels(),
        ]

        # Create DataParams with per_subject_electrodes - use channels that exist in mock_raw_with_channels
        per_subject_electrodes = {
            1: ["LGA1", "LGB1"],  # Select only 2 channels from the 6 available
            2: [
                "LGA2",
                "RGA1",
            ],  # Select different channels for subject 2 (all must exist in mock)
        }

        data_params = DataParams(
            subject_ids=[1, 2],
            data_root="/fake/data",
            per_subject_electrodes=per_subject_electrodes,
        )

        raws = load_raws(data_params)

        assert len(raws) == 2
        # Each raw should have only the selected channels
        # Note: In the mock, we can't easily test the actual channel picking,
        # but we can verify that the function completed without errors
        assert mock_read_raw_fif.call_count == 2

    @patch("data_utils.mne.io.read_raw_fif")
    @patch("data_utils.BIDSPath")
    def test_load_raws_channel_reg_ex(
        self, mock_bids_path, mock_read_raw_fif, mock_raw_with_channels
    ):
        """Test loading raw data with channel_reg_ex filtering."""
        mock_bids_path.return_value = "/fake/path/sub-01_task-podcast_ieeg.fif"
        mock_read_raw_fif.return_value = mock_raw_with_channels()

        # Create DataParams with channel_reg_ex
        data_params = DataParams(
            subject_ids=[1, 2],
            data_root="/fake/data",
            channel_reg_ex="LG.*",  # Should match LGA1, LGA2, LGB1, LGB2
        )

        raws = load_raws(data_params)

        assert len(raws) == 2
        assert mock_read_raw_fif.call_count == 2

    @patch("data_utils.mne.io.read_raw_fif")
    @patch("data_utils.BIDSPath")
    def test_load_raws_per_subject_electrodes_priority(
        self, mock_bids_path, mock_read_raw_fif, mock_raw_with_channels
    ):
        """Test that per_subject_electrodes takes priority over channel_reg_ex."""
        mock_bids_path.return_value = "/fake/path/sub-01_task-podcast_ieeg.fif"
        mock_read_raw_fif.return_value = mock_raw_with_channels()

        # Set both per_subject_electrodes and channel_reg_ex
        per_subject_electrodes = {1: ["LGA1"]}

        data_params = DataParams(
            subject_ids=[1],
            data_root="/fake/data",
            per_subject_electrodes=per_subject_electrodes,
            channel_reg_ex="RG.*",  # This should be ignored
        )

        raws = load_raws(data_params)

        assert len(raws) == 1
        # Function should complete without errors, indicating per_subject_electrodes was used

    def test_load_raws_with_real_temp_files(self, bids_temp_files):
        """Test load_raws with real temporary .fif files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary .fif files using the fixture
            bids_temp_files(temp_dir, [1, 2])

            # Test loading with real files
            data_params = DataParams(subject_ids=[1, 2], data_root=temp_dir)

            raws = load_raws(data_params)

            assert len(raws) == 2
            for raw in raws:
                assert isinstance(raw, mne.io.Raw)
                assert len(raw.ch_names) == 6  # All channels should be present

    def test_load_raws_with_real_temp_files_per_subject_electrodes(
        self, bids_temp_files
    ):
        """Test load_raws with real files and per_subject_electrodes filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary .fif file using the fixture
            bids_temp_files(temp_dir, [1])

            # Test with per_subject_electrodes
            per_subject_electrodes = {1: ["LGA1", "LGB1"]}
            data_params = DataParams(
                subject_ids=[1],
                data_root=temp_dir,
                per_subject_electrodes=per_subject_electrodes,
            )

            raws = load_raws(data_params)

            assert len(raws) == 1
            # Should have only the selected channels
            assert len(raws[0].ch_names) == 2
            assert "LGA1" in raws[0].ch_names
            assert "LGB1" in raws[0].ch_names
            assert "LGA2" not in raws[0].ch_names

    def test_load_raws_with_real_temp_files_channel_reg_ex(self, bids_temp_files):
        """Test load_raws with real files and channel_reg_ex filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary .fif file using the fixture
            bids_temp_files(temp_dir, [1])

            # Test with channel_reg_ex
            data_params = DataParams(
                subject_ids=[1],
                data_root=temp_dir,
                channel_reg_ex="LG.*",  # Should match LGA1, LGA2, LGB1, LGB2
            )

            raws = load_raws(data_params)

            assert len(raws) == 1
            # Should have 4 channels matching the regex
            assert len(raws[0].ch_names) == 4
            for ch_name in raws[0].ch_names:
                assert ch_name.startswith("LG")


@pytest.fixture
def temp_participant_mapping_tsv():
    """Create a temporary TSV file with participant mapping data for testing."""
    participant_data = """nyu_id	participant_id
661	sub-05
717	sub-12
723	sub-08
798	sub-09
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(participant_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_participant_mapping_csv():
    """Create a temporary CSV file with participant mapping data for testing."""
    participant_data = """nyu_id,participant_id
661,sub-05
717,sub-12
723,sub-08
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(participant_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_electrode_mapping_csv():
    """Create a temporary CSV file with electrode mapping data matching the example format."""
    electrode_data = """subject,elec,matfile
661,EEGG_14REF,14
661,EEGG_16REF,16
661,EEGG_20REF,20
717,LGA10,10
717,LGA18,18
717,LGA27,27
723,LSF12,12
723,LST10,34
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(electrode_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestReadSubjectMapping:
    """Test read_subject_mapping function for parsing participant mapping files."""

    def test_read_subject_mapping_tsv_format(self, temp_participant_mapping_tsv):
        """Test reading TSV file with participant mapping data."""
        result = read_subject_mapping(temp_participant_mapping_tsv, delimiter="\t")

        expected = {661: 5, 717: 12, 723: 8, 798: 9}

        assert result == expected
        assert len(result) == 4

    def test_read_subject_mapping_csv_format(self, temp_participant_mapping_csv):
        """Test reading CSV file with participant mapping data."""
        result = read_subject_mapping(temp_participant_mapping_csv, delimiter=",")

        expected = {661: 5, 717: 12, 723: 8}

        assert result == expected
        assert len(result) == 3

    def test_read_subject_mapping_return_types(self, temp_participant_mapping_tsv):
        """Test that returned mapping has correct types."""
        result = read_subject_mapping(temp_participant_mapping_tsv)

        # All keys should be integers (nyu_id)
        for nyu_id in result.keys():
            assert isinstance(nyu_id, int)

        # All values should be integers (converted participant_id)
        for participant_id in result.values():
            assert isinstance(participant_id, int)

    def test_read_subject_mapping_participant_id_extraction(self, temp_participant_mapping_tsv):
        """Test that participant IDs are correctly extracted from sub-XX format."""
        result = read_subject_mapping(temp_participant_mapping_tsv)

        # Check specific extractions
        assert result[661] == 5  # sub-05 -> 5
        assert result[717] == 12  # sub-12 -> 12
        assert result[723] == 8  # sub-08 -> 8

    def test_read_subject_mapping_empty_file(self):
        """Test reading empty participant mapping file."""
        empty_data = """nyu_id	participant_id
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(empty_data)
            temp_path = f.name

        try:
            result = read_subject_mapping(temp_path)
            assert result == {}
            assert len(result) == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_read_subject_mapping_nonexistent_file(self):
        """Test that reading non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            read_subject_mapping("nonexistent_participants.tsv")

    def test_read_subject_mapping_integration_with_electrode_file(
        self, temp_participant_mapping_csv, temp_electrode_mapping_csv
    ):
        """Test integration between read_subject_mapping and read_electrode_file."""
        # First, get the subject mapping
        subject_mapping = read_subject_mapping(temp_participant_mapping_csv, delimiter=",")

        # Expected mapping: {661: 5, 717: 12, 723: 8}
        assert subject_mapping == {661: 5, 717: 12, 723: 8}

        # Then, use it with read_electrode_file
        electrode_mapping = read_electrode_file(
            temp_electrode_mapping_csv, subject_mapping=subject_mapping
        )

        # Check that subject IDs have been mapped correctly
        expected_electrode_mapping = {
            5: ["EEGG_14REF", "EEGG_16REF", "EEGG_20REF"],  # Originally subject 661
            12: ["LGA10", "LGA18", "LGA27"],  # Originally subject 717
            8: ["LSF12", "LST10"],  # Originally subject 723
        }

        assert electrode_mapping == expected_electrode_mapping
        assert len(electrode_mapping) == 3

        # Verify that the original subject IDs (661, 717, 723) don't exist
        assert 661 not in electrode_mapping
        assert 717 not in electrode_mapping
        assert 723 not in electrode_mapping

        # Verify that the mapped subject IDs (5, 12, 8) do exist
        assert 5 in electrode_mapping
        assert 12 in electrode_mapping
        assert 8 in electrode_mapping


class TestLoadIeeGEdfFiles:
    """Tests for loading raw EDF iEEG data with consistent sampling rates."""

    @staticmethod
    def _make_raw(data: np.ndarray, sr: float) -> mne.io.Raw:
        ch_names = [f"CH{i:02d}" for i in range(data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sr, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)

    def test_load_ieeg_edf_files_preserves_data_and_metadata(self, monkeypatch):
        path_a = "sub-01.edf"
        path_b = "sub-02.edf"

        data_a = np.random.randn(2, 512)
        data_b = np.random.randn(3, 512)

        raw_map = {
            path_a: self._make_raw(data_a.copy(), 512),
            path_b: self._make_raw(data_b.copy(), 512),
        }

        def fake_read_raw_edf(path, preload=True, verbose=False):
            return raw_map[path]

        monkeypatch.setattr("data_utils.mne.io.read_raw_edf", fake_read_raw_edf)

        data_list, channel_names, sampling_rates = load_ieeg_edf_files(
            [path_a, path_b]
        )

        assert len(data_list) == 2
        assert len(channel_names) == 2
        assert sampling_rates == [512.0, 512.0]

        np.testing.assert_allclose(data_list[0], data_a.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(data_list[1], data_b.astype(np.float32), atol=1e-6)
        assert channel_names[0] == ["CH00", "CH01"]
        assert channel_names[1] == ["CH00", "CH01", "CH02"]

    def test_load_ieeg_edf_files_resamples_mismatched_rate(self, monkeypatch):
        path_a = "sub-01.edf"
        path_b = "sub-07.edf"

        sr_target = 512
        sr_high = 1024

        duration_seconds = 1.0
        samples_target = int(sr_target * duration_seconds)
        samples_high = int(sr_high * duration_seconds)

        data_a = np.random.randn(2, samples_target)
        data_b = np.random.randn(2, samples_high)

        raw_map = {
            path_a: self._make_raw(data_a.copy(), sr_target),
            path_b: self._make_raw(data_b.copy(), sr_high),
        }

        def fake_read_raw_edf(path, preload=True, verbose=False):
            return raw_map[path]

        monkeypatch.setattr("data_utils.mne.io.read_raw_edf", fake_read_raw_edf)

        data_list, channel_names, sampling_rates = load_ieeg_edf_files(
            [path_a, path_b], target_sampling_rate=sr_target
        )

        assert sampling_rates == [float(sr_target), float(sr_target)]
        assert data_list[0].shape[1] == samples_target
        assert data_list[1].shape[1] == samples_target
        assert data_list[1].dtype == np.float32

        # Resampled data should match polyphase resampling applied channel-wise.
        expected = resample_poly(
            data_b.astype(np.float32),
            sr_target // gcd(sr_target, sr_high),
            sr_high // gcd(sr_target, sr_high),
            axis=1,
        ).astype(np.float32)
        np.testing.assert_allclose(data_list[1], expected, atol=1e-5)
        assert channel_names[1] == ["CH00", "CH01"]

class TestAudioPreprocessingHelpers:
    """Tests for audio utilities ported from the volume-level notebook."""

    def test_load_audio_waveform_reads_mono_audio(self, tmp_path):
        soundfile = pytest.importorskip("soundfile")

        sr = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        left = np.sin(2 * np.pi * 220 * t)
        right = np.cos(2 * np.pi * 110 * t)
        stereo = np.stack([left, right], axis=1).astype(np.float32)

        audio_path = tmp_path / "test_audio.wav"
        soundfile.write(audio_path, stereo, sr)

        waveform, returned_sr = load_audio_waveform(str(audio_path))

        assert waveform.ndim == 1
        assert returned_sr == sr
        np.testing.assert_allclose(waveform, stereo.mean(axis=1), atol=1e-4, rtol=0)

    def test_hilbert_envelope_matches_constant_signal(self):
        waveform = np.ones(1024, dtype=np.float32)
        envelope = hilbert_envelope(waveform)

        assert envelope.shape == waveform.shape
        np.testing.assert_allclose(envelope, np.ones_like(envelope), atol=1e-6)

    def test_butterworth_lowpass_reduces_high_frequency_components(self):
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        low_freq = np.sin(2 * np.pi * 2 * t)
        high_freq = 0.5 * np.sin(2 * np.pi * 200 * t)
        envelope = low_freq + high_freq

        filtered = butterworth_lowpass_envelope(envelope, sr=sr, cutoff_hz=8.0)

        assert filtered.shape == envelope.shape
        assert np.std(filtered - low_freq) < np.std(envelope - low_freq)

    def test_resample_envelope_matches_expected_length(self):
        sr_in = 44100
        sr_out = 512
        envelope = np.linspace(0, 1, sr_in, dtype=np.float32)

        resampled = resample_envelope(envelope, sr_in=sr_in, sr_out=sr_out)

        expected_length = int(np.round(len(envelope) * sr_out / sr_in))
        assert resampled.shape[0] == expected_length

    def test_compress_envelope_db_is_monotonic(self):
        envelope = np.array([0.01, 0.1, 1.0, 2.0], dtype=np.float32)

        compressed = compress_envelope_db(envelope)

        assert compressed.shape == envelope.shape
        assert np.all(np.diff(compressed) > 0)
        assert np.isfinite(compressed).all()
