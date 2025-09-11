"""
Tests for data_utils.py.

Tests the get_data function for handling out-of-bounds time windows.
"""

import pytest
import numpy as np
import pandas as pd
import mne
from data_utils import get_data


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
        )

        assert data.shape[0] == len(df_word_in_bounds)
        assert len(targets) == len(df_word_in_bounds)
        assert len(words) == len(df_word_in_bounds)

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
            lag=lag, raws=[mock_raw], df_word=df_word_mixed, window_width=window_width
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
        )

        print(words)
        expected_valid_events = 3
        assert data.shape[0] == expected_valid_events
        assert len(targets) == expected_valid_events
        assert len(words) == expected_valid_events
