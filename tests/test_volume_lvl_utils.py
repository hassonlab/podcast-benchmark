"""Tests for volume-level utilities extracted from notebooks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from volume_lvl_utils import (
    bandpass_high_gamma,
    butterworth_lowpass_envelope,
    compress_envelope_db,
    extract_high_gamma_features,
    high_gamma_envelope,
    hilbert_envelope,
    load_audio_waveform,
    log_high_gamma_envelope,
    resample_envelope,
    zscore_subjects,
)


class TestZScoreSubjects:
    """Validate per-subject z-scoring behaviour."""

    def test_zscore_subjects_normalizes_each_channel(self) -> None:
        rng = np.random.default_rng(123)
        data_a = rng.normal(loc=2.0, scale=5.0, size=(3, 1024)).astype(np.float32)
        data_b = rng.normal(loc=-1.0, scale=2.0, size=(2, 512)).astype(np.float32)

        normalized, stats = zscore_subjects([data_a, data_b])

        assert len(normalized) == 2
        for arr in normalized:
            means = np.mean(arr, axis=1)
            stds = np.std(arr, axis=1, ddof=0)
            np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-6)
            np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-6)

        assert stats[0]["n_channels"] == 3
        assert stats[1]["n_timepoints"] == 512

    def test_zscore_subjects_respects_electrode_subset(self) -> None:
        data = np.array(
            [
                np.linspace(0, 1, 8),  # channel 0
                np.full(8, 10.0),      # channel 1 constant
            ],
            dtype=np.float32,
        )

        normalized, stats = zscore_subjects([data], electrode_groups=[[0]])

        arr = normalized[0]
        # Channel 0 should be standardised to mean 0, std 1 using itself.
        np.testing.assert_allclose(arr[0].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(arr[0].std(ddof=0), 1.0, atol=1e-6)

        # Channel 1 falls back to its own stats; constant -> zero after centering.
        np.testing.assert_allclose(arr[1], np.zeros_like(arr[1]), atol=1e-6)
        assert stats[0]["global_std"] > 0


class TestAudioPreprocessingHelpers:
    """Tests for audio utilities ported from the volume-level notebook."""

    def test_load_audio_waveform_reads_mono_audio(self, tmp_path) -> None:
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

    def test_hilbert_envelope_matches_constant_signal(self) -> None:
        waveform = np.ones(1024, dtype=np.float32)
        envelope = hilbert_envelope(waveform)

        assert envelope.shape == waveform.shape
        np.testing.assert_allclose(envelope, np.ones_like(envelope), atol=1e-6)

    def test_butterworth_lowpass_reduces_high_frequency_components(self) -> None:
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        low_freq = np.sin(2 * np.pi * 2 * t)
        high_freq = 0.5 * np.sin(2 * np.pi * 200 * t)
        envelope = low_freq + high_freq

        filtered = butterworth_lowpass_envelope(envelope, sr=sr, cutoff_hz=8.0)

        assert filtered.shape == envelope.shape
        assert np.std(filtered - low_freq) < np.std(envelope - low_freq)

    def test_resample_envelope_matches_expected_length(self) -> None:
        sr_in = 44100
        sr_out = 512
        envelope = np.linspace(0, 1, sr_in, dtype=np.float32)

        resampled = resample_envelope(envelope, sr_in=sr_in, sr_out=sr_out)

        expected_length = int(np.round(len(envelope) * sr_out / sr_in))
        assert resampled.shape[0] == expected_length

    def test_compress_envelope_db_is_monotonic(self) -> None:
        envelope = np.array([0.01, 0.1, 1.0, 2.0], dtype=np.float32)

        compressed = compress_envelope_db(envelope)

        assert compressed.shape == envelope.shape
        assert np.all(np.diff(compressed) > 0)
        assert np.isfinite(compressed).all()


class TestHighGammaHelpers:
    """Tests for high-gamma feature extraction pipeline."""

    def test_bandpass_high_gamma_preserves_in_band_energy(self) -> None:
        sr = 1_000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        low_component = np.sin(2 * np.pi * 30 * t)
        high_component = np.sin(2 * np.pi * 110 * t)
        signal = low_component + high_component

        filtered = bandpass_high_gamma(signal, sampling_rate=sr, low=70.0, high=150.0)

        fft = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(len(filtered), d=1 / sr)

        low_band = np.abs(fft[(freqs >= 20) & (freqs <= 40)]).sum()
        high_band = np.abs(fft[(freqs >= 90) & (freqs <= 130)]).sum()
        assert high_band > 5 * low_band

    def test_high_gamma_envelope_matches_manual_computation(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(2, 2_048)).astype(np.float32)
        sr = 512.0

        filtered, envelope = high_gamma_envelope(data, sampling_rate=sr)

        manual_filtered = bandpass_high_gamma(data, sampling_rate=sr)
        manual_envelope = np.abs(np.apply_along_axis(hilbert_envelope, 1, manual_filtered))

        np.testing.assert_allclose(filtered, manual_filtered, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(envelope, manual_envelope, rtol=1e-5, atol=1e-5)

    def test_extract_high_gamma_features_returns_metadata(self) -> None:
        rng = np.random.default_rng(7)
        data_list = [
            rng.normal(size=(3, 1_024)).astype(np.float32),
            rng.normal(size=(2, 800)).astype(np.float32),
        ]
        sampling_rates = [512.0, 512.0]

        filtered_list, envelope_list, log_list, metadata = extract_high_gamma_features(
            data_list,
            sampling_rates,
            electrode_groups=[[0, 2], None],
        )

        assert len(filtered_list) == len(envelope_list) == len(log_list) == len(data_list)
        for filtered, envelope, log_arr in zip(filtered_list, envelope_list, log_list):
            assert filtered.shape == envelope.shape == log_arr.shape
            assert filtered.dtype == np.float32

        assert metadata[0]["electrode_indices"] == [0, 2]
        assert math.isclose(metadata[0]["band"][0], 70.0)

    def test_log_high_gamma_envelope_returns_eps(self) -> None:
        rng = np.random.default_rng(3)
        envelope = rng.random(size=(2, 100)).astype(np.float32)

        logged, eps = log_high_gamma_envelope(envelope)

        assert eps > 0
        assert logged.shape == envelope.shape
        assert np.isfinite(logged).all()
