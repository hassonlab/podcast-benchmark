import mne
import numpy as np

from utils.raw_preprocessing import (
    despike_raw,
    high_gamma_envelope,
    prepare_broadband_raw,
)


def _raw(data, sfreq=512.0, names=None):
    names = names or [f"E{i}" for i in range(len(data))]
    return mne.io.RawArray(
        np.asarray(data, dtype=float),
        mne.create_info(names, sfreq, ch_types="seeg"),
        verbose=False,
    )


def test_despike_interpolates_large_channel_outlier():
    values = np.sin(np.linspace(0, 4 * np.pi, 1000))
    values[500] = 1000.0
    raw = _raw([values], sfreq=1000.0)

    despike_raw(raw, iqr_multiplier=4.0, dilation_s=0.0)

    assert abs(raw.get_data()[0, 500]) < 2.0


def test_prepare_broadband_drops_bads_resamples_and_references():
    rng = np.random.default_rng(1)
    raw = _raw(rng.normal(size=(3, 1000)), sfreq=1000.0, names=["A", "B", "bad"])
    raw.info["bads"] = ["bad"]

    result = prepare_broadband_raw(
        raw,
        target_sfreq=500.0,
        spike_dilation_s=0.0,
        notch_freqs=(),
    )

    assert result.ch_names == ["A", "B"]
    assert result.info["sfreq"] == 500.0
    np.testing.assert_allclose(result.get_data().mean(axis=0), 0.0, atol=1e-12)


def test_high_gamma_envelope_is_nonnegative_and_keeps_shape():
    times = np.arange(2048) / 512.0
    raw = _raw([np.sin(2 * np.pi * 100 * times)], sfreq=512.0)

    result = high_gamma_envelope(raw)

    assert result.get_data().shape == raw.get_data().shape
    assert np.all(result.get_data() >= 0)
