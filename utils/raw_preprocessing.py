"""Reusable MNE Raw preprocessing operations for dataset artifact builders."""

from collections.abc import Iterable

import numpy as np
from scipy import interpolate, ndimage, stats


def despike_raw(raw, iqr_multiplier: float = 4.0, dilation_s: float = 0.1):
    """Interpolate channel outliers after dilating an IQR-based spike mask."""
    raw.load_data()
    data = raw._data
    medians = np.median(data, axis=-1, keepdims=True)
    iqrs = stats.iqr(data, axis=-1)
    spike_mask = np.abs(data - medians) > iqrs[:, None] * iqr_multiplier
    if dilation_s > 0:
        width = max(1, int(round(dilation_s * raw.info["sfreq"])))
        spike_mask = ndimage.binary_dilation(
            spike_mask, structure=np.ones((1, width), dtype=bool)
        )

    for channel_idx, spikes in enumerate(spike_mask):
        if not spikes.any():
            continue
        clean = np.flatnonzero(~spikes)
        if len(clean) < 2:
            raise ValueError(
                f"Cannot interpolate {raw.ch_names[channel_idx]}: fewer than "
                "two clean samples remain"
            )
        raw._data[channel_idx, spikes] = interpolate.pchip_interpolate(
            clean, raw._data[channel_idx, ~spikes], np.flatnonzero(spikes)
        )
    return raw


def prepare_broadband_raw(
    raw,
    *,
    target_sfreq: float = 512.0,
    iqr_multiplier: float = 4.0,
    spike_dilation_s: float = 0.1,
    notch_freqs: Iterable[float] = (60.0, 120.0, 180.0),
    notch_widths: float = 2.0,
):
    """Create a cleaned, common-average-referenced broadband Raw."""
    raw = raw.copy().load_data()
    bads = [name for name in raw.info["bads"] if name in raw.ch_names]
    if bads:
        raw.drop_channels(bads)
    if len(raw.ch_names) < 2:
        raise ValueError("Common-average reference requires at least two channels")
    if not np.isclose(raw.info["sfreq"], target_sfreq):
        raw.resample(target_sfreq)
    despike_raw(raw, iqr_multiplier=iqr_multiplier, dilation_s=spike_dilation_s)
    raw.set_eeg_reference(ref_channels="average", ch_type="seeg", verbose=False)
    freqs = [float(freq) for freq in notch_freqs if freq < raw.info["sfreq"] / 2]
    if freqs:
        raw.notch_filter(
            freqs=freqs,
            notch_widths=notch_widths,
            picks="data",
            n_jobs=1,
            verbose=False,
        )
    return raw


def high_gamma_envelope(
    raw,
    *,
    l_freq: float = 70.0,
    h_freq: float = 200.0,
    filter_order: int = 4,
):
    """Return the Butterworth-bandpassed analytic amplitude envelope."""
    if h_freq >= raw.info["sfreq"] / 2:
        raise ValueError("High-gamma upper frequency must be below Nyquist")
    result = raw.copy().load_data()
    result.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={"order": filter_order, "ftype": "butter"},
        picks="data",
        verbose=False,
    )
    result.apply_hilbert(picks="data", envelope=True, verbose=False)
    return result
