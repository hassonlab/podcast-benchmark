from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, hilbert, resample_poly, sosfilt, sosfiltfilt

from core.config import BaseTaskConfig, TaskConfig
import core.registry as registry
from utils.data_utils import load_raws


@dataclass
class VolumeLevelConfig(BaseTaskConfig):
    """Configuration for volume_level_decoding_task."""
    audio_path: str = "stimuli/podcast.wav"
    target_sr: int = 512
    audio_sr: int = 44100
    cutoff_hz: float = 8.0
    butter_order: int = 4
    zero_phase: bool = True
    log_eps: Optional[float] = None
    allow_resample_audio: bool = False
    window_size: Optional[float] = None  # in milliseconds
    hop_size: Optional[float] = None  # in milliseconds


@registry.register_task_data_getter(config_type=VolumeLevelConfig)
def volume_level_decoding_task(task_config: TaskConfig):
    """Prepare continuous audio-intensity targets for decoding.

      1. Load the podcast waveform from disk.
      2. Compute the Hilbert envelope and apply a Butterworth low-pass filter.
      3. Resample the envelope to match neural sampling rate expectations.
      4. Log-compress the envelope to produce perceptual loudness values.

    Optional sliding-window aggregation can be enabled via config by
    specifying window and hop sizes (ms). Targets are timestamped at the
    window centers, and each window is reduced to a single RMS value.

    Args:
        task_config (TaskConfig): Configuration object containing task-specific config and data params.

    Returns:
        pd.DataFrame: Continuous targets with columns ``start`` (seconds) and
        ``target`` (log-amplitude or windowed representation) ready for the
        decoding pipeline.
    """

    config: VolumeLevelConfig = task_config.task_specific_config
    data_params = task_config.data_params

    audio_rel_path = config.audio_path
    target_sr = config.target_sr
    expected_audio_sr = config.audio_sr
    cutoff_hz = config.cutoff_hz
    butter_order = config.butter_order
    zero_phase = config.zero_phase
    log_eps = config.log_eps
    allow_audio_resample = config.allow_resample_audio
    window_size_ms = config.window_size
    hop_size_ms = config.hop_size

    audio_path = (
        audio_rel_path
        if os.path.isabs(audio_rel_path)
        else os.path.join(data_params.data_root, audio_rel_path)
    )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at '{audio_path}'.")

    sr, waveform = wavfile.read(audio_path)

    if waveform.size == 0:
        raise ValueError(f"Loaded empty audio file from '{audio_path}'.")

    if sr != expected_audio_sr:
        if allow_audio_resample:
            warnings.warn(
                f"Audio sample rate {sr} Hz does not match expected {expected_audio_sr} Hz. "
                "Continuing with the actual sample rate.",
                RuntimeWarning,
            )
        else:
            raise ValueError(
                f"Expected audio sampled at {expected_audio_sr} Hz, got {sr} Hz. "
                "Provide a file with the expected rate or enable 'allow_resample_audio'."
            )

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        max_abs = max(abs(info.min), abs(info.max)) or 1
        waveform = waveform.astype(np.float32) / float(max_abs)
    else:
        waveform = waveform.astype(np.float32)

    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)

    if cutoff_hz <= 0:
        raise ValueError("'cutoff_hz' must be positive.")

    nyquist = 0.5 * sr
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"'cutoff_hz' ({cutoff_hz}) must be below Nyquist ({nyquist})."
        )

    sos = butter(butter_order, cutoff_hz / nyquist, btype="low", output="sos")

    if zero_phase:
        try:
            smoothed = sosfiltfilt(sos, envelope)
        except ValueError as exc:
            warnings.warn(
                f"Zero-phase filtering failed ({exc}); falling back to causal filtering.",
                RuntimeWarning,
            )
            smoothed = sosfilt(sos, envelope)
    else:
        smoothed = sosfilt(sos, envelope)

    if target_sr <= 0:
        raise ValueError("'target_sr' must be positive.")

    g = math.gcd(target_sr, sr)
    up = target_sr // g
    down = sr // g
    envelope_ds = resample_poly(smoothed, up, down)

    # Keep linear envelope non-negative
    envelope_ds = np.clip(envelope_ds, 0.0, None)

    n_samples = envelope_ds.shape[0]

    # If no windowing requested, convert the per-sample linear envelope to dB
    if window_size_ms is None:
        if log_eps is None:
            peak = float(envelope_ds.max()) if envelope_ds.size else 0.0
            log_eps = max(1e-12, peak * 1e-6)
        env_db = 20.0 * np.log10(envelope_ds + float(log_eps))

        times = np.arange(n_samples, dtype=np.float32) / float(target_sr)
        df = pd.DataFrame(
            {
                "start": times.astype(np.float32),
                "target": env_db.astype(np.float32),
            }
        )
        df.attrs["window_params"] = None
        return df

    try:
        width_ms = float(window_size_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("'window_size' must be convertible to milliseconds.") from exc
    if width_ms <= 0:
        raise ValueError("'window_size' must be > 0 milliseconds.")

    stride_ms = float(hop_size_ms) if hop_size_ms is not None else width_ms
    if stride_ms <= 0:
        raise ValueError("'hop_size' must be > 0 milliseconds.")

    width = width_ms / 1000.0
    stride = stride_ms / 1000.0

    window_samples, hop_samples, effective_sr = compute_window_hop(
        target_sr, width_ms, stride_ms
    )

    if window_samples > n_samples:
        raise ValueError(
            f"Requested window of {window_samples} samples exceeds envelope length {n_samples}."
        )

    # Compute RMS over the linear envelope, then convert each window RMS to dB.
    targets_linear = sliding_window_rms(envelope_ds, window_samples, hop_samples)

    if log_eps is None:
        peak = float(targets_linear.max()) if targets_linear.size else 0.0
        log_eps = max(1e-12, peak * 1e-6)

    targets = 20.0 * np.log10(targets_linear + float(log_eps))

    starts = np.arange(0, n_samples - window_samples + 1, hop_samples, dtype=int)
    if starts.size == 0:
        raise ValueError(
            "hop_size/window_size combination produced zero windows; adjust parameters."
        )

    centers = (starts + (window_samples - 1) / 2.0) / float(target_sr)

    if targets.ndim == 1:
        target_column = targets
    else:
        target_column = [window for window in targets]

    df = pd.DataFrame(
        {
            "start": centers.astype(np.float32),
            "target": target_column,
        }
    )

    df.attrs["window_params"] = {
        # Notebook-compatible keys
        "mode": "rms",
        "window_ms": width_ms,
        "hop_ms": stride_ms,
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "effective_sr": effective_sr,
        # Backwards-compatibility aliases used in earlier iterations of the codebase
        "window_size_ms": width_ms,
        "hop_size_ms": stride_ms,
        "window_size_s": width,
        "hop_size_s": stride,
        "window_reduction": "rms",
        # dB conversion was applied after RMS computation
        "db_after_rms": True,
    }

    return df


@registry.register_config_setter(name="volume_level_config_setter")
def volume_level_config_setter(experiment_config, raws, df_word):
    """Align experiment config to volume-level task outputs.

    This setter will:
      - Set data_params.window_width (seconds) from task config window_size (ms)
        so that neural windows align with audio windows by default.
      - Set the data preprocessing function to 'window_rms' if not already set,
        so each neural window is reduced to RMS amplitudes like the audio.

    The function is defensive: it will only set window_width if it is unset
    (<= 0 or falsy) and will not overwrite an explicitly provided preprocessing
    function unless none is present.
    """

    # Ensure nested objects exist
    tc = experiment_config.task_config
    if tc is None:
        return experiment_config

    dp = tc.data_params
    config: VolumeLevelConfig = tc.task_specific_config

    # If the task defines a window size in ms, set the neural window width (s)
    window_size_ms = config.window_size
    if window_size_ms is not None:
        try:
            width_ms = float(window_size_ms)
            if width_ms > 0:
                # Only override if not already set to a positive value
                if not getattr(dp, "window_width", None) or dp.window_width <= 0:
                    dp.window_width = width_ms / 1000.0
        except (TypeError, ValueError):
            # Ignore invalid values and leave window_width unchanged
            pass

    # Ensure neural preprocessing reduces to RMS windows like the audio
    if not dp.preprocessing_fn_name:
        dp.preprocessing_fn_name = "window_rms"

    # No change to experiment_config identity, return for convenience
    return experiment_config


def compute_window_hop(
    sampling_rate: float,
    window_ms: float,
    hop_ms: float,
) -> Tuple[int, int, float]:
    """Convert millisecond window parameters into sample counts and effective rate."""

    sr = float(sampling_rate)
    if window_ms <= 0 or hop_ms <= 0:
        raise ValueError("window_ms and hop_ms must be positive.")

    window_samples = max(1, int(round(window_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_ms * sr / 1000.0)))
    effective_sr = sr / hop_samples
    return window_samples, hop_samples, effective_sr


def sliding_window_rms(
    series: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Apply an RMS reducer across a sliding window along the last axis."""

    arr = np.asarray(series, dtype=np.float64, order="C")
    length = arr.shape[-1]
    if length < window_samples:
        raise ValueError(
            f"Window of {window_samples} samples exceeds series length {length}."
        )

    starts = np.arange(0, length - window_samples + 1, hop_samples, dtype=int)
    if starts.size == 0:
        raise ValueError(
            "hop_samples produced zero windows; choose a smaller hop or window."
        )

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    sq = np.square(arr, dtype=np.float64)
    cumsum = np.concatenate(
        [np.zeros((sq.shape[0], 1), dtype=np.float64), np.cumsum(sq, axis=-1)],
        axis=-1,
    )
    start_vals = cumsum[:, starts]
    stop_vals = cumsum[:, starts + window_samples]
    window_means = (stop_vals - start_vals) / float(window_samples)
    rms = np.sqrt(np.maximum(window_means, 0.0))
    out = rms.astype(np.float32)
    if squeeze:
        out = out[0]
    return out
