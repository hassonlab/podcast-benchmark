#!/usr/bin/env python3
"""Command-line helper to reproduce the volume-level ridge analysis.

This script mirrors the key steps from the exploratory notebook using the
refactored utility modules. It performs the following operations:

1. Load audio (either a waveform or a pre-computed envelope) and neural data.
2. Optionally compute the audio envelope and resample it to the target rate.
3. Optionally apply RMS-based sliding-window aggregation to match the notebook.
4. Run ridge regression cross-validation across the requested lags.
5. Save the resulting metrics and an accompanying diagnostic plot.

Typical usage (with raw waveform input)::

    python scripts/run_ridge_analysis.py \
        --audio-path data/audio/session01.wav \
        --neural-path data/neural/session01_high_gamma.npy \
        --sampling-rate 1000 \
        --lags "-200,-100,0,100,200" \
        --window-ms 50 --hop-ms 25 \
        --output-dir outputs/ridge_session01

If you already have envelope arrays saved to disk, provide ``--audio-kind envelope``
so the script skips redundant envelope computations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from plot_utils import plot_ridge_results
from ridge_utils import apply_sliding_window_rms, ridge_r2_by_lag
from volume_lvl_utils import (
    butterworth_lowpass_envelope,
    hilbert_envelope,
    load_audio_waveform,
    resample_envelope,
)


def _parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item]


def _load_audio(path: Path, *, audio_kind: str, target_sr: int | None) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg"}:
        audio, sr = load_audio_waveform(str(path))
    elif suffix in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "audio" not in data:
                raise KeyError("NPZ file must contain a key named 'audio'")
            audio = data["audio"]
            sr = int(data.get("sr", target_sr or 0))
        else:
            audio = data
            sr = target_sr or 0
    else:
        raise ValueError(f"Unsupported audio file extension: {suffix}")

    if sr <= 0:
        raise ValueError(
            "Sampling rate for audio is unknown. Provide --sampling-rate or include 'sr' in an NPZ file."
        )

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Audio must be 1D; received shape {audio.shape}")

    if audio_kind == "waveform":
        envelope = hilbert_envelope(audio)
        envelope = butterworth_lowpass_envelope(envelope, sr)
    elif audio_kind == "envelope":
        envelope = audio
    else:
        raise ValueError("audio_kind must be 'waveform' or 'envelope'")

    return envelope.astype(np.float32, copy=False), sr


def _load_neural(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Neural file not found: {path}")

    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "neural" not in data:
            raise KeyError("NPZ file must contain a key named 'neural'")
        neural = data["neural"]
    else:
        neural = data
    neural = np.asarray(neural, dtype=np.float32)

    if neural.ndim != 2:
        raise ValueError(f"Neural data must have shape (channels, time); received {neural.shape}")

    return neural


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def _save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline(
    *,
    audio_path: Path,
    audio_kind: str,
    neural_path: Path,
    sampling_rate: float,
    lags: Sequence[float],
    window_ms: float | None,
    hop_ms: float | None,
    alphas: Sequence[float] | None,
    cv_splits: int,
    output_dir: Path,
    plot: bool,
    save_plot: bool,
    sliding_trim: bool,
) -> dict[str, np.ndarray]:
    audio_envelope, audio_sr = _load_audio(audio_path, audio_kind=audio_kind, target_sr=int(round(sampling_rate)))
    neural = _load_neural(neural_path)

    if int(round(sampling_rate)) <= 0:
        raise ValueError("sampling_rate must be positive")

    if audio_sr != int(round(sampling_rate)):
        audio_envelope = resample_envelope(audio_envelope, audio_sr, int(round(sampling_rate)))

    if window_ms and hop_ms:
        neural_win, audio_win, eff_sr, metadata = apply_sliding_window_rms(
            audio_envelope,
            neural,
            sampling_rate,
            window_ms,
            hop_ms,
            trim_to_common=sliding_trim,
        )
        sampling_rate = eff_sr
        neural = neural_win
        audio_envelope = audio_win
    elif window_ms or hop_ms:
        raise ValueError("Both window_ms and hop_ms must be provided together")

    results = ridge_r2_by_lag(
        audio_envelope,
        neural,
        sampling_rate,
        lags,
        alphas=alphas,
        cv_splits=cv_splits,
        verbose=True,
    )

    _ensure_directory(output_dir)
    arrays_path = output_dir / "ridge_results.npz"
    _save_npz(arrays_path, **{k: np.asarray(v) for k, v in results.items()})
    _write_json(output_dir / "ridge_results_summary.json", {k: np.asarray(v).tolist() for k, v in results.items()})

    if plot:
        fig, _ = plot_ridge_results(results, show=True)
        if save_plot:
            fig.savefig(output_dir / "ridge_diagnostics.png", dpi=300)
    elif save_plot:
        fig, _ = plot_ridge_results(results, show=False)
        fig.savefig(output_dir / "ridge_diagnostics.png", dpi=300)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ridge regression analysis on volume-level data")
    parser.add_argument("--audio-path", required=True, type=Path, help="Path to audio waveform/envelope (.wav/.npy/.npz)")
    parser.add_argument("--audio-kind", choices=["waveform", "envelope"], default="waveform", help="Specify whether the audio file is a raw waveform or a pre-computed envelope")
    parser.add_argument("--neural-path", required=True, type=Path, help="Path to neural features (.npy/.npz) shaped (channels, time)")
    parser.add_argument("--sampling-rate", required=True, type=float, help="Target sampling rate (Hz) shared across modalities after preprocessing")
    parser.add_argument("--lags", required=True, type=_parse_float_list, help="Comma-separated lags in milliseconds (e.g. '-200,-100,0,100,200')")
    parser.add_argument("--alphas", type=_parse_float_list, default=None, help="Optional comma-separated ridge alphas; defaults to the notebook grid when omitted")
    parser.add_argument("--window-ms", type=float, default=None, help="Sliding window size in milliseconds (set with --hop-ms)")
    parser.add_argument("--hop-ms", type=float, default=None, help="Sliding window hop in milliseconds (set with --window-ms)")
    parser.add_argument("--no-trim", action="store_true", help="Disable trimming to the common number of windows after RMS aggregation")
    parser.add_argument("--cv-splits", type=int, default=10, help="Number of CV splits (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ridge"), help="Directory for saving results and plots")
    parser.add_argument("--no-plot", action="store_true", help="Skip displaying the diagnostic plot")
    parser.add_argument("--save-plot", action="store_true", help="Persist the diagnostic figure to disk")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_pipeline(
        audio_path=args.audio_path,
        audio_kind=args.audio_kind,
        neural_path=args.neural_path,
        sampling_rate=args.sampling_rate,
        lags=args.lags,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        alphas=args.alphas,
        cv_splits=args.cv_splits,
        output_dir=args.output_dir,
        plot=not args.no_plot,
        save_plot=args.save_plot,
        sliding_trim=not args.no_trim,
    )


if __name__ == "__main__":
    main()
