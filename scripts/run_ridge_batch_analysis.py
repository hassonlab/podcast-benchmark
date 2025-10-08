#!/usr/bin/env python3
"""Batch ridge latency analysis across all subjects using curated electrode sets.

This script recreates the multi-subject workflow from the volume-level encoding
notebook:

1. Load the podcast stimulus, compute its Hilbert envelope, low-pass filter, and
   resample to the neural sampling rate.
2. Load iEEG recordings for each requested subject, resample them to a common
   rate, and z-score using the "good" electrodes identified in the notebook.
3. Extract high-gamma log-power features restricted to those good electrodes.
4. Optionally apply sliding-window RMS aggregation (default: 200 ms window,
   25 ms hop) to match the notebook's temporal resolution.
5. Run ridge regression latency sweeps for each subject individually,
   average the per-subject curves, and fit a pooled model that concatenates all
   electrodes across subjects.
6. Save per-mode results (CSV + NPZ + JSON summaries) and diagnostic plots via
   :func:`plot_utils.plot_ridge_results`.

Example usage (mirrors the notebook defaults)::

    python scripts/run_ridge_batch_analysis.py \
        --audio-path data/stimuli/podcast.wav \
        --data-root data \
        --subjects 1 2 3 4 5 6 7 8 9 \
        --target-sr 512 \
        --lags "-1000:1000:10" \
        --window-ms 200 --hop-ms 25 \
        --output-dir outputs/ridge_batch \
        --save-plots

The script requires the optional MNE dependency to load EDF files and assumes
that each subject is stored at ``{data_root}/sub-XX/ieeg/sub-XX_task-podcast_ieeg.edf``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from plot_utils import plot_ridge_results
from ridge_utils import apply_sliding_window_rms, ridge_r2_by_lag
from volume_lvl_utils import (
    butterworth_lowpass_envelope,
    compress_envelope_db,
    extract_high_gamma_features,
    hilbert_envelope,
    load_audio_waveform,
    resample_envelope,
    zscore_subjects,
)

DEFAULT_SUBJECT_IDS: tuple[int, ...] = tuple(range(1, 10))
BAD_ELECTRODES_BY_SUBJECT: dict[int, list[int]] = {
    1: [33, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
    2: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    3: [254, 255, 256, 257, 258, 259, 260, 261],
    4: [162, 163, 164, 165, 166, 167, 168, 169, 170, 171],
    5: [41, 131, 161, 162, 163, 164, 165],
    6: [128, 137, 159, 174, 175, 176],
    7: [78, 79, 80, 81, 82, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136],
    8: [41, 58, 75, 82, 83, 84, 85, 86, 87, 88, 89],
    9: [58, 59, 60, 192, 193, 194, 195, 196, 197, 199, 200, 201, 202, 203],
}


def _parse_lag_spec(spec: str) -> np.ndarray:
    """Parse ``start:stop:step`` into an inclusive NumPy array."""

    try:
        start_str, stop_str, step_str = spec.split(":")
        start = float(start_str)
        stop = float(stop_str)
        step = float(step_str)
    except ValueError as exc:  # pragma: no cover - guardrail for CLI use
        raise argparse.ArgumentTypeError(
            "--lags must follow 'start:stop:step' (e.g. '-1000:1000:10')"
        ) from exc

    if step == 0:
        raise argparse.ArgumentTypeError("Lag step must be non-zero")

    # Ensure the stop value is inclusive like in the notebook implementation
    n_steps = int(np.floor((stop - start) / step)) + 1
    lags = start + step * np.arange(n_steps, dtype=float)
    if lags[-1] != stop:
        lags = np.append(lags, stop)
    return lags


def _parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    items: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(float(chunk))
    if not items:
        raise argparse.ArgumentTypeError("Provide at least one numeric value when setting --alphas")
    return items


def _load_audio_envelope(audio_path: Path, *, target_sr: int, cutoff_hz: float) -> tuple[np.ndarray, int]:
    waveform, sr = load_audio_waveform(str(audio_path))
    envelope = hilbert_envelope(waveform)
    envelope_lp = butterworth_lowpass_envelope(envelope, sr, cutoff_hz=cutoff_hz)
    envelope_resampled = resample_envelope(envelope_lp, sr, target_sr)
    env_db = compress_envelope_db(envelope_resampled)
    return env_db.astype(np.float32, copy=False), int(target_sr)


def _load_subject_raw(
    subject_id: int,
    *,
    data_root: Path,
    target_sr: float,
) -> tuple[np.ndarray, list[str], float]:
    ieeg_path = data_root / f"sub-{subject_id:02d}" / "ieeg" / f"sub-{subject_id:02d}_task-podcast_ieeg.edf"
    if not ieeg_path.exists():  # pragma: no cover - user-specific environment guardrail
        raise FileNotFoundError(f"Missing EDF file for subject {subject_id}: {ieeg_path}")

    raw = mne.io.read_raw_edf(str(ieeg_path), preload=True, verbose=False)
    sr = float(raw.info["sfreq"])
    if target_sr > 0 and abs(sr - target_sr) > 1e-6:
        raw.resample(target_sr, npad="auto")
        sr = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float32, copy=False)
    return data, list(raw.ch_names), sr


def _build_good_electrode_groups(
    subject_ids: Sequence[int],
    data_list: Sequence[np.ndarray],
) -> list[list[int]]:
    groups: list[list[int]] = []
    for subject_id, data in zip(subject_ids, data_list):
        n_channels = data.shape[0]
        if n_channels == 0:
            groups.append([])
            continue
        base = list(range(n_channels - 1)) if n_channels > 1 else [0]
        bad_candidates = BAD_ELECTRODES_BY_SUBJECT.get(subject_id, [])
        bad = {idx for idx in bad_candidates if 0 <= idx < n_channels}
        good = [idx for idx in base if idx not in bad]
        if not good:  # fallback if manual curation removed everything
            good = base
        groups.append(good)
        removed = sorted(set(base) - set(good))
        print(
            f"Subject {subject_id:02d}: {len(good)} good electrodes retained"
            + (f", removed indices {removed}" if removed else "")
        )
    return groups


def _summarize_best(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    if df.empty:
        return {"label": label, "lag_ms": float("nan"), "r2": float("nan")}
    idx = int(df["r2"].idxmax())
    row = df.loc[idx]
    summary = {
        "label": label,
        "lag_ms": float(row["lag_ms"]),
        "r2": float(row["r2"]),
    }
    for key in ["train_r2", "alpha", "coef_norm", "n_features", "n_samples"]:
        if key in row:
            summary[key] = float(row[key])
    return summary


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(npz_path, **{col: df[col].to_numpy() for col in df.columns})


def _save_plot(results: dict[str, np.ndarray], path: Path | None, *, show: bool, title_suffix: str) -> None:
    fig, _ = plot_ridge_results(results, show=show)
    if path is not None:
        fig.suptitle(f"Ridge regression diagnostics — {title_suffix}", fontsize=16)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close(fig)


def run_batch_pipeline(args: argparse.Namespace) -> None:
    subject_ids = tuple(sorted(set(args.subjects))) if args.subjects else DEFAULT_SUBJECT_IDS
    print(f"Running ridge batch analysis for subjects: {subject_ids}")

    output_dir = args.output_dir
    _ensure_dir(output_dir)

    # 1. Audio preprocessing
    audio_series, audio_sr = _load_audio_envelope(
        args.audio_path,
        target_sr=int(round(args.target_sr)),
        cutoff_hz=args.audio_lowpass,
    )
    print(f"Loaded audio envelope: {audio_series.shape[0]} samples at {audio_sr} Hz")

    # 2. Load neural recordings
    data_list: list[np.ndarray] = []
    ch_names_list: list[list[str]] = []
    sr_list: list[float] = []
    for subject_id in subject_ids:
        data, ch_names, sr = _load_subject_raw(
            subject_id,
            data_root=args.data_root,
            target_sr=args.target_sr,
        )
        data_list.append(data)
        ch_names_list.append(ch_names)
        sr_list.append(sr)
        print(
            f"Loaded subject {subject_id:02d}: {data.shape[0]} channels × {data.shape[1]} samples at {sr:.2f} Hz"
        )

    if len(set(round(sr, 6) for sr in sr_list)) != 1:
        raise RuntimeError(f"Sampling rates differ across subjects after resampling: {sr_list}")
    shared_sr = sr_list[0]

    # 3. Electrode selection + z-score normalisation
    good_groups = _build_good_electrode_groups(subject_ids, data_list)
    zscored_list, zscore_stats = zscore_subjects(data_list, electrode_groups=good_groups)
    for subject_id, stats in zscore_stats.items():
        print(
            f"Subject {subject_id:02d} z-score stats: mean≈{stats['post_mean']:.3e}, std≈{stats['post_std']:.3e}"
        )

    # 4. High-gamma extraction & log transform restricted to good electrodes
    _, _, log_list, hg_metadata = extract_high_gamma_features(
        zscored_list,
        sr_list,
        electrode_groups=good_groups,
        zero_phase=True,
    )

    subject_datasets: list[dict[str, object]] = []
    for idx, subject_id in enumerate(subject_ids):
        neural = np.asarray(log_list[idx], dtype=np.float32)
        if neural.size == 0:
            print(f"Skipping subject {subject_id:02d}: no usable electrodes after filtering.")
            continue
        length = min(neural.shape[1], audio_series.shape[0])
        neural = np.ascontiguousarray(neural[:, :length], dtype=np.float32)
        audio_trim = np.ascontiguousarray(audio_series[:length], dtype=np.float32)
        metadata = hg_metadata[idx]
        dataset = {
            "subject_id": subject_id,
            "neural": neural,
            "audio": audio_trim,
            "sampling_rate": shared_sr,
            "electrode_indices": metadata.get("electrode_indices", good_groups[idx]),
        }
        subject_datasets.append(dataset)
        print(
            f"Subject {subject_id:02d}: prepared neural matrix {neural.shape[0]}×{neural.shape[1]} (log high-gamma)"
        )

    if not subject_datasets:
        raise RuntimeError("No subject datasets prepared; check electrode selections or input data.")

    # Align all subjects to the global minimum duration (samples)
    global_min_len = min(ds["audio"].shape[0] for ds in subject_datasets)
    if any(ds["audio"].shape[0] != global_min_len for ds in subject_datasets):
        for ds in subject_datasets:
            ds["audio"] = ds["audio"][:global_min_len]
            ds["neural"] = ds["neural"][:, :global_min_len]
        print(f"Trimmed all subjects to {global_min_len} samples (~{global_min_len / shared_sr / 60:.1f} min)")

    # 5. Optional sliding-window RMS aggregation
    if not args.disable_window:
        print(
            f"Applying sliding-window RMS with window={args.window_ms} ms, hop={args.hop_ms} ms"
        )
        window_counts: list[int] = []
        for ds in subject_datasets:
            neural_win, audio_win, eff_sr, metadata = apply_sliding_window_rms(
                ds["audio"],
                ds["neural"],
                ds["sampling_rate"],
                args.window_ms,
                args.hop_ms,
                trim_to_common=True,
            )
            ds["neural"] = neural_win
            ds["audio"] = audio_win
            ds["sampling_rate"] = eff_sr
            ds["window_metadata"] = asdict(metadata)
            window_counts.append(neural_win.shape[1])
        min_windows = min(window_counts)
        if any(count != min_windows for count in window_counts):
            for ds in subject_datasets:
                ds["neural"] = ds["neural"][:, :min_windows]
                ds["audio"] = ds["audio"][:min_windows]
        print(
            f"Sliding-window RMS alignment complete: {min_windows} windows at {subject_datasets[0]['sampling_rate']:.2f} Hz"
        )
    else:
        print("Sliding-window RMS disabled; using sample-wise sequences.")

    # 6. Ridge CV sweeps
    lags_ms = _parse_lag_spec(args.lags)
    alphas = _parse_float_list(args.alphas)

    per_subject_results: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | str]] = []

    for ds in subject_datasets:
        subject_id = int(ds["subject_id"])
        print(
            f"\n=== Subject {subject_id:02d} — {ds['neural'].shape[0]} electrodes, {ds['neural'].shape[1]} samples ==="
        )
        results = ridge_r2_by_lag(
            ds["audio"],
            ds["neural"],
            ds["sampling_rate"],
            lags_ms,
            alphas=alphas,
            cv_splits=args.cv_splits,
            device=args.device,
            verbose=args.verbose,
        )
        df = pd.DataFrame(results)
        if df.empty:
            print("  -> Skipping (no usable lags).")
            continue
        df["subject_id"] = subject_id
        df["n_electrodes"] = ds["neural"].shape[0]
        per_subject_results[subject_id] = df
        summary = _summarize_best(df, f"Subject {subject_id:02d}")
        summary_rows.append(summary)
        metrics = []
        for key in ("lag_ms", "r2", "train_r2", "alpha", "coef_norm", "n_features", "n_samples"):
            value = summary.get(key)
            if value is not None and np.isfinite(value):
                metrics.append(f"{key}={value:.3f}")
        print("  -> Best lag summary: " + ", ".join(metrics))
        csv_path = output_dir / f"ridge_subject_{subject_id:02d}.csv"
        _save_dataframe(df, csv_path)
        if args.save_plots or args.show_plots:
            plot_path = output_dir / f"ridge_subject_{subject_id:02d}.png" if args.save_plots else None
            _save_plot(results, plot_path, show=args.show_plots, title_suffix=f"Subject {subject_id:02d}")

    if not per_subject_results:
        raise RuntimeError("No per-subject results were produced; aborting aggregation.")

    # 7a. Subject-average curve
    combined_df = pd.concat(per_subject_results.values(), ignore_index=True)
    grouped = combined_df.groupby("lag_ms", as_index=False).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        train_r2=("train_r2", "mean"),
        alpha=("alpha", "mean"),
        coef_norm=("coef_norm", "mean"),
        n_features=("n_features", "mean"),
        n_samples=("n_samples", "mean"),
        n_subjects=("subject_id", "nunique"),
    )
    grouped.rename(columns={"r2_mean": "r2"}, inplace=True)
    avg_csv_path = output_dir / "ridge_average.csv"
    _save_dataframe(grouped, avg_csv_path)
    avg_results = {
        "lag_ms": grouped["lag_ms"].to_numpy(),
        "r2": grouped["r2"].to_numpy(),
        "train_r2": grouped["train_r2"].to_numpy(),
        "alpha": grouped["alpha"].to_numpy(),
        "coef_norm": grouped["coef_norm"].to_numpy(),
        "n_features": grouped["n_features"].to_numpy(),
        "n_samples": grouped["n_samples"].to_numpy(),
    }
    summary_rows.append(_summarize_best(grouped, "Subject mean"))
    if args.save_plots or args.show_plots:
        plot_path = output_dir / "ridge_average.png" if args.save_plots else None
        _save_plot(avg_results, plot_path, show=args.show_plots, title_suffix="Subject mean")

    # 7b. Pooled electrodes curve
    pooled_neural = np.concatenate([ds["neural"] for ds in subject_datasets], axis=0)
    pooled_audio = subject_datasets[0]["audio"]
    pooled_sr = subject_datasets[0]["sampling_rate"]
    pooled_results = ridge_r2_by_lag(
        pooled_audio,
        pooled_neural,
        pooled_sr,
        lags_ms,
        alphas=alphas,
        cv_splits=args.cv_splits,
        device=args.device,
        verbose=args.verbose,
    )
    pooled_df = pd.DataFrame(pooled_results)
    pooled_df["subject_id"] = "pooled"
    pooled_df["n_electrodes"] = pooled_neural.shape[0]
    pooled_csv_path = output_dir / "ridge_pooled.csv"
    _save_dataframe(pooled_df, pooled_csv_path)
    summary_rows.append(_summarize_best(pooled_df, "All electrodes pooled"))
    if args.save_plots or args.show_plots:
        plot_path = output_dir / "ridge_pooled.png" if args.save_plots else None
        _save_plot(pooled_results, plot_path, show=args.show_plots, title_suffix="All electrodes pooled")

    # 8. Save summary metadata
    summary_path = output_dir / "ridge_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "config": {
                    "subjects": subject_ids,
                    "lags_ms": lags_ms.tolist(),
                    "alphas": alphas,
                    "cv_splits": args.cv_splits,
                    "window_ms": None if args.disable_window else args.window_ms,
                    "hop_ms": None if args.disable_window else args.hop_ms,
                    "sampling_rate": pooled_sr,
                    "audio_path": str(args.audio_path),
                    "data_root": str(args.data_root),
                },
                "per_subject_files": {
                    str(sid): str((output_dir / f"ridge_subject_{sid:02d}.csv").resolve())
                    for sid in per_subject_results
                },
                "average_file": str(avg_csv_path.resolve()),
                "pooled_file": str(pooled_csv_path.resolve()),
                "summary": summary_rows,
            },
            fp,
            indent=2,
        )
    print(f"Summary written to {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ridge latency sweeps across all subjects.")
    parser.add_argument("--audio-path", type=Path, default=Path("data/stimuli/podcast.wav"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(DEFAULT_SUBJECT_IDS))
    parser.add_argument("--target-sr", type=float, default=512.0, help="Target sampling rate for neural/audio alignment")
    parser.add_argument("--audio-lowpass", type=float, default=8.0, help="Low-pass cutoff (Hz) for the audio envelope")
    parser.add_argument("--lags", type=str, default="-1000:1000:10", help="Lag sweep specification (start:stop:step in ms)")
    parser.add_argument("--alphas", type=str, default=None, help="Optional comma-separated ridge alphas")
    parser.add_argument("--cv-splits", type=int, default=10, help="Number of CV folds for ridge regression")
    parser.add_argument("--window-ms", type=float, default=200.0, help="Sliding window size in milliseconds")
    parser.add_argument("--hop-ms", type=float, default=25.0, help="Sliding window hop in milliseconds")
    parser.add_argument("--disable-window", action="store_true", help="Disable sliding-window RMS aggregation")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. 'cpu' or 'cuda')")
    parser.add_argument("--verbose", action="store_true", help="Print verbose ridge diagnostics")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ridge_batch"))
    parser.add_argument("--save-plots", action="store_true", help="Save ridge diagnostic plots to disk")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_batch_pipeline(args)


if __name__ == "__main__":
    main()
