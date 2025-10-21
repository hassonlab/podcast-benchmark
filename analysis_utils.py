from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_outputs(results: Dict[str, Any], output_dir: Path, model_params: Dict[str, Any]) -> None:
    """Write per-subject/average/pooled CSVs and a combined summary.

    Keeps the same CSV conventions as the original CLI.
    """
    _ensure_dir(output_dir)
    records = []
    per_subject = results.get("per_subject", {})
    for sid, df in per_subject.items():
        df.to_csv(output_dir / f"subject_{int(sid):02d}_ridge.csv", index=False)
        tmp = df.copy()
        tmp["mode"] = "per_subject"
        records.append(tmp)

    for key, mode_name in (("average", "average"), ("pooled_electrodes", "pooled_electrodes")):
        if key in results:
            df = results[key].copy()
            if key == "average":
                df.insert(0, "subject_id", "mean")
            df["mode"] = mode_name
            df.to_csv(output_dir / f"{mode_name}_ridge.csv", index=False)
            records.append(df)

    if records:
        pd.concat(records, ignore_index=True).to_csv(
            output_dir / model_params.get("output_csv", "ridge_summary.csv"), index=False
        )


def save_audio_targets(targets, output_dir: Path, name: str = "audio_targets.npy") -> None:
    _ensure_dir(output_dir)
    arr = np.asarray(targets)
    np.save(output_dir / name, arr)
    pd.DataFrame({"target": arr.flatten()}).to_csv(output_dir / (name + ".csv"), index=False)


def aggregate_average(per_subject: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not per_subject:
        raise ValueError("No per-subject results provided for averaging.")
    combined = pd.concat(per_subject.values(), ignore_index=True)
    return combined.groupby("lag_ms", as_index=False).agg(
        r2=("r2", "mean"),
        r2_std=("r2", "std"),
        alpha_mean=("alpha", "mean"),
        train_r2=("train_r2", "mean"),
        n_subjects=("subject_id", "nunique"),
    )


def aggregate_pooled(datasets: list, lags_ms, cv_splits: int, alphas, device, ridge_fn) -> pd.DataFrame:
    if not datasets:
        raise ValueError("No datasets provided for pooled analysis.")
    pooled_neural = np.concatenate([entry["neural"] for entry in datasets], axis=0)
    pooled_audio = datasets[0]["audio"]
    pooled_sr = datasets[0]["effective_sr"]
    df = ridge_fn(pooled_neural, pooled_audio, pooled_sr, lags_ms, cv_splits, alphas, device)
    df["subject_id"] = "pooled"
    return df
