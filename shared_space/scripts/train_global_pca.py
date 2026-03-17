"""
Global PCA training script.

Usage (run from podcast-benchmark/ root):
    python shared_space/scripts/train_global_pca.py --config shared_space/configs/podcast_global_pca.yml

Concatenates ALL subjects' electrodes into one big vector (183 dims total),
fits PCA jointly, and saves the projection matrix W (183 x k).

At evaluation time, reconstruction is:
    x_rec = W @ W^T @ x_norm  (denormalize afterwards)

This is a "global" baseline: no per-patient structure, no cross-patient alignment.
It just finds the top-k directions of variance in the full concatenated brain activity.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import yaml
from sklearn.decomposition import PCA

from core.config import DataParams
from utils.data_utils import load_raws, read_subject_mapping, read_electrode_file


def main():
    parser = argparse.ArgumentParser(description="Train Global PCA (concatenate all electrodes)")
    parser.add_argument("--config", required=True, help="Path to podcast_global_pca.yml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg        = cfg["data_params"]
    pca_cfg         = cfg.get("pca_params", {})
    train_cfg       = cfg.get("training_params", {})
    checkpoint_path = cfg["checkpoint_path"]

    features = pca_cfg.get("features", 50)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_params = DataParams(
        data_root=data_cfg.get("data_root", "data"),
        use_high_gamma=data_cfg.get("use_high_gamma", True),
        window_width=data_cfg.get("window_width", 0.625),
        do_drop_bads=data_cfg.get("do_drop_bads", True),
    )

    if data_cfg.get("electrode_file_path"):
        subject_id_map = read_subject_mapping("data/participants.tsv", delimiter="\t")
        electrode_map  = read_electrode_file(
            data_cfg["electrode_file_path"], subject_mapping=subject_id_map
        )
        data_params.subject_ids          = data_cfg.get("subject_ids") or list(electrode_map.keys())
        data_params.per_subject_electrodes = electrode_map
    else:
        data_params.subject_ids = data_cfg.get("subject_ids", [])

    print(f"Loading data for subjects: {data_params.subject_ids}")
    raws = load_raws(data_params)
    print(f"Loaded {len(raws)} patients")

    # ------------------------------------------------------------------
    # 2. Preload raw arrays
    # ------------------------------------------------------------------
    raw_arrays     = []
    subject_labels = []
    for i, raw in enumerate(raws):
        arr = raw.get_data()
        raw_arrays.append(arr)
        sub_id = data_params.subject_ids[i] if i < len(data_params.subject_ids) else i + 1
        subject_labels.append(f"sub-{sub_id:02d}")
        print(f"  {subject_labels[-1]}: {arr.shape[0]} electrodes, {arr.shape[1]} samples")

    sfreq        = raws[0].info["sfreq"]
    window_width = data_cfg.get("window_width", 0.625)
    half_win     = int(window_width / 2 * sfreq)

    # ------------------------------------------------------------------
    # 3. Per-electrode normalization stats
    # ------------------------------------------------------------------
    print("\nComputing per-electrode normalization statistics...")
    norm_means, norm_stds = [], []
    for arr, label in zip(raw_arrays, subject_labels):
        mean = arr.mean(axis=1)
        std  = arr.std(axis=1).clip(min=1e-6)
        norm_means.append(mean)
        norm_stds.append(std)
        print(f"  {label}: mean={mean.mean():.4e}, std={std.mean():.4e}")

    # Flat arrays across all electrodes in concatenation order
    norm_means_flat = np.concatenate(norm_means)   # (n_elec_total,)
    norm_stds_flat  = np.concatenate(norm_stds)    # (n_elec_total,)

    # ------------------------------------------------------------------
    # 4. Normalize and sample timepoints
    # ------------------------------------------------------------------
    normed_arrays = [
        (arr - mean[:, None]) / std[:, None]
        for arr, mean, std in zip(raw_arrays, norm_means, norm_stds)
    ]

    stride_ms      = train_cfg.get("vae_stride_ms", 50)
    stride_samples = max(1, int(stride_ms / 1000 * sfreq))
    min_duration   = min(arr.shape[1] for arr in raw_arrays)
    center_samples = list(range(half_win, min_duration - half_win, stride_samples))
    print(f"\nTimepoint sampling: stride={stride_ms}ms ({stride_samples} samples)")
    print(f"Valid center samples: {len(center_samples)}")

    # ------------------------------------------------------------------
    # 5. Build global observation matrix: (n_elec_total, n_samples)
    #    Stack all patients' electrodes vertically → one big electrode space
    # ------------------------------------------------------------------
    global_obs = np.vstack(
        [normed[:, center_samples] for normed in normed_arrays]
    )  # (n_elec_total, n_samples)
    n_elec_total = global_obs.shape[0]
    print(f"\nGlobal observation matrix: {global_obs.shape}")
    print(f"  Total electrodes: {n_elec_total}")

    # ------------------------------------------------------------------
    # 6. Fit PCA on (n_samples, n_elec_total)
    # ------------------------------------------------------------------
    features = min(features, n_elec_total)
    print(f"\nFitting PCA: features={features} / {n_elec_total} total electrodes...")

    pca = PCA(n_components=features, random_state=42)
    pca.fit(global_obs.T)   # sklearn: (n_samples, n_features)

    W = pca.components_.T   # (n_elec_total, features)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"Variance explained by top-{features} components: {var_explained:.3f}")

    # Reconstruction quality
    P   = W @ W.T                          # (n_elec_total, n_elec_total)
    rec = P @ global_obs                   # (n_elec_total, n_samples)
    mse = float(np.mean((global_obs - rec) ** 2))
    var = float(np.mean(global_obs ** 2))
    print(f"Global reconstruction: MSE={mse:.4f}  var={var:.4f}  fraction_explained={1 - mse/var:.3f}")

    # Per-patient breakdown
    print("\nPer-patient reconstruction quality:")
    offset = 0
    for arr_norm, label in zip(normed_arrays, subject_labels):
        n = arr_norm.shape[0]
        orig = global_obs[offset:offset + n, :]
        recon = rec[offset:offset + n, :]
        mse_p = float(np.mean((orig - recon) ** 2))
        var_p = float(np.mean(orig ** 2))
        print(f"  {label}: fraction_explained={1 - mse_p/var_p:.3f}")
        offset += n

    # ------------------------------------------------------------------
    # 7. Save checkpoint
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    checkpoint = {
        "W":               W.tolist(),
        "n_electrodes_list": [arr.shape[0] for arr in raw_arrays],
        "features":        features,
        "norm_means_flat": norm_means_flat.tolist(),
        "norm_stds_flat":  norm_stds_flat.tolist(),
        "var_explained":   var_explained,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\nCheckpoint saved → {checkpoint_path}")


if __name__ == "__main__":
    main()
