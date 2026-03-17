"""
Standalone SRM training script.

Usage (run from podcast-benchmark/ root):
    python shared_space/scripts/train_srm.py --config shared_space/configs/podcast_srm.yml

The script:
    1. Loads iEEG data for all patients
    2. Computes per-electrode z-score normalization stats (from the full recording)
    3. Samples timepoints across the full podcast at configurable stride
    4. Fits a Shared Response Model (SRM) on the normalized single-sample observations
    5. Saves the checkpoint (W matrices + normalization stats)

Reconstruction logic:
    For patient i with W_i (n_elec_i × features):
        x_rec = W_i @ W_i.T @ x_norm  →  denormalize  →  x_rec_original_scale
    This projects each patient's data onto the shared subspace (removing noise
    orthogonal to the shared response).
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

from core.config import DataParams
from utils.data_utils import load_raws, read_subject_mapping, read_electrode_file


def main():
    parser = argparse.ArgumentParser(description="Train SRM (Shared Response Model)")
    parser.add_argument("--config", required=True, help="Path to podcast_srm.yml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg      = cfg["data_params"]
    srm_cfg       = cfg.get("srm_params", {})
    train_cfg     = cfg.get("training_params", {})
    checkpoint_path = cfg["checkpoint_path"]

    features = srm_cfg.get("features", 50)
    n_iter   = srm_cfg.get("n_iter", 10)

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
    # 3. Per-electrode normalization stats (from full recording)
    # ------------------------------------------------------------------
    print("\nComputing per-electrode normalization statistics...")
    norm_means, norm_stds = [], []
    for arr, label in zip(raw_arrays, subject_labels):
        mean = arr.mean(axis=1)
        std  = arr.std(axis=1).clip(min=1e-6)
        norm_means.append(mean)
        norm_stds.append(std)
        print(f"  {label}: mean={mean.mean():.4e}, std={std.mean():.4e}")

    # ------------------------------------------------------------------
    # 4. Z-score normalize
    # ------------------------------------------------------------------
    normed_arrays = [
        (arr - mean[:, None]) / std[:, None]
        for arr, mean, std in zip(raw_arrays, norm_means, norm_stds)
    ]

    # ------------------------------------------------------------------
    # 5. Sample timepoints (same stride logic as train_vae.py)
    # ------------------------------------------------------------------
    stride_ms      = train_cfg.get("vae_stride_ms", 50)
    stride_samples = max(1, int(stride_ms / 1000 * sfreq))
    min_duration   = min(arr.shape[1] for arr in raw_arrays)
    center_samples = list(range(half_win, min_duration - half_win, stride_samples))
    print(f"\nTimepoint sampling: stride={stride_ms}ms ({stride_samples} samples)")
    print(f"Valid center samples: {len(center_samples)}")

    # ------------------------------------------------------------------
    # 6. Build observation matrices: (n_elec_i, n_samples)
    #    SRM expects list of (n_features, n_observations) arrays
    # ------------------------------------------------------------------
    observations = []
    for i, normed in enumerate(normed_arrays):
        obs = normed[:, center_samples]   # (n_elec_i, n_samples)
        observations.append(obs)
        print(f"  {subject_labels[i]}: observation matrix {obs.shape}")

    # ------------------------------------------------------------------
    # 7. Fit SRM
    # ------------------------------------------------------------------
    min_elec = min(obs.shape[0] for obs in observations)
    if features > min_elec:
        print(f"  WARNING: features={features} > min electrodes={min_elec}. Capping at {min_elec}.")
        features = min_elec

    print(f"\nFitting SRM: features={features}, n_iter={n_iter}...")
    try:
        import brainiak.funcalign.srm as srm_module
    except ImportError:
        raise ImportError(
            "brainiak is required for SRM. Install with: pip install brainiak"
        )

    srm = srm_module.SRM(n_iter=n_iter, features=features)
    srm.fit(observations)

    W_matrices = srm.w_   # list of (n_elec_i, features) orthonormal matrices
    print("SRM fitting complete.")
    for W, label in zip(W_matrices, subject_labels):
        print(f"  {label}: W shape {W.shape}")

    # Quick sanity check: reconstruction quality
    print("\nReconstruction quality (normalized MSE per patient):")
    for i, (obs, W, label) in enumerate(zip(observations, W_matrices, subject_labels)):
        P   = W @ W.T                # (n_elec_i, n_elec_i) projection matrix
        rec = P @ obs                # (n_elec_i, n_samples)
        mse = np.mean((obs - rec) ** 2)
        var = np.mean(obs ** 2)
        print(f"  {label}: MSE={mse:.4f}  var={var:.4f}  fraction_explained={1 - mse/var:.3f}")

    # ------------------------------------------------------------------
    # 8. Save checkpoint
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    checkpoint = {
        "W":              [W.tolist() for W in W_matrices],
        "n_electrodes_list": [arr.shape[0] for arr in raw_arrays],
        "features":       features,
        "norm_means":     [m.tolist() for m in norm_means],
        "norm_stds":      [s.tolist() for s in norm_stds],
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\nCheckpoint saved → {checkpoint_path}")


if __name__ == "__main__":
    main()
