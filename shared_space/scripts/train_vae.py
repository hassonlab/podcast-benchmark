"""
Standalone VAE training script.

Usage (run from podcast-benchmark/ root):
    python shared_space/scripts/train_vae.py --config shared_space/configs/podcast_vae.yml

The script:
    1. Loads iEEG data for all patients
    2. Computes per-electrode z-score normalization stats (from the full recording)
    3. Samples random timepoints across the full podcast at configurable stride
    4. Trains a MultiPatientVAE with per-epoch logging of MSE (normalized) and KL
    5. Saves the checkpoint (including normalization stats)

No word-locking is used — training covers any timepoint in the podcast.
Loss is computed in normalized space so all patients contribute equally.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow imports from the podcast-benchmark root (parent of shared_space)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import yaml

from core.config import DataParams
from utils.data_utils import load_raws, read_subject_mapping, read_electrode_file
from shared_space.models.patient_vae import MultiPatientVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def window_average_np(data: np.ndarray, num_average_samples: int) -> np.ndarray:
    """Average consecutive samples into bins.

    Args:
        data: (batch, n_electrodes, n_samples)
        num_average_samples: samples per bin

    Returns:
        (batch, n_electrodes, n_bins)
    """
    n = data.shape[2]
    n_keep = (n // num_average_samples) * num_average_samples
    trimmed = data[:, :, :n_keep]
    reshaped = trimmed.reshape(trimmed.shape[0], trimmed.shape[1], -1, num_average_samples)
    return reshaped.mean(-1)


def extract_batch_windows(
    raw_arrays: list[np.ndarray],
    center_samples: list[int],
    half_win: int,
    num_average_samples: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract and preprocess per-patient windows for a batch of timepoints.

    Returns:
        list of (batch_size, n_elec_i, n_bins) tensors, one per patient
    """
    xs = []
    for arr in raw_arrays:
        windows = np.stack(
            [arr[:, c - half_win : c + half_win] for c in center_samples],
            axis=0,
        )  # (batch, n_elec, 2*half_win)
        windows_binned = window_average_np(windows, num_average_samples)
        xs.append(torch.tensor(windows_binned, dtype=torch.float32).to(device))
    return xs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Patient VAE")
    parser.add_argument("--config", required=True, help="Path to podcast_vae.yml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg   = cfg["data_params"]
    vae_cfg    = cfg["vae_params"]
    train_cfg  = cfg["training_params"]
    checkpoint_path = cfg["checkpoint_path"]

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
        data_params.subject_ids         = data_cfg.get("subject_ids") or list(electrode_map.keys())
        data_params.per_subject_electrodes = electrode_map
    else:
        data_params.subject_ids = data_cfg.get("subject_ids", [])

    print(f"Loading data for subjects: {data_params.subject_ids}")
    raws = load_raws(data_params)
    print(f"Loaded {len(raws)} patients")

    # ------------------------------------------------------------------
    # 2. Preload raw data arrays
    # ------------------------------------------------------------------
    raw_arrays     = []
    subject_labels = []
    for i, raw in enumerate(raws):
        arr = raw.get_data()  # (n_elec, n_times)
        raw_arrays.append(arr)
        sub_id = data_params.subject_ids[i] if i < len(data_params.subject_ids) else i + 1
        subject_labels.append(f"sub-{sub_id:02d}")
        print(f"  {subject_labels[-1]}: {arr.shape[0]} electrodes, {arr.shape[1]} samples")

    sfreq              = raws[0].info["sfreq"]
    window_width       = data_cfg.get("window_width", 0.625)
    half_win           = int(window_width / 2 * sfreq)
    num_average_samples = data_cfg.get("num_average_samples", 32)
    input_timesteps    = vae_cfg.get(
        "input_timesteps",
        int(np.floor(window_width * sfreq) // num_average_samples),
    )

    print(
        f"\nWindow: {window_width}s = {2*half_win} samples → "
        f"{input_timesteps} bins of {num_average_samples} samples each"
    )

    # ------------------------------------------------------------------
    # 3. Compute per-electrode normalization stats (from full recording)
    # ------------------------------------------------------------------
    print("\nComputing per-electrode normalization statistics...")
    norm_means, norm_stds = [], []
    for i, (arr, label) in enumerate(zip(raw_arrays, subject_labels)):
        # arr: (n_elec, n_times) — use full recording for statistics
        mean = arr.mean(axis=1)   # (n_elec,)
        std  = arr.std(axis=1)    # (n_elec,)
        norm_means.append(torch.tensor(mean, dtype=torch.float32))
        norm_stds.append(torch.tensor(std,  dtype=torch.float32))
        print(
            f"  {label}: "
            f"mean={mean.mean():.4e}, "
            f"std={std.mean():.4e}  "
            f"(min_std={std.min():.4e})"
        )

    # ------------------------------------------------------------------
    # 4. Valid center samples (valid for ALL patients)
    # ------------------------------------------------------------------
    stride_ms      = train_cfg.get("vae_stride_ms", 50)
    stride_samples = max(1, int(stride_ms / 1000 * sfreq))
    min_duration   = min(arr.shape[1] for arr in raw_arrays)
    center_samples = list(range(half_win, min_duration - half_win, stride_samples))
    print(f"\nTimepoint sampling: stride={stride_ms}ms ({stride_samples} samples)")
    print(f"Valid center samples: {len(center_samples)}")

    # ------------------------------------------------------------------
    # 5. Build VAE and set normalization stats
    # ------------------------------------------------------------------
    n_electrodes_list = [arr.shape[0] for arr in raw_arrays]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    vae = MultiPatientVAE(
        n_electrodes_list=n_electrodes_list,
        encoder_channels=vae_cfg.get("encoder_channels", [64, 32]),
        decoder_channels=vae_cfg.get("decoder_channels", [32, 64]),
        latent_dim=vae_cfg.get("latent_dim", 64),
        input_timesteps=input_timesteps,
    ).to(device)

    vae.set_normalization_stats(norm_means, norm_stds)

    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"VAE parameters: {total_params:,}")
    print(f"  latent_dim={vae_cfg.get('latent_dim', 64)}, beta={vae_cfg.get('beta', 1.0)}")
    print(f"  encoder_channels={vae_cfg.get('encoder_channels', [64, 32])}")
    print(f"  decoder_channels={vae_cfg.get('decoder_channels', [32, 64])}")

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    batch_size = train_cfg.get("batch_size", 64)
    n_epochs   = train_cfg.get("epochs", 50)
    beta       = vae_cfg.get("beta", 1.0)
    n_patients = len(raws)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Starting VAE training: {n_epochs} epochs, batch_size={batch_size}")
    print(f"MSE is computed in z-score normalized space (should reach ~0)")
    print(f"{'='*65}\n")

    for epoch in range(n_epochs):
        vae.train()
        epoch_loss             = 0.0
        epoch_kl               = 0.0
        epoch_mse_per_patient  = [0.0] * n_patients
        n_batches              = 0

        # Shuffle every epoch
        shuffled_idx     = np.random.permutation(len(center_samples))
        shuffled_centers = [center_samples[i] for i in shuffled_idx]

        for batch_start in range(0, len(shuffled_centers), batch_size):
            batch_centers = shuffled_centers[batch_start : batch_start + batch_size]
            if not batch_centers:
                continue

            xs = extract_batch_windows(
                raw_arrays, batch_centers, half_win, num_average_samples, device
            )

            optimizer.zero_grad()
            x_recs_norm, xs_norm, mu_avg, log_var_avg = vae(xs)
            total_loss, mse_avg, mse_per_patient, kl = vae.vae_loss(
                x_recs_norm, xs_norm, mu_avg, log_var_avg, beta=beta
            )
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_kl   += kl
            for i, mse in enumerate(mse_per_patient):
                epoch_mse_per_patient[i] += mse
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        avg_kl   = epoch_kl   / n_batches
        avg_mse  = sum(epoch_mse_per_patient) / (n_batches * n_patients)

        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"KL: {avg_kl:.4f} | "
            f"MSE avg (norm): {avg_mse:.4f}"
        )
        for i, label in enumerate(subject_labels):
            n_elec = n_electrodes_list[i]
            mse_i  = epoch_mse_per_patient[i] / n_batches
            print(f"  {label} ({n_elec:2d} elec): MSE(norm)={mse_i:.4f}")

    # ------------------------------------------------------------------
    # 7. Save checkpoint
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    vae.save(checkpoint_path)
    print(f"\nCheckpoint saved → {checkpoint_path}")


if __name__ == "__main__":
    main()
