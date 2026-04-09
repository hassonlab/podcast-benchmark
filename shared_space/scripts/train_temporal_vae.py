"""
Multi-Patient Temporal VAE training script.

Loss = L_recon + alpha * L_cross + beta * L_kl
  L_recon = (1/N) * sum_i MSE(decoder_i(z_avg), x_i_norm)
  L_cross = (1/N(N-1)) * sum_{i≠j} MSE(decoder_j(mu_i), x_j_norm)
  L_kl    = KL( N(mu_avg, exp(logvar_avg)) || N(0,I) )

Usage (run from podcast-benchmark/ root):
    python shared_space/scripts/train_temporal_vae.py --config shared_space/configs/podcast_temporal_vae.yml
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from core.config import DataParams
from utils.data_utils import load_raws, read_subject_mapping, read_electrode_file
from shared_space.models.patient_temporal_vae import MultiPatientTemporalVAE


# ---------------------------------------------------------------------------
# Helpers (copied from train_vae.py)
# ---------------------------------------------------------------------------

def window_average_np(data: np.ndarray, num_average_samples: int) -> np.ndarray:
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

def _parse_overrides(unknown_args: list) -> dict:
    """Parse --key=value or --section.key=value into a nested dict."""
    overrides = {}
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            if key and val:
                overrides[key] = yaml.safe_load(val)
    return overrides


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply flat dot-notation overrides to a nested dict in-place."""
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Patient Temporal VAE")
    parser.add_argument("--config", required=True)
    args, unknown_args = parser.parse_known_args()
    overrides = _parse_overrides(unknown_args)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        print(f"Applying CLI overrides: {overrides}")
        _apply_overrides(cfg, overrides)

    data_cfg   = cfg["data_params"]
    tvae_cfg   = cfg["tvae_params"]
    train_cfg  = cfg["training_params"]
    ckpt_path  = cfg["checkpoint_path"]

    shared_channels  = tvae_cfg.get("shared_channels", 8)
    enc_ch           = tvae_cfg.get("enc_ch", 32)
    dec_ch           = tvae_cfg.get("dec_ch", 32)
    input_timesteps  = tvae_cfg.get("input_timesteps", 10)
    beta             = tvae_cfg.get("beta", 0.1)
    alpha            = tvae_cfg.get("alpha", 1.0)
    dropout_p        = float(tvae_cfg.get("dropout_p", 0.0))
    num_avg_samples  = data_cfg.get("num_average_samples", 32)

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
        electrode_map  = read_electrode_file(data_cfg["electrode_file_path"], subject_mapping=subject_id_map)
        data_params.subject_ids            = data_cfg.get("subject_ids") or list(electrode_map.keys())
        data_params.per_subject_electrodes = electrode_map
    else:
        data_params.subject_ids = data_cfg.get("subject_ids", [])

    print(f"Loading data for subjects: {data_params.subject_ids}")
    raws = load_raws(data_params)
    print(f"Loaded {len(raws)} patients")

    raw_arrays, subject_labels = [], []
    for i, raw in enumerate(raws):
        arr = raw.get_data()
        raw_arrays.append(arr)
        sub_id = data_params.subject_ids[i] if i < len(data_params.subject_ids) else i + 1
        subject_labels.append(f"sub-{sub_id:02d}")
        print(f"  {subject_labels[-1]}: {arr.shape[0]} electrodes, {arr.shape[1]} samples")

    sfreq      = raws[0].info["sfreq"]
    window_width = data_cfg.get("window_width", 0.625)
    half_win   = int(window_width / 2 * sfreq)

    # ------------------------------------------------------------------
    # 2. Normalization stats
    # ------------------------------------------------------------------
    norm_means, norm_stds = [], []
    for arr in raw_arrays:
        mean = arr.mean(axis=1)
        std  = arr.std(axis=1).clip(min=1e-6)
        norm_means.append(torch.tensor(mean, dtype=torch.float32))
        norm_stds.append(torch.tensor(std,  dtype=torch.float32))

    # ------------------------------------------------------------------
    # 3. Sample windows at stride
    # ------------------------------------------------------------------
    stride_ms      = train_cfg.get("vae_stride_ms", 50)
    stride_samples = max(1, int(stride_ms / 1000 * sfreq))
    min_duration   = min(arr.shape[1] for arr in raw_arrays)
    center_samples = list(range(half_win, min_duration - half_win, stride_samples))
    print(f"\nTimepoint sampling: stride={stride_ms}ms, n_windows={len(center_samples)}")

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    n_electrodes_list = [arr.shape[0] for arr in raw_arrays]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MultiPatientTemporalVAE(
        n_electrodes_list=n_electrodes_list,
        enc_ch=enc_ch,
        dec_ch=dec_ch,
        shared_channels=shared_channels,
        input_timesteps=input_timesteps,
        dropout_p=dropout_p,
    ).to(device)

    model.set_normalization_stats(norm_means, norm_stds)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TemporalVAE parameters: {total_params:,}")
    print(f"  shared_channels={shared_channels}, enc_ch={enc_ch}, alpha={alpha}, beta={beta}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    batch_size = train_cfg.get("batch_size", 128)
    n_epochs   = train_cfg.get("epochs", 100)
    N          = len(raws)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Training TemporalVAE: {n_epochs} epochs, batch_size={batch_size}")
    print(f"Loss = L_recon + {alpha}*L_cross + {beta}*L_kl")
    print(f"{'='*65}\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss_acc   = 0.0
        l_recon_acc      = 0.0
        l_cross_acc      = 0.0
        l_kl_acc         = 0.0
        n_batches        = 0

        shuffled_centers = [center_samples[i] for i in np.random.permutation(len(center_samples))]

        for batch_start in range(0, len(shuffled_centers), batch_size):
            batch_centers = shuffled_centers[batch_start : batch_start + batch_size]
            if not batch_centers:
                continue

            xs_batch = extract_batch_windows(
                raw_arrays, batch_centers, half_win, num_avg_samples, device
            )

            x_recs_norm, xs_norm, mu_avg, logvar_avg, cross_recs, _mus = model(xs_batch)

            # L_recon: reconstruction from z_avg
            l_recon = torch.stack([
                F.mse_loss(rec, xn) for rec, xn in zip(x_recs_norm, xs_norm)
            ]).mean()

            # L_cross: cross-reconstruction (i≠j)
            cross_list = [
                F.mse_loss(cross_recs[(j, i)], xs_norm[j])
                for i in range(N) for j in range(N) if i != j
            ]
            l_cross = torch.stack(cross_list).mean() if cross_list else torch.tensor(0.0, device=device)

            # L_kl: KL divergence on mu_avg / logvar_avg
            l_kl = -0.5 * (1 + logvar_avg - mu_avg.pow(2) - logvar_avg.exp()).mean()

            loss = l_recon + alpha * l_cross + beta * l_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_acc += loss.item()
            l_recon_acc    += l_recon.item()
            l_cross_acc    += l_cross.item()
            l_kl_acc       += l_kl.item()
            n_batches      += 1

        if n_batches == 0:
            continue

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Loss: {total_loss_acc/n_batches:.4f} | "
            f"L_recon: {l_recon_acc/n_batches:.4f} | "
            f"L_cross: {l_cross_acc/n_batches:.4f} | "
            f"L_kl: {l_kl_acc/n_batches:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    model.save(ckpt_path)
    print(f"\nCheckpoint saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
