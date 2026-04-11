#!/usr/bin/env python3
"""
Train ``MultiPatientTemporalVAE`` on Pieman2 movie data (numpy, Raider-compatible layout).

This is the **Raider-style deep-learning** path, not BrainIAK tutorial 10 (ISC). For ISC on the
raw Pieman2 NIfTI tree, use ``fmri_pieman/run_tutorial.py``.

Same model and loss as ``fmri_raiders/train_temporal_vae.py``; only data paths and defaults differ.

Run::

    python fmri_pieman/train_temporal_vae.py --config fmri_pieman/configs/pieman_temporal_vae.yml

Requires: torch (no BrainIAK). Checkpoint bundles ``raider_window`` metadata key for compatibility
with ``fmri_pieman/eval_temporal_vae.py`` (same key as Raider eval).

**GPU memory:** Pieman ``movie.npy`` has ~100k masked voxels per subject; the VAE’s first Conv1d is
over the voxel axis, so width must stay small (see ``configs/pieman_temporal_vae*.yml``). If you
still hit CUDA OOM, use ``--device cpu`` or lower ``batch_size`` / ``enc_ch`` / ``dec_ch`` in YAML.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from fmri_pieman import pieman_data
from fmri_raiders import cli_style
from fmri_raiders.torch_device import pick_device
from fmri_raiders.vae_windows import assert_window_shapes, extract_batch_windows_fmri
from shared_space.models.patient_temporal_vae import MultiPatientTemporalVAE


def _batch_losses(
    model: MultiPatientTemporalVAE,
    xs_batch: list[torch.Tensor],
    N: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_recs_norm, xs_norm, mu_avg, logvar_avg, cross_recs, _mus = model(xs_batch)
    l_recon = torch.stack(
        [F.mse_loss(rec, xn) for rec, xn in zip(x_recs_norm, xs_norm)]
    ).mean()
    cross_list = [
        F.mse_loss(cross_recs[(j, i)], xs_norm[j])
        for i in range(N)
        for j in range(N)
        if i != j
    ]
    l_cross = torch.stack(cross_list).mean() if cross_list else torch.tensor(0.0, device=device)
    l_kl = -0.5 * (1 + logvar_avg - mu_avg.pow(2) - logvar_avg.exp()).mean()
    loss = l_recon + alpha * l_cross + beta * l_kl
    return loss, l_recon, l_cross, l_kl


def _parse_overrides(unknown_args: list) -> dict:
    overrides = {}
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            if key and val:
                overrides[key] = yaml.safe_load(val)
    return overrides


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal VAE on Pieman fMRI")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="cuda (default): require GPU; auto: GPU if available else CPU; cpu: CPU only",
    )
    parser.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
    args, unknown_args = parser.parse_known_args()
    if args.cpu:
        args.device = "cpu"
    overrides = _parse_overrides(unknown_args)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cli_style.rule(title="CLI overrides")
        for k, v in overrides.items():
            print(f"  {cli_style.dim(k)} → {v}")
        print()
        _apply_overrides(cfg, overrides)

    data_cfg = cfg["data_params"]
    tvae_cfg = cfg["tvae_params"]
    train_cfg = cfg["training_params"]
    ckpt_path = cfg["checkpoint_path"]

    pd = data_cfg.get("pieman_dir", "data/pieman")
    pieman_dir = pd if os.path.isabs(pd) else os.path.join(_PROJECT_ROOT, pd)

    movie_data, vox_num, n_tr, num_subs = pieman_data.load_movie(pieman_dir)
    split_mode = data_cfg.get("movie_split_for_training", "first_half")
    if split_mode == "first_half":
        half_movie = n_tr // 2
        movie_slice = movie_data[:, :half_movie, :]
        n_tr_use = half_movie
    elif split_mode == "full":
        movie_slice = movie_data
        n_tr_use = n_tr
    else:
        raise ValueError("movie_split_for_training must be 'first_half' or 'full'")

    raw_arrays = [movie_slice[:, :, sub].astype(np.float32, copy=False) for sub in range(num_subs)]

    half_tr = int(data_cfg["half_tr"])
    num_average_samples = int(data_cfg["num_average_samples"])
    stride_tr = int(data_cfg.get("stride_tr", 5))
    shared_channels = int(tvae_cfg.get("shared_channels", 8))
    enc_ch = int(tvae_cfg.get("enc_ch", 32))
    dec_ch = int(tvae_cfg.get("dec_ch", 32))
    input_timesteps = int(tvae_cfg.get("input_timesteps", 10))
    beta = float(tvae_cfg.get("beta", 0.1))
    alpha = float(tvae_cfg.get("alpha", 1.0))

    assert_window_shapes(half_tr, num_average_samples, input_timesteps)

    norm_means, norm_stds = [], []
    for arr in raw_arrays:
        mean = arr.mean(axis=1)
        std = arr.std(axis=1).clip(min=1e-6)
        norm_means.append(torch.tensor(mean, dtype=torch.float32))
        norm_stds.append(torch.tensor(std, dtype=torch.float32))

    center_trs = list(range(half_tr, n_tr_use - half_tr, stride_tr))
    if not center_trs:
        raise ValueError(
            f"No window centers: n_TR_use={n_tr_use}, half_tr={half_tr}. "
            "Reduce half_tr or use more TRs."
        )

    n_electrodes_list = [arr.shape[0] for arr in raw_arrays]
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cli_style.banner("training", title="Pieman temporal VAE")
    _protocol = (
        "First half of TRs for training (half 2 held out for TSM-style eval)"
        if split_mode == "first_half"
        else "Full movie TRs for training"
    )
    _norm_label = (
        "full movie (all training TRs)"
        if split_mode == "full"
        else "first-half TRs only (voxel μ/σ; half 2 excluded)"
    )
    cli_style.key_value_block(
        [
            ("Device", str(device)),
            ("Protocol", _protocol),
            ("Data dir", pieman_dir),
            ("Voxels × TR × subjects", f"{vox_num} × {n_tr} × {num_subs}"),
            ("Training span", f"{n_tr_use} TR ({split_mode})"),
            ("Window centers", str(len(center_trs))),
            ("Stride (TR)", str(stride_tr)),
            ("Norm scope", _norm_label),
            ("Latent k × T", f"{shared_channels} × {input_timesteps}"),
            ("Channels enc / dec", f"{enc_ch} / {dec_ch}"),
        ]
    )
    cli_style.rule()
    print(
        f"  {cli_style.dim('Loss')}  "
        f"L_recon + {alpha}·L_cross + {beta}·L_kl (same as Raider / podcast temporal VAE)"
    )
    cli_style.rule()

    model = MultiPatientTemporalVAE(
        n_electrodes_list=n_electrodes_list,
        enc_ch=enc_ch,
        dec_ch=dec_ch,
        shared_channels=shared_channels,
        input_timesteps=input_timesteps,
        dropout_p=0.0,
    ).to(device)
    model.set_normalization_stats(norm_means, norm_stds)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    batch_size = int(train_cfg.get("batch_size", 64))
    n_epochs = int(train_cfg.get("epochs", 50))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    N = num_subs

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = l_recon_acc = l_cross_acc = l_kl_acc = 0.0
        n_batches = 0
        shuffled = [center_trs[i] for i in np.random.permutation(len(center_trs))]

        for batch_start in range(0, len(shuffled), batch_size):
            batch_centers = shuffled[batch_start : batch_start + batch_size]
            if not batch_centers:
                continue
            xs_batch = extract_batch_windows_fmri(
                raw_arrays, batch_centers, half_tr, num_average_samples, device
            )

            loss, l_recon, l_cross, l_kl = _batch_losses(
                model, xs_batch, N, alpha, beta, device
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            l_recon_acc += l_recon.item()
            l_cross_acc += l_cross.item()
            l_kl_acc += l_kl.item()
            n_batches += 1

        if n_batches == 0:
            continue

        cli_style.epoch_line(
            epoch,
            n_epochs,
            total_loss / n_batches,
            l_recon_acc / n_batches,
            l_cross_acc / n_batches,
            l_kl_acc / n_batches,
        )

    out_path = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(_PROJECT_ROOT, ckpt_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    bundle = {
        "state_dict": model.state_dict(),
        "config": model._config,
        "raider_window": {
            "half_tr": half_tr,
            "num_average_samples": num_average_samples,
            "stride_tr": stride_tr,
            "movie_split_for_training": split_mode,
        },
        "dataset": "pieman",
    }
    torch.save(bundle, out_path)
    cli_style.done_line(f"Checkpoint saved → {out_path}")


if __name__ == "__main__":
    main()
