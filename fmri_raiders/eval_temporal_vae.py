#!/usr/bin/env python3
"""
Evaluate Raider temporal VAE alongside BrainIAK tutorial 11 conventions (no BrainIAK required).

Movie TSM: chronological split (first / second half of TRs). Raw voxels are z-scored within each
half (tutorial). VAE latents use **μ_avg** per window (mean over subjects), mean-pooled over
window time → **k** features per TR.

SRM baseline (BrainIAK): fits SRM with the same movie split as the checkpoint. Use
``--no-compare-srm`` if BrainIAK/MPI is unavailable.

Usage::

    python fmri_raiders/eval_temporal_vae.py \\
        --checkpoint fmri_raiders/checkpoints/raider_temporal_vae.pt \\
        --data-dir data/raider
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

from fmri_raiders import cli_style
from fmri_raiders import raider_data
from fmri_raiders.torch_device import pick_device
from fmri_raiders.tutorial_algorithms import image_class_prediction, time_segment_matching_numpy
from fmri_raiders.vae_windows import extract_batch_windows_fmri
from shared_space.models.patient_temporal_vae import MultiPatientTemporalVAE

TSM_WINDOW = 10
ENCODE_BATCH = 1024
SRM_N_ITER = 20


def load_raider_vae_checkpoint(path: str, map_location: str | None = None):
    try:
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location or "cpu")
    model = MultiPatientTemporalVAE(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = ckpt.get("raider_window", {})
    return model, meta


@torch.no_grad()
def encode_per_subject_latent_timeseries(
    model: MultiPatientTemporalVAE,
    arrays: list[np.ndarray],
    half_tr: int,
    num_average_samples: int,
    device: torch.device,
    center_batch: int = ENCODE_BATCH,
) -> list[np.ndarray]:
    """Raw BOLD (vox, T) → per-subject (k, T); each subject gets the same **μ_avg** trajectory."""
    t = arrays[0].shape[1]
    k = int(model.shared_channels)
    out = [np.zeros((k, t), dtype=np.float32) for _ in arrays]
    centers = list(range(half_tr, t - half_tr))
    model.eval()
    for start in range(0, len(centers), center_batch):
        chunk = centers[start : start + center_batch]
        xs = extract_batch_windows_fmri(arrays, chunk, half_tr, num_average_samples, device)
        xs_norm = [model._normalize(i, x) for i, x in enumerate(xs)]
        mus = [model.encoders[i](xs_norm[i])[0] for i in range(len(arrays))]
        mu_avg = torch.stack(mus, dim=0).mean(0)
        pooled = mu_avg.mean(dim=-1).cpu().numpy()
        for i in range(len(arrays)):
            for j, c in enumerate(chunk):
                out[i][:, c] = pooled[j]

    for i in range(len(arrays)):
        out[i][:, :half_tr] = out[i][:, half_tr : half_tr + 1]
        out[i][:, t - half_tr :] = out[i][:, t - half_tr - 1 : t - half_tr]
    return out


def zscore_time_features(data_list: list[np.ndarray]) -> None:
    from scipy import stats

    for i in range(len(data_list)):
        data_list[i] = np.nan_to_num(stats.zscore(data_list[i], axis=1, ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Raider temporal VAE (TSM + image LOO)")
    parser.add_argument("--checkpoint", required=True, help="Path from train_temporal_vae.py")
    parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data", "raider"))
    parser.add_argument(
        "--no-compare-srm",
        action="store_true",
        help="Skip BrainIAK SRM baseline (TSM + image LOO)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="cuda (default): require GPU; auto: GPU if available else CPU; cpu: CPU only",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(_PROJECT_ROOT, args.data_dir)
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_PROJECT_ROOT, args.checkpoint)

    device = pick_device(args.device)
    model, meta = load_raider_vae_checkpoint(ckpt_path, map_location=str(device))
    model.to(device)

    half_tr = int(meta.get("half_tr", 15))
    num_average_samples = int(meta.get("num_average_samples", 3))
    train_mode = meta.get("movie_split_for_training", "unknown")
    tsm_feat_dim = int(model.shared_channels)

    cli_style.banner("evaluation")
    if train_mode == "first_half":
        _tsm_note = "TSM half 2 = held out from VAE (tutorial movie protocol)"
    elif train_mode == "full":
        _tsm_note = "TSM is not held-out generalization (VAE saw all TRs)"
    else:
        _tsm_note = "Unknown train split in checkpoint; treat TSM rows cautiously"

    cli_style.key_value_block(
        [
            ("Checkpoint", ckpt_path),
            ("Window (half_TR / bin)", f"{half_tr} / {num_average_samples}"),
            ("movie_split_for_training", train_mode),
            ("VAE latent (eval)", "μ_avg only, mean-pooled → k"),
            ("VAE TSM feature dim", str(tsm_feat_dim)),
            ("TSM interpretation", _tsm_note),
        ]
    )
    cli_style.rule()

    movie_data, vox_num, n_tr, num_subs = raider_data.load_movie(data_dir)
    train_list, test_list = raider_data.split_half_movie(movie_data, num_subs)

    raw_train = [x.copy() for x in train_list]
    raw_test = [x.copy() for x in test_list]
    raider_data.zscore_voxelwise(raw_train)
    raider_data.zscore_voxelwise(raw_test)

    vae_train = [train_list[s].astype(np.float32) for s in range(num_subs)]
    vae_test = [test_list[s].astype(np.float32) for s in range(num_subs)]

    print(
        f"\n  {cli_style.cyan('…')} "
        f"{cli_style.dim('Encoding latents for movie half 1 and half 2 (chronological split)')}"
    )
    lat_train = encode_per_subject_latent_timeseries(
        model, vae_train, half_tr, num_average_samples, device
    )
    lat_test = encode_per_subject_latent_timeseries(
        model, vae_test, half_tr, num_average_samples, device
    )

    z_lat_tr = [x.copy() for x in lat_train]
    z_lat_te = [x.copy() for x in lat_test]
    zscore_time_features(z_lat_tr)
    zscore_time_features(z_lat_te)

    cli_style.rule(title="Time-segment matching (BrainIAK tutorial 11 style)")
    if train_mode == "first_half":
        print(
            cli_style.dim(
                "  Like the tutorial: raw data are z-scored within each movie half separately. "
                "VAE was trained only on movie half 1; half 2 never appeared in the loss. "
                "Compare VAE vs raw on half 2 for generalization to new movie TRs."
            )
        )
    elif train_mode == "full":
        print(
            cli_style.bold("  Note: ")
            + cli_style.dim(
                "This checkpoint used movie_split_for_training=full (all TRs). "
                "Both chronological halves were in VAE training; half 2 is not held out. "
                "TSM rows below are descriptive only. For tutorial-style held-out TSM, train with "
                "raider_temporal_vae.yml (first_half) and evaluate that checkpoint."
            )
        )
    else:
        print(
            cli_style.dim(
                f"  Checkpoint missing or unknown movie_split_for_training ({train_mode!r}). "
                "Cannot assert held-out protocol; verify how the model was trained."
            )
        )

    acc_raw_tr = time_segment_matching_numpy(raw_train, win_size=TSM_WINDOW)
    acc_vae_tr = time_segment_matching_numpy(z_lat_tr, win_size=TSM_WINDOW)
    acc_raw_te = time_segment_matching_numpy(raw_test, win_size=TSM_WINDOW)
    acc_vae_te = time_segment_matching_numpy(z_lat_te, win_size=TSM_WINDOW)
    d_h2 = acc_vae_te.mean() - acc_raw_te.mean()

    acc_srm_tr = acc_srm_te = None
    srm_k = None
    compare_srm = not args.no_compare_srm
    if compare_srm and train_mode in ("first_half", "full"):
        try:
            from fmri_raiders.srm_raider import fit_srm_movie_halves

            srm_k = tsm_feat_dim
            _sh_tr, _sh_te = fit_srm_movie_halves(
                train_mode,
                movie_data,
                num_subs,
                vox_num,
                raw_train,
                raw_test,
                features=srm_k,
                n_iter=SRM_N_ITER,
            )
            acc_srm_tr = time_segment_matching_numpy(_sh_tr, win_size=TSM_WINDOW)
            acc_srm_te = time_segment_matching_numpy(_sh_te, win_size=TSM_WINDOW)
        except (ImportError, RuntimeError) as e:
            print(f"  {cli_style.dim(f'Skip SRM baseline: {e}')}")

    def _srm_half1_label() -> str:
        assert srm_k is not None
        if train_mode == "first_half":
            return f"SRM shared · movie half 1 (k={srm_k}; fit on half 1 only)"
        return f"SRM shared · movie half 1 (k={srm_k}; full-movie SRM)"

    def _srm_half2_label() -> str:
        assert srm_k is not None
        if train_mode == "first_half":
            return f"SRM shared · movie half 2 (k={srm_k}; same W, half 2 not in SRM fit)"
        return f"SRM shared · movie half 2 (k={srm_k}; full-movie SRM)"

    if train_mode == "first_half":
        tsm_rows = [
            (
                "Raw voxels · movie half 1 (VAE train split)",
                f"{acc_raw_tr.mean():.4f}",
                f"{acc_raw_tr.std():.4f}",
            ),
            (
                "VAE latent · movie half 1",
                f"{acc_vae_tr.mean():.4f}",
                f"{acc_vae_tr.std():.4f}",
            ),
        ]
        if acc_srm_tr is not None:
            tsm_rows.append(
                (
                    _srm_half1_label(),
                    f"{acc_srm_tr.mean():.4f}",
                    f"{acc_srm_tr.std():.4f}",
                )
            )
        tsm_rows.extend(
            [
                (
                    "Raw voxels · movie half 2 (held out from VAE)",
                    f"{acc_raw_te.mean():.4f}",
                    f"{acc_raw_te.std():.4f}",
                ),
                (
                    "VAE latent · movie half 2",
                    f"{acc_vae_te.mean():.4f}",
                    f"{acc_vae_te.std():.4f}",
                ),
            ]
        )
        if acc_srm_te is not None:
            tsm_rows.append(
                (
                    _srm_half2_label(),
                    f"{acc_srm_te.mean():.4f}",
                    f"{acc_srm_te.std():.4f}",
                )
            )
    elif train_mode == "full":
        tsm_rows = [
            (
                "Raw voxels · movie half 1 (chronological)",
                f"{acc_raw_tr.mean():.4f}",
                f"{acc_raw_tr.std():.4f}",
            ),
            (
                "VAE latent · movie half 1",
                f"{acc_vae_tr.mean():.4f}",
                f"{acc_vae_tr.std():.4f}",
            ),
        ]
        if acc_srm_tr is not None:
            tsm_rows.append(
                (
                    _srm_half1_label(),
                    f"{acc_srm_tr.mean():.4f}",
                    f"{acc_srm_tr.std():.4f}",
                )
            )
        tsm_rows.extend(
            [
                (
                    "Raw voxels · movie half 2 (chronological; also in VAE train)",
                    f"{acc_raw_te.mean():.4f}",
                    f"{acc_raw_te.std():.4f}",
                ),
                (
                    "VAE latent · movie half 2",
                    f"{acc_vae_te.mean():.4f}",
                    f"{acc_vae_te.std():.4f}",
                ),
            ]
        )
        if acc_srm_te is not None:
            tsm_rows.append(
                (
                    _srm_half2_label(),
                    f"{acc_srm_te.mean():.4f}",
                    f"{acc_srm_te.std():.4f}",
                )
            )
    else:
        tsm_rows = [
            ("Raw voxels · movie half 1", f"{acc_raw_tr.mean():.4f}", f"{acc_raw_tr.std():.4f}"),
            ("VAE latent · movie half 1", f"{acc_vae_tr.mean():.4f}", f"{acc_vae_tr.std():.4f}"),
        ]
        if acc_srm_tr is not None:
            tsm_rows.append(
                (
                    f"SRM shared · movie half 1 (k={srm_k})",
                    f"{acc_srm_tr.mean():.4f}",
                    f"{acc_srm_tr.std():.4f}",
                )
            )
        tsm_rows.extend(
            [
                ("Raw voxels · movie half 2", f"{acc_raw_te.mean():.4f}", f"{acc_raw_te.std():.4f}"),
                ("VAE latent · movie half 2", f"{acc_vae_te.mean():.4f}", f"{acc_vae_te.std():.4f}"),
            ]
        )
        if acc_srm_te is not None:
            tsm_rows.append(
                (
                    f"SRM shared · movie half 2 (k={srm_k})",
                    f"{acc_srm_te.mean():.4f}",
                    f"{acc_srm_te.std():.4f}",
                )
            )

    cli_style.metrics_table(("Condition", "Mean", "Std"), tsm_rows)

    if train_mode == "first_half":
        sign = "+" if d_h2 >= 0 else ""
        print(
            f"  {cli_style.bold('Half 2 TSM — mean (VAE latent − raw voxels):')} "
            f"{cli_style.cyan(f'{sign}{d_h2:.4f}')}"
        )
        print(
            "  "
            + cli_style.dim(
                "Positive ⇒ better segment matching in shared latent vs raw on held-out movie TRs."
            )
        )
        if acc_srm_te is not None:
            d_srm = acc_srm_te.mean() - acc_raw_te.mean()
            sign_s = "+" if d_srm >= 0 else ""
            print(
                f"  {cli_style.bold('Half 2 TSM — mean (SRM shared − raw voxels):')} "
                f"{cli_style.cyan(f'{sign_s}{d_srm:.4f}')}"
            )
    elif train_mode == "full":
        print(
            "  "
            + cli_style.dim(
                "No held-out movie-half summary for this checkpoint (full-movie training). "
                f"Half 2 mean (VAE − raw) = {d_h2:+.4f} (descriptive only, not generalization)."
            )
        )
    else:
        print(
            "  "
            + cli_style.dim(
                f"Half 2 mean (VAE − raw) = {d_h2:+.4f}; interpret only after confirming train split."
            )
        )

    cli_style.rule(title="Image runs · LOO Nu-SVM")
    image_np, labels = raider_data.load_image_and_labels(data_dir)
    n_img = min(num_subs, image_np.shape[2], movie_data.shape[2])
    image_np = image_np[:, :, :n_img]
    tr_img = image_np.shape[1]
    if train_mode == "full":
        print(
            cli_style.dim(
                "  Matches tutorial image exercise: VAE trained on full movie, then image runs encoded."
            )
        )
    else:
        print(
            cli_style.dim(
                "  Tutorial image exercise uses **full-movie** VAE; this checkpoint is first_half only. "
                "Retrain with raider_temporal_vae_full_movie.yml for comparable image transfer."
            )
        )
    if tr_img < 2 * half_tr + 1:
        print(f"  {cli_style.dim(f'Skip: image TRs {tr_img} < 2·half_tr+1 for half_tr={half_tr}')}")
    else:
        lat_img = encode_per_subject_latent_timeseries(
            model,
            [image_np[:, :, s] for s in range(n_img)],
            half_tr,
            num_average_samples,
            device,
        )
        z_img = [x.copy() for x in lat_img]
        zscore_time_features(z_img)
        lab = labels[:tr_img]
        acc_vae_img = image_class_prediction(z_img, lab)

        i_list = [image_np[:, :, s].astype(np.float32) for s in range(n_img)]
        raider_data.zscore_voxelwise(i_list)
        acc_raw_img = image_class_prediction(i_list, lab)

        acc_srm_img = None
        if compare_srm and train_mode in ("first_half", "full"):
            try:
                from fmri_raiders.srm_raider import fit_srm_for_images

                shared_img = fit_srm_for_images(
                    train_mode,
                    movie_data[:, :, :n_img],
                    [image_np[:, :, s] for s in range(n_img)],
                    n_img,
                    vox_num,
                    features=tsm_feat_dim,
                    n_iter=SRM_N_ITER,
                )
                acc_srm_img = image_class_prediction(shared_img, lab)
            except (ImportError, RuntimeError) as e:
                print(f"  {cli_style.dim(f'Skip SRM image LOO: {e}')}")

        img_rows = [
            ("Raw voxels (LOO)", f"{acc_raw_img.mean():.4f}", f"{acc_raw_img.std():.4f}"),
            ("VAE latent (LOO)", f"{acc_vae_img.mean():.4f}", f"{acc_vae_img.std():.4f}"),
        ]
        if acc_srm_img is not None:
            img_rows.append(
                (
                    f"SRM shared (LOO, k={tsm_feat_dim})",
                    f"{acc_srm_img.mean():.4f}",
                    f"{acc_srm_img.std():.4f}",
                )
            )
        img_rows.append(("Chance (1/7)", "0.1429", "—"))

        cli_style.metrics_table(("Model", "Mean acc.", "Std"), img_rows)
        raw_s = "  ".join(f"{x:.3f}" for x in acc_raw_img)
        vae_s = "  ".join(f"{x:.3f}" for x in acc_vae_img)
        print(f"  {cli_style.dim('Per-subject raw')}  {raw_s}")
        print(f"  {cli_style.dim('Per-subject VAE')}  {vae_s}")
        if acc_srm_img is not None:
            srm_s = "  ".join(f"{x:.3f}" for x in acc_srm_img)
            print(f"  {cli_style.dim('Per-subject SRM')}  {srm_s}")


if __name__ == "__main__":
    main()
    cli_style.done_line("Evaluation finished")
    print()
