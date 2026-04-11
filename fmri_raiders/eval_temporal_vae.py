#!/usr/bin/env python3
"""
Evaluate Raider temporal VAE alongside BrainIAK tutorial 11 conventions (no BrainIAK required).

**Movie TSM:** Chronological halves (z-scored voxels per half). **SRM** for TSM is **always**
fit on movie **half 1 only**, **k=50** shared features (capped by voxels), then half 2 is
transformed with the same ``W`` (held-out movie TRs for SRM). **VAE** latents use **μ_avg**
(mean over encoders, mean-pooled) for each half; use ``--checkpoint-tsm`` with a **first_half**
checkpoint for held-out half-2 VAE TSM.

**Image LOO:** **Full-movie** protocol — SRM is fit on the **entire** movie, then image runs are
transformed (**k=50**). Use a **full_movie** VAE checkpoint for image encoding.

**Two VAE checkpoints (recommended):** pass **both** ``--checkpoint-tsm`` and ``--checkpoint-image``
(synonyms: ``--model-tsm``, ``--model-image``) — one network trained for movie TSM (typically
``raider_temporal_vae.pt``, first half of the movie), and a **separate** network for image transfer
(typically ``raider_temporal_vae_full_movie.pt``). For a single shared weights file only, pass
``--checkpoint`` once (uses that file for both tasks).

SRM baseline needs BrainIAK (``pip install -e ".[fmri]"``). Use ``--no-compare-srm`` to skip.
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
# BrainIAK SRM latent dimension for TSM + image (fixed; capped by voxel count).
SRM_SHARED_FEATURES = 50


def _resolve_ckpt(path: str | None) -> str | None:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


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
    parser = argparse.ArgumentParser(
        description="Evaluate Raider temporal VAE (TSM + image LOO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two-model example (recommended):\n"
            "  python fmri_raiders/eval_temporal_vae.py \\\n"
            "    --checkpoint-tsm fmri_raiders/checkpoints/raider_temporal_vae.pt \\\n"
            "    --checkpoint-image fmri_raiders/checkpoints/raider_temporal_vae_full_movie.pt \\\n"
            "    --data-dir data/raider --device auto\n"
            "Same using aliases: --model-tsm … --model-image …"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="One .pt used for **both** TSM and image (only if you do not pass the two paths below)",
    )
    parser.add_argument(
        "--checkpoint-tsm",
        "--model-tsm",
        default=None,
        dest="checkpoint_tsm",
        metavar="PATH",
        help="VAE checkpoint **only** for movie TSM (e.g. first_half / raider_temporal_vae.pt)",
    )
    parser.add_argument(
        "--checkpoint-image",
        "--model-image",
        default=None,
        dest="checkpoint_image",
        metavar="PATH",
        help="VAE checkpoint **only** for image LOO (e.g. full_movie / raider_temporal_vae_full_movie.pt)",
    )
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
    path_single = _resolve_ckpt(args.checkpoint)
    path_tsm = _resolve_ckpt(args.checkpoint_tsm)
    path_img = _resolve_ckpt(args.checkpoint_image)
    if path_tsm and path_img:
        ckpt_tsm, ckpt_img = path_tsm, path_img
    elif path_single and not path_tsm and not path_img:
        ckpt_tsm = ckpt_img = path_single
    elif path_single and (path_tsm or path_img):
        parser.error("Do not mix --checkpoint with --checkpoint-tsm / --checkpoint-image; pick one style.")
    elif path_tsm or path_img:
        parser.error(
            "Two models: pass **both** --checkpoint-tsm and --checkpoint-image "
            "(or --model-tsm and --model-image). "
            "Or pass only --checkpoint to use one file for both tasks."
        )
    else:
        parser.error(
            "Provide --checkpoint-tsm **and** --checkpoint-image (two checkpoints), "
            "or a single --checkpoint for both TSM and image."
        )

    device = pick_device(args.device)
    model_tsm, meta_tsm = load_raider_vae_checkpoint(ckpt_tsm, map_location=str(device))
    model_tsm.to(device)
    if ckpt_tsm == ckpt_img:
        model_img, meta_img = model_tsm, meta_tsm
    else:
        model_img, meta_img = load_raider_vae_checkpoint(ckpt_img, map_location=str(device))
        model_img.to(device)

    half_tr = int(meta_tsm.get("half_tr", 15))
    num_average_samples = int(meta_tsm.get("num_average_samples", 3))
    vae_tsm_train = meta_tsm.get("movie_split_for_training", "unknown")
    tsm_feat_dim = int(model_tsm.shared_channels)

    half_tr_img = int(meta_img.get("half_tr", 15))
    num_av_img = int(meta_img.get("num_average_samples", 3))
    vae_img_train = meta_img.get("movie_split_for_training", "unknown")

    movie_data, vox_num, n_tr, num_subs = raider_data.load_movie(data_dir)
    srm_k = min(SRM_SHARED_FEATURES, vox_num)

    cli_style.banner("evaluation")
    if vae_tsm_train == "first_half":
        _tsm_note = "TSM VAE: half 2 held out from VAE if ckpt is first_half"
    elif vae_tsm_train == "full":
        _tsm_note = "TSM VAE: full-movie ckpt — half 2 seen in VAE loss (SRM TSM still half-1 fit)"
    else:
        _tsm_note = "Unknown VAE TSM checkpoint train split"

    _kv = [
        ("Checkpoint · TSM", ckpt_tsm),
        ("Checkpoint · image LOO", ckpt_img),
        ("SRM shared features k", str(srm_k)),
        ("Window (TSM ckpt)", f"{half_tr} / {num_average_samples}"),
        ("Window (image ckpt)", f"{half_tr_img} / {num_av_img}"),
        ("movie_split_for_training · TSM ckpt", vae_tsm_train),
        ("movie_split_for_training · image ckpt", vae_img_train),
        ("VAE latent (eval)", "μ_avg only, mean-pooled → k"),
        ("VAE k · TSM", str(tsm_feat_dim)),
        ("VAE k · image", str(int(model_img.shared_channels))),
        ("TSM protocol", "SRM always fit movie half 1; half 2 test"),
        ("Image SRM / VAE protocol", "SRM on full movie; VAE from image ckpt"),
        ("TSM note (VAE)", _tsm_note),
    ]
    if ckpt_tsm != ckpt_img:
        _kv.insert(
            2,
            ("Note", "TSM uses --checkpoint-tsm; image LOO uses --checkpoint-image"),
        )
    cli_style.key_value_block(_kv)
    cli_style.rule()

    train_list, test_list = raider_data.split_half_movie(movie_data, num_subs)

    raw_train = [x.copy() for x in train_list]
    raw_test = [x.copy() for x in test_list]
    raider_data.zscore_voxelwise(raw_train)
    raider_data.zscore_voxelwise(raw_test)

    vae_train = [train_list[s].astype(np.float32) for s in range(num_subs)]
    vae_test = [test_list[s].astype(np.float32) for s in range(num_subs)]

    print(
        f"\n  {cli_style.cyan('…')} "
        f"{cli_style.dim('Encoding TSM latents (movie half 1 & 2) with TSM checkpoint')}"
    )
    lat_train = encode_per_subject_latent_timeseries(
        model_tsm, vae_train, half_tr, num_average_samples, device
    )
    lat_test = encode_per_subject_latent_timeseries(
        model_tsm, vae_test, half_tr, num_average_samples, device
    )

    z_lat_tr = [x.copy() for x in lat_train]
    z_lat_te = [x.copy() for x in lat_test]
    zscore_time_features(z_lat_tr)
    zscore_time_features(z_lat_te)

    cli_style.rule(title="Time-segment matching (BrainIAK tutorial 11 style)")
    print(
        cli_style.dim(
            "  Raw: z-scored within each movie half. SRM: always fit on **half 1 only** (k=50), "
            "transform half 2 (held out for SRM). VAE: encoded with TSM checkpoint."
        )
    )
    if vae_tsm_train == "first_half":
        print(
            cli_style.dim(
                "  VAE half 2: not in VAE training loss (first_half checkpoint)."
            )
        )
    elif vae_tsm_train == "full":
        print(
            cli_style.bold("  Note: ")
            + cli_style.dim(
                "TSM checkpoint is full_movie — half 2 was in the VAE loss. "
                "Use --checkpoint-tsm with raider_temporal_vae.pt for held-out half-2 VAE TSM."
            )
        )

    acc_raw_tr = time_segment_matching_numpy(raw_train, win_size=TSM_WINDOW)
    acc_vae_tr = time_segment_matching_numpy(z_lat_tr, win_size=TSM_WINDOW)
    acc_raw_te = time_segment_matching_numpy(raw_test, win_size=TSM_WINDOW)
    acc_vae_te = time_segment_matching_numpy(z_lat_te, win_size=TSM_WINDOW)
    d_h2 = acc_vae_te.mean() - acc_raw_te.mean()

    acc_srm_tr = acc_srm_te = None
    compare_srm = not args.no_compare_srm
    if compare_srm:
        try:
            from fmri_raiders.srm_raider import fit_srm_movie_halves

            _sh_tr, _sh_te = fit_srm_movie_halves(
                "first_half",
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
        return f"SRM shared · movie half 1 (k={srm_k}; fit on half 1 only)"

    def _srm_half2_label() -> str:
        return f"SRM shared · movie half 2 (k={srm_k}; same W, half 2 not in SRM fit)"

    vae_h1 = "VAE latent · movie half 1"
    vae_h2 = "VAE latent · movie half 2"
    if vae_tsm_train == "first_half":
        vae_h1 += " (VAE train split)"
        vae_h2 += " (held out from VAE)"
    elif vae_tsm_train == "full":
        vae_h1 += " (chronological)"
        vae_h2 += " (chronological; in VAE train)"

    tsm_rows = [
        ("Raw voxels · movie half 1", f"{acc_raw_tr.mean():.4f}", f"{acc_raw_tr.std():.4f}"),
        (vae_h1, f"{acc_vae_tr.mean():.4f}", f"{acc_vae_tr.std():.4f}"),
    ]
    if acc_srm_tr is not None:
        tsm_rows.append((_srm_half1_label(), f"{acc_srm_tr.mean():.4f}", f"{acc_srm_tr.std():.4f}"))
    tsm_rows.extend(
        [
            ("Raw voxels · movie half 2", f"{acc_raw_te.mean():.4f}", f"{acc_raw_te.std():.4f}"),
            (vae_h2, f"{acc_vae_te.mean():.4f}", f"{acc_vae_te.std():.4f}"),
        ]
    )
    if acc_srm_te is not None:
        tsm_rows.append((_srm_half2_label(), f"{acc_srm_te.mean():.4f}", f"{acc_srm_te.std():.4f}"))

    cli_style.metrics_table(("Condition", "Mean", "Std"), tsm_rows)

    sign = "+" if d_h2 >= 0 else ""
    print(
        f"  {cli_style.bold('Half 2 TSM — mean (VAE latent − raw voxels):')} "
        f"{cli_style.cyan(f'{sign}{d_h2:.4f}')}"
    )
    if vae_tsm_train == "first_half":
        print(
            "  "
            + cli_style.dim(
                "Positive ⇒ better segment matching in VAE latent vs raw on held-out movie TRs "
                "(VAE + SRM half 2 both unseen at train time for their respective fits)."
            )
        )
    if acc_srm_te is not None:
        d_srm = acc_srm_te.mean() - acc_raw_te.mean()
        sign_s = "+" if d_srm >= 0 else ""
        print(
            f"  {cli_style.bold('Half 2 TSM — mean (SRM shared − raw voxels):')} "
            f"{cli_style.cyan(f'{sign_s}{d_srm:.4f}')}"
        )

    cli_style.rule(title="Image runs · LOO Nu-SVM")
    print(
        cli_style.dim(
            "  SRM: fit on **full movie** (k=50), transform image runs (tutorial §7). "
            "VAE: latents from **image** checkpoint (full_movie recommended)."
        )
    )
    image_np, labels = raider_data.load_image_and_labels(data_dir)
    n_img = min(num_subs, image_np.shape[2], movie_data.shape[2])
    image_np = image_np[:, :, :n_img]
    tr_img = image_np.shape[1]
    if vae_img_train != "full":
        print(
            cli_style.dim(
                "  Warning: image checkpoint is not full_movie — image LOO still runs, "
                "but protocol matches tutorial best with raider_temporal_vae_full_movie.pt."
            )
        )
    if tr_img < 2 * half_tr_img + 1:
        print(
            f"  {cli_style.dim(f'Skip: image TRs {tr_img} < 2·half_tr+1 for image ckpt half_tr={half_tr_img}')}"
        )
    else:
        lat_img = encode_per_subject_latent_timeseries(
            model_img,
            [image_np[:, :, s] for s in range(n_img)],
            half_tr_img,
            num_av_img,
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
        if compare_srm:
            try:
                from fmri_raiders.srm_raider import fit_srm_for_images

                shared_img = fit_srm_for_images(
                    "full",
                    movie_data[:, :, :n_img],
                    [image_np[:, :, s] for s in range(n_img)],
                    n_img,
                    vox_num,
                    features=srm_k,
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
                    f"SRM shared (LOO, k={srm_k})",
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
