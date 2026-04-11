#!/usr/bin/env python3
"""
Run BrainIAK tutorial 11–style analyses on the Raider fMRI dataset (Haxby et al., 2011).

Data (preprocessed numpy, no nibabel required for Raider):
  Only the small Raider archive (~32 MB) is needed — not the full brainiak_datasets.zip::

    python fmri_raiders/download_raider_data.py

  Manual: https://zenodo.org/records/2598755 — file ``raider.zip``.

  Default path: ``<repo>/data/raider`` (override with --data-dir or env RAIDER_DATA_DIR).

Reference: https://brainiak.org/tutorials/11-SRM/
Rendered notebook: https://brainiak.org/notebooks/tutorials/html/11-srm.html

Dependencies::
  pip install -e ".[fmri]"

Usage::
  python fmri_raiders/run_tutorial.py --data-dir data/raider --out-dir results/raider_srm
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
import scipy.spatial.distance as sp_distance  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fmri_raiders import raider_data  # noqa: E402
from fmri_raiders.tutorial_algorithms import (  # noqa: E402
    image_class_prediction,
    time_segment_matching,
)


def _require_brainiak():
    try:
        import brainiak.funcalign.srm  # noqa: F401
        from brainiak.isc import isc  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Install BrainIAK for SRM/ISC: pip install -e \".[fmri]\"  (or pip install brainiak)"
        ) from e
    except RuntimeError as e:
        err = str(e).lower()
        if "mpi" in err or "libmpi" in err:
            raise RuntimeError(
                "brainiak/mpi4py needs the system MPI runtime. On Ubuntu/Debian try:\n"
                "  sudo apt install libopenmpi3 openmpi-bin\n"
                "Then open a new shell and run this script again.\n"
                f"(Original error: {e})"
            ) from e
        raise


def _stack_isc_format(data_list):
    """List of (n_feat, n_time) -> (n_feat, n_time, n_subj)."""
    a = np.zeros((data_list[0].shape[0], data_list[0].shape[1], len(data_list)))
    for ppt, x in enumerate(data_list):
        a[:, :, ppt] = x
    return a


def run_isc_raw_vs_shared(train_data, shared_train):
    from brainiak.isc import isc

    raw_obj = _stack_isc_format(train_data)
    corr_raw = np.nan_to_num(isc(raw_obj, summary_statistic="mean"))
    shared_obj = _stack_isc_format(shared_train)
    corr_shared = np.nan_to_num(isc(shared_obj, summary_statistic="mean"))
    tstat = stats.ttest_ind(np.arctanh(corr_shared), np.arctanh(corr_raw))
    return corr_raw, corr_shared, tstat


def plot_isc_hist(corr_raw, corr_shared, tstat, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].set_title("ISC for all voxels (raw)")
    axes[0].hist(corr_raw.ravel(), bins=30)
    axes[0].set_xlabel("correlation")
    axes[0].set_xlim([-1, 1])
    axes[1].set_title("ISC for shared features (SRM)")
    axes[1].hist(corr_shared.ravel(), bins=30)
    axes[1].set_xlabel("correlation")
    axes[1].set_xlim([-1, 1])
    fig.suptitle(
        f"Independent t-test (Fisher z): t={tstat.statistic:.3f}, p={tstat.pvalue:.3e}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reconstruction_isc_test(test_data, shared_test, w_matrices):
    """Per-subject W_i @ shared_i; ISC in voxel space (tutorial §4, corrected W per subject)."""
    from brainiak.isc import isc

    signal_srm = np.zeros(
        (test_data[0].shape[0], test_data[0].shape[1], len(test_data))
    )
    for ppt in range(len(test_data)):
        signal_srm[:, :, ppt] = w_matrices[ppt] @ shared_test[ppt]
    corr_reconstructed = np.nan_to_num(isc(signal_srm, summary_statistic="mean"))
    raw_obj = _stack_isc_format(test_data)
    corr_raw = np.nan_to_num(isc(raw_obj, summary_statistic="mean"))
    t_dep = stats.ttest_1samp(
        np.arctanh(corr_reconstructed) - np.arctanh(corr_raw), 0
    )
    return corr_reconstructed, corr_raw, t_dep


def main():
    parser = argparse.ArgumentParser(description="BrainIAK SRM tutorial flows on Raider .npy data")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("RAIDER_DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "raider")),
        help="Directory with movie.npy, image.npy, label.npy",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "results", "raider_srm"),
        help="Figures and logs",
    )
    parser.add_argument("--features", type=int, default=50, help="SRM latent dim (tutorial default 50)")
    parser.add_argument("--n-iter", type=int, default=20, help="SRM iterations")
    parser.add_argument(
        "--tsm-window",
        type=int,
        default=10,
        help="TRs per window for time-segment matching",
    )
    parser.add_argument(
        "--image-srm-features",
        type=int,
        default=50,
        help="SRM features for movie→image transfer (tutorial exercise 9; default 50 like --features)",
    )
    parser.add_argument("--skip-image", action="store_true", help="Skip §7 image classification")
    args = parser.parse_args()

    _require_brainiak()
    import brainiak.funcalign.srm as srm_module

    os.makedirs(args.out_dir, exist_ok=True)
    raider_dir = os.path.abspath(args.data_dir)
    print(f"Raider data directory: {raider_dir}")

    movie_data, vox_num, n_tr, num_subs = raider_data.load_movie(raider_dir)
    print(f"  movie.npy: {vox_num} voxels × {n_tr} TRs × {num_subs} subjects")

    train_data, test_data = raider_data.split_half_movie(movie_data, num_subs)
    raider_data.zscore_voxelwise(train_data)
    raider_data.zscore_voxelwise(test_data)

    features = min(args.features, vox_num)
    srm = srm_module.SRM(n_iter=args.n_iter, features=features)
    print(f"Fitting SRM (features={features}, n_iter={args.n_iter}) …")
    srm.fit(train_data)
    print(f"  shared response s_.shape = {srm.s_.shape}")

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.imshow(srm.s_, aspect="auto", cmap="viridis")
    ax.set_title("SRM: features × time (train half)")
    ax.set_xlabel("TR")
    ax.set_ylabel("feature")
    fig.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "srm_shared_response.png"), dpi=150)
    plt.close(fig)

    dist_mat = sp_distance.squareform(sp_distance.pdist(srm.s_.T))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(dist_mat, cmap="viridis", aspect="auto")
    ax.set_title("Distance between TR pairs in shared space")
    ax.set_xlabel("TR")
    ax.set_ylabel("TR")
    fig.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "srm_tr_distance_matrix.png"), dpi=150)
    plt.close(fig)

    shared_train = srm.transform(train_data)
    for sub in range(num_subs):
        shared_train[sub] = stats.zscore(shared_train[sub], axis=1, ddof=1)
        shared_train[sub] = np.nan_to_num(shared_train[sub])

    corr_raw, corr_shared, tstat = run_isc_raw_vs_shared(train_data, shared_train)
    print(
        f"ISC train: independent t-test (Fisher z) raw vs shared: t={tstat.statistic:.3f}, p={tstat.pvalue:.3e}"
    )
    plot_isc_hist(
        corr_raw,
        corr_shared,
        tstat,
        os.path.join(args.out_dir, "isc_train_raw_vs_shared.png"),
    )

    shared_test = srm.transform(test_data)
    for sub in range(num_subs):
        shared_test[sub] = stats.zscore(shared_test[sub], axis=1, ddof=1)
        shared_test[sub] = np.nan_to_num(shared_test[sub])

    corr_raw_te, corr_shared_te, tstat_te = run_isc_raw_vs_shared(test_data, shared_test)
    print(
        f"ISC test:  independent t-test (Fisher z) raw vs shared: t={tstat_te.statistic:.3f}, p={tstat_te.pvalue:.3e}"
    )
    plot_isc_hist(
        corr_raw_te,
        corr_shared_te,
        tstat_te,
        os.path.join(args.out_dir, "isc_test_raw_vs_shared.png"),
    )

    w_list = srm.w_
    corr_rec, corr_raw_t, t_dep = reconstruction_isc_test(test_data, shared_test, w_list)
    print(
        f"Reconstruction ISC (test): dependent t-test (rec - raw) Fisher z: t={t_dep.statistic:.3f}, p={t_dep.pvalue:.3e}"
    )

    print("\nTime-segment matching (raw train, voxel space) …")
    acc_raw_tr = time_segment_matching(train_data, win_size=args.tsm_window)
    print(f"  per-subject: {acc_raw_tr}")
    print(f"  mean ± std: {acc_raw_tr.mean():.4f} ± {acc_raw_tr.std():.4f}")

    print("Time-segment matching (SRM train features) …")
    acc_srm_tr = time_segment_matching(shared_train, win_size=args.tsm_window)
    print(f"  per-subject: {acc_srm_tr}")
    print(f"  mean ± std: {acc_srm_tr.mean():.4f} ± {acc_srm_tr.std():.4f}")

    print("Time-segment matching (raw test) …")
    acc_raw_te = time_segment_matching(test_data, win_size=args.tsm_window)
    print(f"  mean ± std: {acc_raw_te.mean():.4f} ± {acc_raw_te.std():.4f}")

    print("Time-segment matching (SRM test features) …")
    acc_srm_te = time_segment_matching(shared_test, win_size=args.tsm_window)
    print(f"  mean ± std: {acc_srm_te.mean():.4f} ± {acc_srm_te.std():.4f}")

    names = ["raw train", "SRM train", "raw test", "SRM test"]
    acc_groups = [acc_raw_tr, acc_srm_tr, acc_raw_te, acc_srm_te]
    means = np.array([a.mean() for a in acc_groups])
    se = np.array([a.std(ddof=1) / np.sqrt(len(a)) for a in acc_groups])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=se, capsize=4, color=["#4477aa", "#66ccee", "#ccbb44", "#ee8866"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Time-segment matching (mean ± SE across subjects)")
    ax.set_ylim(0, max(0.15, (means + se).max() * 1.15))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "time_segment_matching_summary.png"), dpi=150)
    plt.close(fig)

    if not args.skip_image:
        print("\nImage class prediction (SRM fit on full movie, test on image runs) …")
        image_np, labels = raider_data.load_image_and_labels(raider_dir)
        movie_full, _, _, _ = raider_data.load_movie(raider_dir)
        n_img_subs = min(num_subs, image_np.shape[2], movie_full.shape[2])
        movie_full = movie_full[:, :, :n_img_subs]
        image_np = image_np[:, :, :n_img_subs]

        tr_img = image_np.shape[1]
        if labels.shape[0] != tr_img:
            print(
                f"  WARNING: label.npy length {labels.shape[0]} != image TRs {tr_img}; check data bundle."
            )

        m_list, i_list = raider_data.movie_and_image_lists_for_classification(
            movie_full, image_np, n_img_subs
        )
        raider_data.zscore_voxelwise(m_list)
        raider_data.zscore_voxelwise(i_list)

        k_img = min(args.image_srm_features, vox_num)
        srm_img = srm_module.SRM(n_iter=args.n_iter, features=k_img)
        srm_img.fit(m_list)
        shared_image = srm_img.transform(i_list)
        for sub in range(n_img_subs):
            shared_image[sub] = np.nan_to_num(
                stats.zscore(shared_image[sub], axis=1, ddof=1)
            )

        acc_srm_img = image_class_prediction(shared_image, labels[:tr_img])
        print(f"  SRM + NuSVC LOO subject accuracy: {acc_srm_img}")
        print(f"  mean ± std: {acc_srm_img.mean():.4f} ± {acc_srm_img.std():.4f}")

        acc_raw_img = image_class_prediction(i_list, labels[:tr_img])
        print(f"  Raw (z-scored) LOO subject accuracy: {acc_raw_img}")
        print(f"  mean ± std: {acc_raw_img.mean():.4f} ± {acc_raw_img.std():.4f}")

    print(f"\nDone. Figures under: {args.out_dir}")


if __name__ == "__main__":
    main()
