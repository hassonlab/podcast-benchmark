#!/usr/bin/env python3
"""
BrainIAK **tutorial 10** (ISC / Pieman2) on local NIfTI data — same loading and core analyses as the
notebook, not the Raider-style temporal VAE pipeline.

- **Mask:** ``masks/avg152T1_gray_3mm.nii.gz``
- **Tasks:** ``word`` (word-level scramble) and ``intact1`` (intact story), same as the tutorial
- **Pipeline:** ``io.load_images`` → ``mask_images`` → ``MaskedMultiSubjectData`` → ``isc(..., pairwise=False)``
  → ``permutation_isc`` on stacked ISC maps (optional; can reduce permutations for speed)

Data: extracted Zenodo Pieman2 tree (e.g. ``data/pieman/Pieman2``). See ``download_pieman_data.py``.

Reference: https://brainiak.org/notebooks/tutorials/html/10-isc.html

Dependencies::

  pip install -e ".[fmri]"

Usage::

  python fmri_pieman/run_tutorial.py --pieman2-dir data/pieman/Pieman2 --out-dir results/pieman_isc
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

ALL_TASK_NAMES = ("word", "intact1")


def _require_brainiak() -> None:
    try:
        from brainiak import image, io  # noqa: F401
        from brainiak.isc import isc, permutation_isc  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'Install BrainIAK:  pip install -e ".[fmri]"  (or pip install brainiak)'
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


def _subject_id_from_path(path: str) -> str | None:
    base = os.path.basename(path)
    m = re.match(r"sub-(\d+)-task-", base)
    return m.group(1) if m else None


def paired_task_files(pieman2_root: str, tasks: tuple[str, ...]) -> dict[str, list[str]]:
    """One file list per task, same subjects (intersection), sorted by subject id."""
    from collections import defaultdict

    by_sub: dict[str, dict[str, str]] = defaultdict(dict)
    for task in tasks:
        pattern = os.path.join(pieman2_root, "sub-*", "func", f"sub-*-task-{task}.nii.gz")
        for fp in sorted(glob.glob(pattern)):
            sid = _subject_id_from_path(fp)
            if sid is not None:
                by_sub[sid][task] = fp
    common = sorted(
        (s for s, m in by_sub.items() if len(m) == len(tasks)),
        key=lambda x: int(x),
    )
    if not common:
        raise FileNotFoundError(
            f"No subjects with all tasks {tasks} under {pieman2_root!r} "
            f"(glob …/sub-*/func/sub-*-task-<task>.nii.gz)"
        )
    return {t: [by_sub[s][t] for s in common] for t in tasks}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BrainIAK tutorial 10 (ISC) on Pieman2 NIfTI data"
    )
    parser.add_argument(
        "--pieman2-dir",
        default=os.environ.get(
            "PIEMAN2_DIR", os.path.join(_PROJECT_ROOT, "data", "pieman", "Pieman2")
        ),
        help="Pieman2 root (contains masks/ and sub-*/func/)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "results", "pieman_isc"),
        help="Figures and numpy summaries",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Use only first N subjects after pairing (0 = all)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="permutation_isc iterations (tutorial default 1000; try 100 for a quick run)",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip permutation_isc (only run voxelwise ISC per task)",
    )
    args = parser.parse_args()

    _require_brainiak()
    import nibabel as nib
    from brainiak import image, io
    from brainiak.isc import isc, permutation_isc

    pieman2_dir = os.path.abspath(args.pieman2_dir)
    if not os.path.isdir(pieman2_dir):
        raise SystemExit(f"Not a directory: {pieman2_dir}")

    mask_name = os.path.join(pieman2_dir, "masks", "avg152T1_gray_3mm.nii.gz")
    if not os.path.isfile(mask_name):
        raise SystemExit(f"Missing mask (tutorial layout): {mask_name}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fnames: dict[str, list[str]] = paired_task_files(pieman2_dir, ALL_TASK_NAMES)
    n_sub = len(next(iter(fnames.values())))
    if args.max_subjects and args.max_subjects < n_sub:
        for t in ALL_TASK_NAMES:
            fnames[t] = fnames[t][: args.max_subjects]
        n_sub = args.max_subjects

    print(f"Pieman2 directory: {pieman2_dir}")
    print(f"  Subjects (paired word + intact1): {n_sub}")
    for t in ALL_TASK_NAMES:
        print(f"  {t}: {len(fnames[t])} runs")

    brain_mask = io.load_boolean_mask(mask_name)
    coords = np.where(brain_mask)
    brain_nii = nib.load(mask_name)

    images: dict = {}
    masked_images: dict = {}
    bold: dict = {}
    group_assignment: list[int] = []
    group_assignment_dict = {task_name: i for i, task_name in enumerate(ALL_TASK_NAMES)}

    for task_name in ALL_TASK_NAMES:
        images[task_name] = io.load_images(fnames[task_name])
        masked_images[task_name] = image.mask_images(images[task_name], brain_mask)
        bold[task_name] = image.MaskedMultiSubjectData.from_masked_images(
            masked_images[task_name], len(fnames[task_name])
        )
        bold[task_name][np.isnan(bold[task_name])] = 0.0
        n_this = int(np.shape(bold[task_name])[-1])
        group_assignment.extend(
            list(np.repeat(group_assignment_dict[task_name], n_this))
        )
        print(f"  bold[{task_name!r}] shape: {np.shape(bold[task_name])}")

    # Voxelwise ISC, leave-one-subject-out (pairwise=False), tutorial §2.3
    isc_maps: dict[str, np.ndarray] = {}
    for task_name in ALL_TASK_NAMES:
        isc_maps[task_name] = isc(bold[task_name], pairwise=False)
        print(f"  isc_maps[{task_name!r}] shape: {np.shape(isc_maps[task_name])}")

    # Mean ISC across voxels (per subject) for a quick bar summary
    fig, ax = plt.subplots(figsize=(7, 4))
    means, labels = [], []
    for task_name in ALL_TASK_NAMES:
        m = isc_maps[task_name]
        # tutorial plots per-subject maps; m is (n_subj, n_voxels) after pairwise=False
        per_sub_mean = np.nanmean(m, axis=1)
        means.append(per_sub_mean)
        labels.append(task_name)
    ax.boxplot(means)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean voxel ISC (per subject)")
    ax.set_title("Pieman2 — tutorial 10 style ISC")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "isc_mean_voxel_boxplot.png"), dpi=150)
    plt.close(fig)

    # Optional: map one subject’s ISC to volume (tutorial cell 16 style)
    sub_plot = 0
    task_plot = "intact1"
    isc_vol = np.zeros(brain_nii.shape, dtype=np.float32)
    isc_vol[coords] = isc_maps[task_plot][sub_plot, :]
    mid = isc_vol.shape[2] // 2
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.rot90(isc_vol[:, :, mid]), cmap="RdYlBu_r", vmin=-0.2, vmax=0.8)
    ax.set_title(f"ISC slice (z≈{mid}) {task_plot} sub {sub_plot}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"isc_volume_slice_{task_plot}_sub{sub_plot}.png"), dpi=150)
    plt.close(fig)

    np.savez_compressed(
        os.path.join(out_dir, "isc_maps.npz"),
        **{f"isc_{k}": isc_maps[k] for k in ALL_TASK_NAMES},
    )

    if args.skip_permutation:
        print("\nSkipped permutation_isc (--skip-permutation).")
    else:
        isc_maps_all_tasks = np.vstack(
            [isc_maps[task_name] for task_name in ALL_TASK_NAMES]
        )
        print(
            f"\npermutation_isc (n={args.n_permutations}) on stacked maps "
            f"{isc_maps_all_tasks.shape} …"
        )
        observed, pvals, distribution = permutation_isc(
            isc_maps_all_tasks,
            pairwise=False,
            group_assignment=np.asarray(group_assignment, dtype=np.int32),
            summary_statistic="mean",
            n_permutations=args.n_permutations,
        )
        p_flat = np.asarray(pvals).ravel()
        obs_flat = np.asarray(observed).ravel()
        print(f"  observed shape: {obs_flat.shape}, p shape: {p_flat.shape}")
        np.savez_compressed(
            os.path.join(out_dir, "permutation_isc.npz"),
            observed=obs_flat,
            p=p_flat,
            distribution=np.asarray(distribution),
        )
        # Histogram of voxelwise p-values (exploratory)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(p_flat, bins=50, color="#4477aa", edgecolor="white")
        ax.set_xlabel("p (permutation ISC)")
        ax.set_ylabel("count (voxels)")
        ax.set_title("Permutation ISC p-value distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "permutation_p_hist.png"), dpi=150)
        plt.close(fig)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
