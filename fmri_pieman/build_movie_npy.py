#!/usr/bin/env python3
"""
Build ``movie.npy`` (voxels × TR × subjects) from BrainIAK Pieman2 **NIfTI** layout.

Zenodo ``Pieman2.zip`` does **not** ship ``movie.npy``; it ships BIDS-style runs and a gray-matter
mask (see BrainIAK tutorial 10). This script mirrors that notebook’s ``intact1`` loading: all
``sub-*/func/*-task-intact1.nii.gz`` volumes, masked with ``masks/avg152T1_gray_3mm.nii.gz``.

Requires: ``pip install nibabel`` (included with ``pip install -e ".[fmri]"`` / BrainIAK).

Example::

  python fmri_pieman/build_movie_npy.py --pieman2-dir data/pieman/Pieman2
  python fmri_pieman/build_movie_npy.py --pieman2-dir data/pieman/Pieman2 --out data/pieman/movie.npy
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def find_pieman2_root(search_dir: str) -> str | None:
    """Return dataset root (parent of ``masks/``) if BrainIAK Pieman2 layout is present."""
    for dirpath, _, files in os.walk(search_dir):
        if os.path.basename(dirpath) == "masks" and "avg152T1_gray_3mm.nii.gz" in files:
            return os.path.dirname(dirpath)
    return None


def list_intact1_runs(pieman2_root: str) -> list[str]:
    pattern = os.path.join(pieman2_root, "sub-*", "func", "sub-*-task-intact1.nii.gz")
    return sorted(glob.glob(pattern))


def build_movie_npy(pieman2_root: str, out_path: str) -> tuple[int, int, int]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "nibabel is required. Install with:  pip install nibabel\n"
            "Or use the fmri extra:  pip install -e \".[fmri]\""
        ) from e

    mask_fp = os.path.join(pieman2_root, "masks", "avg152T1_gray_3mm.nii.gz")
    if not os.path.isfile(mask_fp):
        raise FileNotFoundError(f"Expected mask not found: {mask_fp}")

    runs = list_intact1_runs(pieman2_root)
    if not runs:
        raise FileNotFoundError(
            f"No intact1 runs under {pieman2_root!r} "
            f"(glob sub-*/func/sub-*-task-intact1.nii.gz)"
        )

    mask_img = nib.load(mask_fp)
    mask = mask_img.get_fdata(dtype=np.float32) > 0.5

    stacks: list[np.ndarray] = []
    n_tr = None
    for i, fp in enumerate(runs):
        img = nib.load(fp)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"{fp}: expected 4D BOLD, got shape {data.shape}")
        if data.shape[:3] != mask.shape:
            raise ValueError(
                f"{fp}: spatial shape {data.shape[:3]} != mask {mask.shape}. "
                "Resample to mask grid (e.g. nilearn) or use unmodified Pieman2.zip."
            )
        if n_tr is None:
            n_tr = data.shape[3]
        elif data.shape[3] != n_tr:
            raise ValueError(f"{fp}: TR count {data.shape[3]} != {n_tr}")

        # (nx, ny, nz, t) boolean mask (3D) -> (n_vox, n_tr)
        masked = data[mask]
        stacks.append(masked)
        print(f"  [{i + 1}/{len(runs)}] {os.path.basename(fp)}  →  {masked.shape}", flush=True)

    movie = np.stack(stacks, axis=2).astype(np.float32, copy=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.save(out_path, movie)
    vox, tr, subs = movie.shape
    print(f"Wrote {out_path}  shape (voxels × TR × subjects) = {vox} × {tr} × {subs}")
    return vox, tr, subs


def main() -> None:
    p = argparse.ArgumentParser(description="Pieman2 NIfTI → movie.npy (intact story)")
    p.add_argument(
        "--pieman2-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "pieman", "Pieman2"),
        help="Extracted Pieman2 directory (contains masks/ and sub-*/func/)",
    )
    p.add_argument(
        "--out",
        default=os.path.join(_PROJECT_ROOT, "data", "pieman", "movie.npy"),
        help="Output path for movie.npy",
    )
    args = p.parse_args()
    root = os.path.abspath(args.pieman2_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")
    out = args.out if os.path.isabs(args.out) else os.path.join(_PROJECT_ROOT, args.out)
    try:
        build_movie_npy(root, out)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
