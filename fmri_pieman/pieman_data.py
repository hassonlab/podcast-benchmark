"""
Load preprocessed Pieman2 data (numpy) in the **same file layout** as Raider (BrainIAK tutorial 11).

Expected files in ``pieman_dir`` (default ``data/pieman`` or env ``PIEMAN_DATA_DIR``):

  - movie.npy   shape (n_voxels, n_TR_movie, n_subjects)
  - image.npy   optional — same as Raider if you have a localizer / image run
  - label.npy   optional — length n_TR_image

The BrainIAK Zenodo ``Pieman2.zip`` is large (~2.7 GB) and ships **NIfTI** (BrainIAK tutorial 10), not
``movie.npy``. Use ``fmri_pieman/download_pieman_data.py`` to fetch/extract. For **BrainIAK tutorial 10 (ISC)** on the
NIfTI tree, run ``fmri_pieman/run_tutorial.py``. For the **temporal VAE** path, build ``movie.npy``
with ``fmri_pieman/build_movie_npy.py`` (or ``--build-movie-npy`` on download).

Reference: Simony et al., 2016 — https://doi.org/10.1038/ncomms12141
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from scipy import stats


def load_movie(pieman_dir: str) -> Tuple[np.ndarray, int, int, int]:
    path = os.path.join(pieman_dir, "movie.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path} — see fmri_pieman/download_pieman_data.py and fmri_pieman/pieman_data.py docstring."
        )
    movie_data = np.load(path)
    vox_num, n_tr, num_subs = movie_data.shape
    return movie_data, vox_num, n_tr, num_subs


def load_image_and_labels(pieman_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    ip = os.path.join(pieman_dir, "image.npy")
    lp = os.path.join(pieman_dir, "label.npy")
    if not os.path.isfile(ip) or not os.path.isfile(lp):
        raise FileNotFoundError(f"Missing {ip} or {lp}")
    image_data = np.load(ip)
    labels = np.load(lp)
    labels = np.asarray(labels).ravel()
    return image_data, labels


def split_half_movie(
    movie_data: np.ndarray, num_subs: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """First / second half of TRs per subject; each list element (n_voxels, n_TR_half)."""
    _, n_tr, _ = movie_data.shape
    half = n_tr // 2
    train_data: List[np.ndarray] = []
    test_data: List[np.ndarray] = []
    for sub in range(num_subs):
        train_data.append(movie_data[:, :half, sub].copy())
        test_data.append(movie_data[:, -half:, sub].copy())
    return train_data, test_data


def zscore_voxelwise(data_list: List[np.ndarray], axis: int = 1, ddof: int = 1) -> None:
    """In-place z-score each subject's array along time (axis 1 for voxels × TR)."""
    for sub in range(len(data_list)):
        data_list[sub] = stats.zscore(data_list[sub], axis=axis, ddof=ddof)
        data_list[sub] = np.nan_to_num(data_list[sub])


def movie_and_image_lists_for_classification(
    movie_data: np.ndarray, image_data: np.ndarray, num_subs: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Full movie and image runs per subject (voxels × TR)."""
    train_list = [movie_data[:, :, sub].copy() for sub in range(num_subs)]
    test_list = [image_data[:, :, sub].copy() for sub in range(num_subs)]
    return train_list, test_list
