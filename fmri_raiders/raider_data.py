"""
Load preprocessed Raider VT data (numpy) as in BrainIAK tutorial 11.

Expected files in ``raider_dir`` (e.g. from BrainIAK condensed datasets on Zenodo):
  - movie.npy   shape (n_voxels, n_TR_movie, n_subjects)
  - image.npy   shape (n_voxels, n_TR_image, n_subjects)
  - label.npy   length n_TR_image (same category order for all subjects)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from scipy import stats


def load_movie(raider_dir: str) -> Tuple[np.ndarray, int, int, int]:
    path = os.path.join(raider_dir, "movie.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path} — download BrainIAK Raider data (see fmri_raiders/run_tutorial.py docstring).")
    movie_data = np.load(path)
    vox_num, n_tr, num_subs = movie_data.shape
    return movie_data, vox_num, n_tr, num_subs


def load_image_and_labels(raider_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    ip = os.path.join(raider_dir, "image.npy")
    lp = os.path.join(raider_dir, "label.npy")
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
    """In-place z-score each subject's array along voxel axis (tutorial convention)."""
    for sub in range(len(data_list)):
        data_list[sub] = stats.zscore(data_list[sub], axis=axis, ddof=ddof)
        data_list[sub] = np.nan_to_num(data_list[sub])


def movie_and_image_lists_for_classification(
    movie_data: np.ndarray, image_data: np.ndarray, num_subs: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Full movie and image runs per subject (voxels × TR), tutorial §7."""
    train_list = [movie_data[:, :, sub].copy() for sub in range(num_subs)]
    test_list = [image_data[:, :, sub].copy() for sub in range(num_subs)]
    return train_list, test_list
