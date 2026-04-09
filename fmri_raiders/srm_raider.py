"""
BrainIAK SRM on Raider movie data (same conventions as ``run_tutorial.py``).

Used by ``eval_temporal_vae.py`` for VAE vs SRM comparison. Optional dependency:
``pip install -e ".[fmri]"`` (needs MPI for some installs).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats

from fmri_raiders import raider_data


def import_srm_module():
    """Return ``brainiak.funcalign.srm`` or raise ImportError / RuntimeError with context."""
    try:
        import brainiak.funcalign.srm as srm_module  # noqa: F401

        return srm_module
    except ImportError as e:
        raise ImportError(
            "SRM comparison needs BrainIAK: pip install -e \".[fmri]\" (or pip install brainiak)"
        ) from e
    except RuntimeError as e:
        err = str(e).lower()
        if "mpi" in err or "libmpi" in err:
            raise RuntimeError(
                "brainiak/mpi4py needs the system MPI runtime. On Ubuntu/Debian try:\n"
                "  sudo apt install libopenmpi3 openmpi-bin\n"
                f"(Original error: {e})"
            ) from e
        raise


def _zscore_shared_list(shared: List[np.ndarray]) -> None:
    for i in range(len(shared)):
        shared[i] = np.nan_to_num(stats.zscore(shared[i], axis=1, ddof=1))


def fit_srm_movie_halves(
    train_mode: str,
    movie_data: np.ndarray,
    num_subs: int,
    vox_num: int,
    raw_train_z: List[np.ndarray],
    raw_test_z: List[np.ndarray],
    features: int,
    n_iter: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Fit SRM aligned with checkpoint ``movie_split_for_training``.

    - ``first_half``: fit on z-scored first half (``raw_train_z``), transform train + test halves.
    - ``full``: z-score full movie per subject, fit on all TRs, split shared trajectories for TSM.

    ``raw_*_z`` must already be z-scored voxelwise like the tutorial (same arrays as raw TSM).
    """
    srm_module = import_srm_module()
    k = int(min(features, vox_num))
    srm = srm_module.SRM(n_iter=int(n_iter), features=k)

    if train_mode == "first_half":
        train_fit = [x.astype(np.float32, copy=False) for x in raw_train_z]
        srm.fit(train_fit)
        sh_tr = srm.transform([x.astype(np.float32, copy=False) for x in raw_train_z])
        sh_te = srm.transform([x.astype(np.float32, copy=False) for x in raw_test_z])
    elif train_mode == "full":
        full_list = [movie_data[:, :, s].astype(np.float32).copy() for s in range(num_subs)]
        raider_data.zscore_voxelwise(full_list)
        srm.fit(full_list)
        shared_full = srm.transform(full_list)
        _, n_tr, _ = movie_data.shape
        half = n_tr // 2
        sh_tr = [shared_full[s][:, :half].copy() for s in range(num_subs)]
        sh_te = [shared_full[s][:, -half:].copy() for s in range(num_subs)]
    else:
        raise ValueError(f"train_mode must be 'first_half' or 'full', got {train_mode!r}")

    _zscore_shared_list(sh_tr)
    _zscore_shared_list(sh_te)
    return sh_tr, sh_te


def fit_srm_for_images(
    train_mode: str,
    movie_data: np.ndarray,
    image_per_subj: List[np.ndarray],
    n_subjects: int,
    vox_num: int,
    features: int,
    n_iter: int,
) -> List[np.ndarray]:
    """
    Fit SRM on movie (first half or full), transform z-scored image runs per subject.
    ``image_per_subj[s]`` shape (n_voxels, n_TR_image), raw — will be z-scored in-place copy.
    """
    srm_module = import_srm_module()
    k = int(min(features, vox_num))
    srm = srm_module.SRM(n_iter=int(n_iter), features=k)
    _, n_tr, _ = movie_data.shape
    half = n_tr // 2

    if train_mode == "first_half":
        m_fit = [movie_data[:, :half, s].astype(np.float32).copy() for s in range(n_subjects)]
    elif train_mode == "full":
        m_fit = [movie_data[:, :, s].astype(np.float32).copy() for s in range(n_subjects)]
    else:
        raise ValueError(f"train_mode must be 'first_half' or 'full', got {train_mode!r}")

    raider_data.zscore_voxelwise(m_fit)
    srm.fit(m_fit)

    img_z = [image_per_subj[s].astype(np.float32).copy() for s in range(n_subjects)]
    raider_data.zscore_voxelwise(img_z)
    shared_img = srm.transform(img_z)
    _zscore_shared_list(shared_img)
    return shared_img
