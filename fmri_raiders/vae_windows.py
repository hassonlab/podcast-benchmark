"""Sliding TR windows for Raider fMRI → MultiPatientTemporalVAE (voxels × time bins)."""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def window_average_np(data: np.ndarray, num_average_samples: int) -> np.ndarray:
    """Pool last axis in groups of ``num_average_samples`` (mean). Same as podcast VAE script."""
    n = data.shape[2]
    n_keep = (n // num_average_samples) * num_average_samples
    trimmed = data[:, :, :n_keep]
    reshaped = trimmed.reshape(trimmed.shape[0], trimmed.shape[1], -1, num_average_samples)
    return reshaped.mean(axis=-1)


def extract_batch_windows_fmri(
    arrays: List[np.ndarray],
    center_trs: List[int],
    half_tr: int,
    num_average_samples: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    ``arrays[s]``: (n_voxels, n_TR). For each center TR ``c``, take ``[c-half_tr, c+half_tr)``.
    Returns list of tensors (batch, n_voxels, input_timesteps) with
    ``input_timesteps = 2 * half_tr // num_average_samples``.
    """
    xs: List[torch.Tensor] = []
    for arr in arrays:
        windows = np.stack(
            [arr[:, c - half_tr : c + half_tr] for c in center_trs],
            axis=0,
        )
        windows_binned = window_average_np(windows, num_average_samples)
        xs.append(torch.tensor(windows_binned, dtype=torch.float32, device=device))
    return xs


def assert_window_shapes(half_tr: int, num_average_samples: int, input_timesteps: int) -> None:
    total = 2 * half_tr
    if total % num_average_samples != 0:
        raise ValueError(f"2*half_tr={total} must be divisible by num_average_samples={num_average_samples}")
    if total // num_average_samples != input_timesteps:
        raise ValueError(
            f"Expected input_timesteps {input_timesteps}, got {total // num_average_samples} "
            f"from half_tr={half_tr}, num_average_samples={num_average_samples}"
        )
