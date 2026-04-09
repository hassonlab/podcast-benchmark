"""
Algorithms transcribed from BrainIAK tutorial 11 (SRM).
https://brainiak.org/notebooks/tutorials/html/11-srm.html

Requires: pip install brainiak (see pyproject optional dependency ``fmri``).
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import stats
from sklearn.svm import NuSVC


def time_segment_matching(data: List[np.ndarray], win_size: int = 10) -> np.ndarray:
    """
    For each held-out subject, correlate sliding windows with the group sum of others.
    ``data[m]`` shape (n_features, n_TR) — voxels or SRM features.

    Returns per-subject accuracy (fraction of windows where argmax correlation hits true segment).
    """
    from brainiak.fcma.util import compute_correlation

    nsubjs = len(data)
    ndim, nsample = data[0].shape
    accu = np.zeros(shape=nsubjs)
    nseg = nsample - win_size

    trn_data = np.zeros((ndim * win_size, nseg), order="f")
    for m in range(nsubjs):
        for w in range(win_size):
            trn_data[w * ndim : (w + 1) * ndim, :] += data[m][:, w : (w + nseg)]

    for tst_subj in range(nsubjs):
        tst_data = np.zeros((ndim * win_size, nseg), order="f")
        for w in range(win_size):
            tst_data[w * ndim : (w + 1) * ndim, :] = data[tst_subj][:, w : (w + nseg)]

        a = np.nan_to_num(stats.zscore((trn_data - tst_data), axis=0, ddof=1))
        b = np.nan_to_num(stats.zscore(tst_data, axis=0, ddof=1))
        corr_mtx = compute_correlation(b.T, a.T)

        ii = np.arange(nseg)[:, None]
        jj = np.arange(nseg)[None, :]
        neigh_mask = (np.abs(ii - jj) < win_size) & (ii != jj)
        corr_mtx[neigh_mask] = -np.inf
        max_idx = np.argmax(corr_mtx, axis=1)
        accu[tst_subj] = float(np.sum(max_idx == np.arange(nseg))) / nseg

    return accu


def _pearson_corr_matrix_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation between each row of ``X`` and each row of ``Y`` (same width)."""
    xm = X - X.mean(axis=1, keepdims=True)
    ym = Y - Y.mean(axis=1, keepdims=True)
    xn = np.sqrt(np.sum(xm * xm, axis=1, keepdims=True)).clip(1e-12, None)
    yn = np.sqrt(np.sum(ym * ym, axis=1, keepdims=True)).clip(1e-12, None)
    return (xm @ ym.T) / (xn @ yn.T)


def time_segment_matching_numpy(data: List[np.ndarray], win_size: int = 10) -> np.ndarray:
    """
    Same logic as ``time_segment_matching`` but uses NumPy correlations only (no BrainIAK).
    """
    nsubjs = len(data)
    ndim, nsample = data[0].shape
    accu = np.zeros(shape=nsubjs)
    nseg = nsample - win_size

    trn_data = np.zeros((ndim * win_size, nseg), order="f")
    for m in range(nsubjs):
        for w in range(win_size):
            trn_data[w * ndim : (w + 1) * ndim, :] += data[m][:, w : (w + nseg)]

    for tst_subj in range(nsubjs):
        tst_data = np.zeros((ndim * win_size, nseg), order="f")
        for w in range(win_size):
            tst_data[w * ndim : (w + 1) * ndim, :] = data[tst_subj][:, w : (w + nseg)]

        a = np.nan_to_num(stats.zscore((trn_data - tst_data), axis=0, ddof=1))
        b = np.nan_to_num(stats.zscore(tst_data, axis=0, ddof=1))
        corr_mtx = _pearson_corr_matrix_rows(b.T, a.T)

        ii = np.arange(nseg)[:, None]
        jj = np.arange(nseg)[None, :]
        neigh_mask = (np.abs(ii - jj) < win_size) & (ii != jj)
        corr_mtx[neigh_mask] = -np.inf
        max_idx = np.argmax(corr_mtx, axis=1)
        accu[tst_subj] = float(np.sum(max_idx == np.arange(nseg))) / nseg

    return accu


def image_class_prediction(image_data_shared: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """
    Leave-one-subject-out Nu-SVM on (flattened) shared-space patterns, tutorial §7.
    ``image_data_shared[s]`` shape (n_features, n_TR).
    """
    subjects = len(image_data_shared)
    train_labels = np.tile(labels, subjects - 1)
    test_labels = labels
    accuracy = np.zeros((subjects,))
    for subject in range(subjects):
        train_subjects = list(range(subjects))
        train_subjects.remove(subject)
        trs = image_data_shared[0].shape[1]
        train_mat = np.zeros((image_data_shared[0].shape[0], len(train_labels)))
        for train_subject in range(len(train_subjects)):
            ts = train_subjects[train_subject]
            start_index = train_subject * trs
            end_index = start_index + trs
            train_mat[:, start_index:end_index] = image_data_shared[ts]

        classifier = NuSVC(nu=0.5, kernel="linear")
        classifier.fit(train_mat.T, train_labels)
        predicted = classifier.predict(image_data_shared[subject].T)
        accuracy[subject] = float(np.sum(predicted == test_labels)) / len(predicted)
    return accuracy
