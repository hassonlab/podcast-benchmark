"""sentence_onset with pause-matched negatives (`sentence_onset_pausematched_task`).

Why: `sentence_onset_task` draws its negatives uniformly from the other word onsets. Sentence onsets follow a
silence (median silence before the word 0.62 s, against 0.04 s for the negatives; 78 % against 8 % above 0.2 s),
and the window is centred on the onset, so the label can be read from the stimulus alone, without any brain data:
a classifier on transcript timing reaches ROC-AUC 0.896 and one on the audio envelope 0.855. A decoder can then
score well by detecting the auditory onset response to "silence, then speech" rather than anything about
sentences.

What changes: the positives (word onsets that start a sentence) and the negative pool (all other word onsets) are
the same as in `sentence_onset_task`. Negatives are drawn so that, in every bin of silence-before
(`GAP_EDGES`), there are `neg_per_pos_matched` negatives per positive. The two classes then have the same
silence distribution, up to the bin width. A bin with too few negatives keeps only as many positives as it can
match. With 1:1 matching on the podcast this keeps 265 of 366 positives (530 examples), and the stimulus-only
ROC-AUC falls to 0.56 (timing) / 0.50 (audio envelope).

Caveats: the long-silence sentence onsets are the ones dropped, and the window still contains the word itself,
so word identity and prosody are not controlled. Keep `sentence_onset_task` for comparison with earlier
results.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import registry
from core.config import TaskConfig
from tasks.sentence_onset import SentenceOnsetConfig

# Bin edges (s) of silence-before used for the matching.
GAP_EDGES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)


@dataclass
class SentenceOnsetPauseMatchedConfig(SentenceOnsetConfig):
    """Configuration for sentence_onset_pausematched_task.

    `negatives_per_positive` and `negative_margin_s` (inherited) are ignored: the matching sets the ratio per bin.
    """

    neg_per_pos_matched: int = 1
    sampling_seed: int = 0


def silence_before(onsets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Time between each word's onset and the previous word's offset (the first word: its onset).

    Negative for words that overlap the previous one.
    """
    onsets = np.asarray(onsets, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if len(onsets) == 0:
        return np.empty(0)
    return np.r_[onsets[0], onsets[1:] - offsets[:-1]]


def pause_matched_sample(
    is_pos: np.ndarray,
    gap: np.ndarray,
    neg_per_pos: int = 1,
    edges=GAP_EDGES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (positives, negatives) with `neg_per_pos` negatives per positive inside every silence bin.

    In a bin with too few negatives the positives are subsampled. Both arrays are sorted. Deterministic for a
    given `seed`.
    """
    is_pos = np.asarray(is_pos, dtype=bool)
    if neg_per_pos < 1:
        raise ValueError("neg_per_pos must be >= 1")
    bins = np.digitize(np.asarray(gap, dtype=float), np.asarray(edges, dtype=float))
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for k in np.unique(bins):
        p = np.flatnonzero(is_pos & (bins == k))
        n = np.flatnonzero(~is_pos & (bins == k))
        m = min(len(p), len(n) // neg_per_pos)
        if m == 0:
            continue
        pos.append(rng.choice(p, m, replace=False))
        neg.append(rng.choice(n, m * neg_per_pos, replace=False))
    if not pos:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.sort(np.concatenate(pos)), np.sort(np.concatenate(neg))


def build_pausematched_table(
    sentence_csv_path: str, word_csv_path: str, neg_per_pos: int = 1, seed: int = 0
):
    """The task table (`start`, `target`, sorted by time) and a summary of the silence before each kept row."""
    df_word = pd.read_csv(word_csv_path)
    for column in ("onset", "offset"):
        if column not in df_word.columns:
            raise ValueError(f"Expected column '{column}' in word CSV")
    df_sentence = pd.read_csv(sentence_csv_path)
    if "sentence_onset" not in df_sentence.columns:
        raise ValueError("Expected column 'sentence_onset' in sentence CSV")

    onsets = df_word["onset"].to_numpy(dtype=float)
    offsets = df_word["offset"].to_numpy(dtype=float)
    sentence_onsets = df_sentence["sentence_onset"].to_numpy(dtype=float)
    if len(onsets) == 0 or len(sentence_onsets) == 0:
        is_pos = np.zeros(len(onsets), dtype=bool)
    else:
        is_pos = np.isclose(
            onsets[:, None], sentence_onsets[None, :], atol=1e-6, rtol=0.0
        ).any(axis=1)

    gap = silence_before(onsets, offsets)
    pos, neg = pause_matched_sample(is_pos, gap, neg_per_pos=neg_per_pos, seed=seed)
    df = pd.concat(
        [
            pd.DataFrame({"start": onsets[pos], "target": 1.0}),
            pd.DataFrame({"start": onsets[neg], "target": 0.0}),
        ],
        ignore_index=True,
    )
    df = df.sort_values("start", kind="stable").reset_index(drop=True)
    info = {"n_pos_pool": int(is_pos.sum()), "gap_pos": gap[pos], "gap_neg": gap[neg]}
    return df, info


@registry.register_task_data_getter(config_type=SentenceOnsetPauseMatchedConfig)
def sentence_onset_pausematched_task(task_config: TaskConfig):
    """Binary classification: sentence-onset word onsets (1.0) against pause-matched other word onsets (0.0).

    Returns a DataFrame with columns `start` (window centre, s) and `target`.
    """
    config: SentenceOnsetPauseMatchedConfig = task_config.task_specific_config
    if config.labels_path:
        raise NotImplementedError(
            "pause matching needs word offsets; use sentence_csv_path + word_csv_path"
        )
    df, info = build_pausematched_table(
        config.sentence_csv_path,
        config.word_csv_path,
        config.neg_per_pos_matched,
        config.sampling_seed,
    )

    print("\n=== SENTENCE ONSET (PAUSE-MATCHED) DATASET ===")
    print(f"Total examples: {len(df)}")
    print(f"Positives kept: {int((df.target == 1).sum())} / {info['n_pos_pool']}")
    print(f"Negatives: {int((df.target == 0).sum())}")
    if len(df):
        print(
            "Median silence before: "
            f"positives {np.median(info['gap_pos']):.3f} s, "
            f"negatives {np.median(info['gap_neg']):.3f} s"
        )
    print("=" * 50)
    return df
