import numpy as np
import pandas as pd

from core import registry
from core.config import DataParams, TaskConfig
from tasks.sentence_onset_pausematched import (
    GAP_EDGES,
    SentenceOnsetPauseMatchedConfig,
    pause_matched_sample,
    sentence_onset_pausematched_task,
    silence_before,
)


def _write_csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_silence_before():
    onsets, offsets = np.array([1.0, 2.0, 2.5]), np.array([1.5, 2.6, 3.0])
    np.testing.assert_allclose(silence_before(onsets, offsets), [1.0, 0.5, -0.1])


def test_pause_matched_sample_equal_counts_per_bin():
    rng = np.random.default_rng(1)
    gap = np.r_[
        rng.uniform(0.3, 1.0, 50), rng.uniform(0.0, 0.1, 400), rng.uniform(0.3, 1.0, 20)
    ]
    is_pos = np.r_[np.ones(50, bool), np.zeros(420, bool)]
    pos, neg = pause_matched_sample(is_pos, gap)
    assert is_pos[pos].all() and not is_pos[neg].any()
    bins_pos, bins_neg = np.digitize(gap[pos], GAP_EDGES), np.digitize(gap[neg], GAP_EDGES)
    for k in np.unique(np.r_[bins_pos, bins_neg]):
        assert (bins_pos == k).sum() == (bins_neg == k).sum()
    # Only 20 long-silence negatives exist, so the long-silence positives are subsampled.
    assert len(pos) <= 20


def test_pause_matched_sample_ratio_and_determinism():
    gap = np.full(110, 0.5)
    is_pos = np.r_[np.ones(10, bool), np.zeros(100, bool)]
    pos, neg = pause_matched_sample(is_pos, gap, neg_per_pos=3, seed=4)
    assert (len(pos), len(neg)) == (10, 30)
    _, neg2 = pause_matched_sample(is_pos, gap, neg_per_pos=3, seed=4)
    np.testing.assert_array_equal(neg, neg2)


def test_task_is_registered_with_its_config():
    assert "sentence_onset_pausematched_task" in registry.task_registry
    assert (
        registry.task_registry["sentence_onset_pausematched_task"]["config_type"]
        is SentenceOnsetPauseMatchedConfig
    )


def test_task_matches_silence_distribution(tmp_path):
    # Words 0, 4, 8 start sentences after a 1 s silence; word 6 is a non-sentence word after a 1 s silence;
    # the rest follow each other with no silence.
    onsets, offsets, t = [], [], 0.0
    for i in range(10):
        t += 1.0 if i in (0, 4, 6, 8) else 0.0
        onsets.append(t)
        t += 0.3
        offsets.append(t)
    sentence_path = _write_csv(
        tmp_path, "sentences.csv", [{"sentence_onset": onsets[i]} for i in (0, 4, 8)]
    )
    word_path = _write_csv(
        tmp_path, "words.csv", [{"onset": a, "offset": b} for a, b in zip(onsets, offsets)]
    )
    df = sentence_onset_pausematched_task(
        TaskConfig(
            task_name="sentence_onset_pausematched_task",
            data_params=DataParams(),
            task_specific_config=SentenceOnsetPauseMatchedConfig(
                sentence_csv_path=sentence_path, word_csv_path=word_path
            ),
        )
    )
    # One long-silence negative exists, so exactly one positive is kept and word 6 is its negative.
    assert list(df.columns) == ["start", "target"]
    assert (df.target == 1.0).sum() == 1 and (df.target == 0.0).sum() == 1
    assert np.isclose(df.loc[df.target == 0.0, "start"].item(), onsets[6])
    assert df["start"].is_monotonic_increasing
