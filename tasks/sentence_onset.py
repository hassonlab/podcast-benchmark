from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import BaseWordLabelTaskConfig, TaskConfig
from core import registry
from utils.label_utils import load_label_table


@dataclass
class SentenceOnsetConfig(BaseWordLabelTaskConfig):
    """Configuration for sentence_onset_task."""
    sentence_csv_path: str = "processed_data/all_sentences_podcast.csv"
    word_csv_path: str = "processed_data/df_word_onset_with_pos_class.csv"
    negatives_per_positive: int = 1
    # Legacy field accepted for older configs. Word-onset controlled sampling
    # does not use this value.
    negative_margin_s: float = 2.0


@registry.register_task_data_getter(config_type=SentenceOnsetConfig)
def sentence_onset_task(task_config: TaskConfig):
    """
    Binary classification dataset for sentence onset detection.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 for sentence-onset word onsets, 0.0 for other word onsets

    Notes:
      - Positive examples are word onsets matching sentence onsets.
      - Negative examples are sampled from word onsets that do not match any
        sentence onset.
      - `negative_margin_s` remains accepted for config compatibility but is
        intentionally unused.
    """
    # Get typed task-specific config
    config: SentenceOnsetConfig = task_config.task_specific_config

    canonical_table = None
    if config.labels_path:
        canonical_table = load_label_table(
            config.labels_path, required_columns=("sentence_onset",)
        )
        word_onsets = canonical_table["start"].to_numpy(dtype=float)
        sentence_values = canonical_table["sentence_onset"]
        if sentence_values.dtype == object:
            normalized = sentence_values.astype(str).str.strip().str.lower()
            unknown = ~normalized.isin({"true", "false", "1", "0"})
            if unknown.any():
                raise ValueError(
                    "sentence_onset values must be booleans or 0/1"
                )
            sentence_values = normalized.isin({"true", "1"})
        word_matches_sentence = sentence_values.astype(bool).to_numpy()
        sentence_matches_word = np.ones(int(word_matches_sentence.sum()), dtype=bool)
    else:
        df_sentence = pd.read_csv(config.sentence_csv_path)
        df_word = pd.read_csv(config.word_csv_path)
        if "sentence_onset" not in df_sentence.columns:
            raise ValueError("Expected column 'sentence_onset' in sentence CSV")
        if "onset" not in df_word.columns:
            raise ValueError("Expected column 'onset' in word CSV")
        sentence_onsets = df_sentence["sentence_onset"].to_numpy(dtype=float)
        word_onsets = df_word["onset"].to_numpy(dtype=float)
        if len(sentence_onsets) == 0 or len(word_onsets) == 0:
            word_matches_sentence = np.zeros(len(word_onsets), dtype=bool)
            sentence_matches_word = np.zeros(len(sentence_onsets), dtype=bool)
        else:
            matches = np.isclose(
                word_onsets[:, None], sentence_onsets[None, :], atol=1e-6, rtol=0.0
            )
            word_matches_sentence = matches.any(axis=1)
            sentence_matches_word = matches.any(axis=0)

    word_positions = np.arange(len(word_onsets), dtype=int)
    positive_positions = word_positions[word_matches_sentence]
    pos = pd.DataFrame(
        {
            "source_position": positive_positions,
            "start": word_onsets[positive_positions],
            "target": 1.0,
        }
    )

    negative_candidates = word_positions[~word_matches_sentence]
    negatives_per_positive = config.negatives_per_positive
    requested_negatives = len(pos) * negatives_per_positive
    sampling_capped = requested_negatives > len(negative_candidates)
    sampled_negatives = min(requested_negatives, len(negative_candidates))

    rng = np.random.default_rng(0)
    if requested_negatives == 0:
        negative_positions = np.array([], dtype=int)
    elif len(negative_candidates) == 0:
        raise ValueError(
            "Cannot sample sentence onset negatives: no non-sentence word onsets found"
        )
    else:
        negative_positions = rng.choice(
            negative_candidates,
            size=sampled_negatives,
            replace=False,
        )

    neg = pd.DataFrame(
        {
            "source_position": negative_positions,
            "start": word_onsets[negative_positions],
            "target": 0.0,
        }
    )

    unmatched_sentence_onsets = int((~sentence_matches_word).sum())

    df_out = (
        pd.concat([pos, neg], ignore_index=True)
        .sort_values("start")
        .reset_index(drop=True)
    )
    if canonical_table is not None and len(df_out):
        for column in ("event_id", "word", "end"):
            if column in canonical_table:
                df_out[column] = canonical_table.iloc[
                    df_out["source_position"].to_numpy(dtype=int)
                ][column].to_numpy()
    df_out = df_out.drop(columns="source_position")

    # Print dataset summary for inspection
    print(f"\n=== SENTENCE ONSET DATASET ===")
    print(f"Total examples: {len(df_out)}")
    print(f"Positives: {len(pos)}")
    print(f"Negatives: {len(neg)}")
    print(f"Requested negative ratio: {negatives_per_positive}")
    print(f"Unmatched sentence onsets dropped: {unmatched_sentence_onsets}")
    print(f"Negative sampling capped: {sampling_capped}")
    if len(df_out) > 0:
        print(f"Time range: {df_out['start'].min():.2f}s - {df_out['start'].max():.2f}s")
    print(f"First 10 examples:")
    print(df_out.head(10))
    print("=" * 50)

    return df_out
