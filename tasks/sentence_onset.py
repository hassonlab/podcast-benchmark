import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import BaseTaskConfig, TaskConfig
from core import registry


@dataclass
class SentenceOnsetConfig(BaseTaskConfig):
    """Configuration for sentence_onset_task."""
    sentence_csv_path: str = "processed_data/all_sentences_podcast.csv"
    negatives_per_positive: int = 1
    negative_margin_s: float = 2.0


@registry.register_task_data_getter(config_type=SentenceOnsetConfig)
def sentence_onset_task(task_config: TaskConfig):
    """
    Binary classification dataset for sentence onset detection.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 for sentence onset, 0.0 for negatives sampled away from onsets

    Notes:
      - Uses config.sentence_csv_path if provided; else defaults to
        `<data_root>/all_sentences_podcast.csv`.
      - Negative examples are sampled within each sentence, at least
        `negative_margin_s` seconds after onset and at least one window width
        before the sentence end.
    """
    # Get typed task-specific config
    config: SentenceOnsetConfig = task_config.task_specific_config
    data_params = task_config.data_params
    csv_path = config.sentence_csv_path

    df = pd.read_csv(csv_path, index_col=0)

    # Expect columns: sentence_onset, sentence_offset
    if not {"sentence_onset", "sentence_offset"}.issubset(df.columns):
        raise ValueError(
            "Expected columns 'sentence_onset' and 'sentence_offset' in sentence CSV"
        )

    onsets = df["sentence_onset"].to_numpy()
    offsets = df["sentence_offset"].to_numpy()

    # Positives
    pos = pd.DataFrame({"start": onsets, "target": 1.0})

    # Negatives: sample away from onsets within the same sentence when possible
    window = data_params.window_width if data_params.window_width > 0 else 0.625
    negatives_per_positive = config.negatives_per_positive
    negative_margin_s = config.negative_margin_s

    rng = np.random.default_rng(0)
    neg_starts = []
    for onset, offset in zip(onsets, offsets):
        # Start sampling after a margin to avoid including the onset in the window
        left = onset + negative_margin_s
        # Ensure we can still place a full window without touching sentence end
        right = max(left, offset - window - 1e-3)
        if right > left:
            samples = rng.uniform(left, right, size=max(0, negatives_per_positive))
            neg_starts.extend(samples.tolist())

    neg = pd.DataFrame({"start": neg_starts, "target": 0.0})

    df_out = (
        pd.concat([pos, neg], ignore_index=True)
        .sort_values("start")
        .reset_index(drop=True)
    )

    # Print dataset summary for inspection
    print(f"\n=== SENTENCE ONSET DATASET ===")
    print(f"Total examples: {len(df_out)}")
    print(f"Positives: {len(pos)} ({len(pos)/len(df_out)*100:.1f}%)")
    print(f"Negatives: {len(neg)} ({len(neg)/len(df_out)*100:.1f}%)")
    print(f"Time range: {df_out['start'].min():.2f}s - {df_out['start'].max():.2f}s")
    print(f"First 10 examples:")
    print(df_out.head(10))
    print("=" * 50)

    return df_out
