import os

import numpy as np
import pandas as pd

from core.config import DataParams
from core import registry


@registry.register_task_data_getter()
def sentence_onset_task(data_params: DataParams):
    """
    Binary classification dataset for sentence onset detection.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 for sentence onset, 0.0 for negatives sampled away from onsets

    Notes:
      - Uses `data_params.sentence_csv_path` if provided; else defaults to
        `<data_root>/all_sentences_podcast.csv`.
      - Negative examples are sampled within each sentence, at least
        `negative_margin_s` seconds after onset and at least one window width
        before the sentence end.
    """
    # Pull task-specific params from data_params.task_params with local defaults
    tp = getattr(data_params, "task_params", {}) or {}

    # Resolve CSV path
    default_csv = os.path.join(
        os.getcwd(), "processed_data", "all_sentences_podcast.csv"
    )
    csv_path = tp.get("sentence_csv_path", default_csv)

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
    window = getattr(data_params, "window_width", 0.625) or 0.625
    negatives_per_positive = int(tp.get("negatives_per_positive", 1))
    negative_margin_s = float(tp.get("negative_margin_s", 2.0))

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
