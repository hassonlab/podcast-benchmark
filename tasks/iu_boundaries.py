from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import BaseTaskConfig, TaskConfig
from core import registry


@dataclass
class IUBoundaryConfig(BaseTaskConfig):
    """Configuration for iu_boundary_task."""
    iu_boundaries_csv_path: str = "processed_data/iu_boundaries.csv"
    negatives_per_positive: int = 1
    negative_margin_s: float = 1.0


@registry.register_task_data_getter(config_type=IUBoundaryConfig)
def iu_boundary_task(task_config: TaskConfig):
    """
    Binary classification dataset for intonation unit (IU) boundary detection.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 at an IU boundary, 0.0 for negatives sampled from IU interiors

    Notes:
      - Positive examples are the IU boundary times produced by the PSST model
        and aligned to the reference transcript (processed_data/iu_boundaries.csv).
      - Negative examples are sampled from within each IU segment (the interval
        between two consecutive boundaries), at least `negative_margin_s` seconds
        away from both the preceding and following boundary.  Segments that are
        too short to place a full window are skipped.
    """
    config: IUBoundaryConfig = task_config.task_specific_config
    data_params = task_config.data_params

    df_iu = pd.read_csv(config.iu_boundaries_csv_path)

    if "time" not in df_iu.columns:
        raise ValueError("Expected a 'time' column in the IU boundaries CSV")

    boundary_times = np.sort(df_iu["time"].to_numpy())

    # Positives
    pos = pd.DataFrame({"start": boundary_times, "target": 1.0})

    # Negatives: one (or more) random times per IU segment, safely interior
    window = data_params.window_width if data_params.window_width > 0 else 0.625
    margin = config.negative_margin_s
    n_neg = config.negatives_per_positive

    rng = np.random.default_rng(0)
    neg_starts = []

    for i in range(len(boundary_times) - 1):
        seg_start = boundary_times[i]
        seg_end = boundary_times[i + 1]

        # Stay at least `margin` past the left boundary and at least
        # `margin + window` before the right boundary so the neural window
        # does not overlap with either edge.
        left = seg_start + margin
        right = seg_end - margin - window - 1e-3

        if right > left:
            samples = rng.uniform(left, right, size=n_neg)
            neg_starts.extend(samples.tolist())

    neg = pd.DataFrame({"start": neg_starts, "target": 0.0})

    n_skipped = (len(boundary_times) - 1) - len(neg_starts) // max(n_neg, 1)

    df_out = (
        pd.concat([pos, neg], ignore_index=True)
        .sort_values("start")
        .reset_index(drop=True)
    )

    print(f"\n=== IU BOUNDARY DATASET ===")
    print(f"Total examples:           {len(df_out)}")
    print(f"Positives:                {len(pos)} ({len(pos)/len(df_out)*100:.1f}%)")
    print(f"Negatives:                {len(neg)} ({len(neg)/len(df_out)*100:.1f}%)")
    print(f"Skipped segments (too short): {n_skipped}")
    print(f"Time range:               {df_out['start'].min():.2f}s – {df_out['start'].max():.2f}s")
    print(f"First 10 examples:")
    print(df_out.head(10))
    print("=" * 50)

    return df_out
