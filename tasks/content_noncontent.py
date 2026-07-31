import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import BaseWordLabelTaskConfig, TaskConfig
from core import registry
from utils.label_utils import task_frame_from_labels


@dataclass
class ContentNonContentConfig(BaseWordLabelTaskConfig):
    """Configuration for content_noncontent_task."""
    content_noncontent_path: str = "processed_data/df_word_onset_with_pos_class.csv"


@registry.register_task_data_getter(config_type=ContentNonContentConfig)
def content_noncontent_task(task_config: TaskConfig):
    """
    Binary classification dataset for content vs non-content word classification.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 for content, 0.0 non-content

    Notes:
      - Uses config.content_noncontent_path if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.

    """
    config: ContentNonContentConfig = task_config.task_specific_config
    csv_path = config.labels_path or config.content_noncontent_path
    df = task_frame_from_labels(csv_path, "is_content")

    print(f"\n=== Content Non-content words DATASET ===")
    print(f"Total examples: {len(df)}")
    print(f"Positives: {np.sum(df.target==1)}")
    print(f"Negatives: {len(df) - np.sum(df.target==1)}")

    return df
