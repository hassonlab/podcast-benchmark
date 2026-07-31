import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import BaseWordLabelTaskConfig, TaskConfig
from core import registry
from utils.label_utils import task_frame_from_labels


@dataclass
class PosTaskConfig(BaseWordLabelTaskConfig):
    """Configuration for pos_task."""
    pos_path: str = "processed_data/df_word_onset_with_pos_class.csv"


@registry.register_task_data_getter(config_type=PosTaskConfig)
def pos_task(task_config: TaskConfig):
    """
    Dataset for Parts of Speech Classification. It contains five classes:
    Noun (0), Verb (1), Adjective (2), Adverb (3) and others (4)

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: Parts of Speech class label

    Notes:
      - Uses config.pos_path if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.

    """
    config: PosTaskConfig = task_config.task_specific_config
    csv_path = config.labels_path or config.pos_path
    df = task_frame_from_labels(csv_path, "pos_class")

    print(f"\n=== Parts of Speech DATASET ===")
    print(f"Total examples: {len(df)}")
    print(f"Noun: {np.sum(df.target==0)}")
    print(f"Verb: {np.sum(df.target==1)}")
    print(f"Adjective: {np.sum(df.target==2)}")
    print(f"Adverb: {np.sum(df.target==3)}")
    print(f"Other: {np.sum(df.target==4)}")

    return df
