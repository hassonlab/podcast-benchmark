import os
from dataclasses import dataclass

import pandas as pd

from core.config import BaseTaskConfig, TaskConfig
from core import registry


@dataclass
class GptSurpriseConfig(BaseTaskConfig):
    """Configuration for gpt_surprise_task and gpt_surprise_multiclass_task."""
    content_noncontent_path: str = "processed_data/df_word_onset_with_pos_class.csv"


@registry.register_task_data_getter(config_type=GptSurpriseConfig)
def gpt_surprise_task(task_config: TaskConfig):
    """
    Dataset for GPT@XL surprise values as regression targets for each word.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: surprise value from GPT-2 XL

    Notes:
      - Uses config.content_noncontent_path if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.

    """
    config: GptSurpriseConfig = task_config.task_specific_config
    csv_path = config.content_noncontent_path

    df1 = pd.read_csv(csv_path, index_col=0)

    df = pd.DataFrame()
    df["start"] = df1["onset"]  # convert samples to seconds
    df["target"] = df1["surprise"]

    return df


@registry.register_task_data_getter(config_type=GptSurpriseConfig)
def gpt_surprise_multiclass_task(task_config: TaskConfig):
    """
    Multiclass classification dataset for GPT2 XL surprise levels. The surprise levels are binned into 3 classes:
    Low (0): < mean-std, Medium (1) within std distance from mean, High (2) : > mean+std.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: class label: 0 (Low), 1 (Medium), 2 (High)

    Notes:
      - Uses config.content_noncontent_path if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.

    """
    config: GptSurpriseConfig = task_config.task_specific_config
    csv_path = config.content_noncontent_path

    df1 = pd.read_csv(csv_path, index_col=0)

    df = pd.DataFrame()
    df["start"] = df1["onset"]  # convert samples to seconds
    df["target"] = df1["surprise_class"]

    return df
