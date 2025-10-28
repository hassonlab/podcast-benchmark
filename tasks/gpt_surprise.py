import os

import pandas as pd

from core.config import DataParams
from core import registry


@registry.register_task_data_getter()
def gpt_surprise_task(data_params: DataParams):
    """
    Dataset for GPT@XL surprise values as regression targets for each word.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: surprise value from GPT-2 XL

    Notes:
      - Uses `data_params.content_noncontent_path` if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.

    """
    # Pull task-specific params from data_params.task_params with local defaults
    tp = getattr(data_params, "task_params", {}) or {}

    # Resolve CSV path
    default_csv = os.path.join(
        os.getcwd(), "processed_data", "df_word_onset_with_pos_class.csv"
    )
    csv_path = tp.get("content_noncontent_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df = pd.DataFrame()
    df["start"] = df1["onset"]  # convert samples to seconds
    df["target"] = df1["surprise"]

    return df


@registry.register_task_data_getter()
def gpt_surprise_multiclass_task(data_params: DataParams):
    """
    Multiclass classification dataset for GPT2 XL surprise levels. The surprise levels are binned into 3 classes:
    Low (0): < mean-std, Medium (1) within std distance from mean, High (2) : > mean+std.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: class label: 0 (Low), 1 (Medium), 2 (High)

    Notes:
      - Uses `data_params.content_noncontent_path` if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.

    """
    # Pull task-specific params from data_params.task_params with local defaults
    tp = getattr(data_params, "task_params", {}) or {}

    # Resolve CSV path
    default_csv = os.path.join(
        os.getcwd(), "processed_data", "df_word_onset_with_pos_class.csv"
    )
    csv_path = tp.get("content_noncontent_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df = pd.DataFrame()
    df["start"] = df1["onset"]  # convert samples to seconds
    df["target"] = df1["surprise_class"]

    return df
