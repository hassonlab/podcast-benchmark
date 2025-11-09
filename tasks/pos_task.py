import os

import numpy as np
import pandas as pd

from core.config import DataParams
from core import registry


@registry.register_task_data_getter()
def pos_task(data_params: DataParams):
    """
    Dataset for Parts of Speech Classification. It contains five classes:
    Noun (0), Verb (1), Adjective (2), Adverb (3) and others (4)

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: Parts of Speech class label

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
    csv_path = tp.get("pos_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df1=df1[df1['pos_class'].isin([0, 1,4])]
    
    # Update pos_class values: change 2 to 1
    df1.loc[df1['pos_class'] == 4, 'pos_class'] = 2

    df = pd.DataFrame()
    df["start"] = df1["onset"]  # convert samples to seconds
    df["target"] = df1["pos_class"]

    print(f"\n=== Parts of Speech DATASET ===")
    print(f"Total examples: {len(df)}")
    print(f"Noun: {np.sum(df.target==0)}")
    print(f"Verb: {np.sum(df.target==1)}")
    print(f"Adjective: {np.sum(df.target==2)}")
    print(f"Adverb: {np.sum(df.target==3)}")
    print(f"Other: {np.sum(df.target==4)}")

    return df
