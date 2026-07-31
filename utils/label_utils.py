"""Canonical event/word label table loading shared by decoding tasks."""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "word_idx": "event_id",
    "onset": "start",
    "offset": "end",
    "surprise": "surprisal",
    "surprise_class": "surprisal_class",
}


def load_label_table(path: str, required_columns=()) -> pd.DataFrame:
    """Read and normalize a canonical CSV/TSV label table."""
    sep = "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","
    table = pd.read_csv(path, sep=sep)
    table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed:")].copy()
    for old, new in COLUMN_ALIASES.items():
        if new not in table and old in table:
            table[new] = table[old]
    if "event_id" not in table:
        table["event_id"] = np.arange(len(table), dtype=int)
    if table["event_id"].isna().any() or table["event_id"].duplicated().any():
        raise ValueError("Label table event_id values must be present and unique")
    missing = {"start", *required_columns} - set(table)
    if missing:
        raise ValueError(f"Label table is missing columns: {sorted(missing)}")
    table["start"] = pd.to_numeric(table["start"], errors="coerce")
    if table["start"].isna().any():
        raise ValueError("Label table start values must all be numeric")
    return table.reset_index(drop=True)


def task_frame_from_labels(path: str, target_column: str) -> pd.DataFrame:
    table = load_label_table(path, required_columns=(target_column,))
    table = table.loc[table[target_column].notna()].copy()
    table["target"] = table[target_column]
    return table
