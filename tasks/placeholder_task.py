import os

import pandas as pd

from core.config import DataParams
from core import registry


@registry.register_task_data_getter()
def placeholder_task(data_params: DataParams):
    # Just an example of the bare minimum of what's needed for
    transcript_path = os.path.join(
        data_params.data_root, "stimuli/gpt2-xl/transcript.tsv"
    )

    # Load transcript
    df_contextual = pd.read_csv(transcript_path, sep="\t", index_col=0)

    # Group sub-tokens together into words.
    df_word = df_contextual.groupby("word_idx").agg(dict(start="first"))

    # Just fill in a 1 for the column "target" since this is a placeholder. Model will learn to always output 1.
    df_word["target"] = 1.0

    # So now our dataframe has columns start and target and we can pass it into our training code.
    return df_word
