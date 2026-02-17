import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.decomposition import PCA
import h5py

from core.config import BaseTaskConfig, TaskConfig
from core import registry


@dataclass
class WhisperEmbeddingConfig(BaseTaskConfig):
    """Configuration for whisper_embedding_decoding_task."""

    embedding_pca_dim: Optional[int] = None


@registry.register_task_data_getter(config_type=WhisperEmbeddingConfig)
def whisper_embedding_decoding_task(task_config: TaskConfig):
    """
    Loads and processes word-level data and retrieves corresponding embeddings based on specified parameters.

    This function performs the following steps:
    1. Loads a transcript file containing token-level information.
    2. Retrieves aligned embeddings for each token or word, depending on the specified embedding type.
    3. Groups sub-token entries into full words using word indices.
    4. Optionally applies PCA to reduce the dimensionality of the embeddings.

    Args:
        task_config (TaskConfig): Configuration object containing task-specific config and data params.

    Returns:
        pd.DataFrame: A DataFrame containing word-level information (word, start time, end time),
        and corresponding word-level embeddings under the header target.
    """
    config: WhisperEmbeddingConfig = task_config.task_specific_config
    data_params = task_config.data_params

    # Load transcript
    transcript_path = os.path.join(
        data_params.data_root, "stimuli/whisper-medium/transcript.tsv"
    )
    df_word = pd.read_csv(transcript_path, sep="\t", index_col=0)

    # Load embeddings
    embedding_path = os.path.join(
        data_params.data_root, "stimuli/whisper-medium/features.hdf5"
    )
    with h5py.File(embedding_path, "r") as f:
        whisper_embeddings = f["vectors"][...]

    df_word["target"] = list(whisper_embeddings)

    if config.embedding_pca_dim:
        pca = PCA(n_components=config.embedding_pca_dim, svd_solver="auto")
        df_word.target = list(pca.fit_transform(df_word.target.tolist()))

    return df_word
