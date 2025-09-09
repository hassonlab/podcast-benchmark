import os

import pandas as pd
from sklearn.decomposition import PCA

from config import DataParams
import embeddings
import registry


@registry.task_data_getter_registry()
def word_embedding_decoding_task(data_params: DataParams):
    """
    Loads and processes word-level data and retrieves corresponding embeddings based on specified parameters.

    This function performs the following steps:
    1. Loads a transcript file containing token-level information.
    2. Retrieves aligned embeddings for each token or word, depending on the specified embedding type.
    3. Groups sub-token entries into full words using word indices.
    4. Optionally applies PCA to reduce the dimensionality of the embeddings.

    Args:
        data_params (DataParams): Configuration object containing paths, embedding type, and PCA settings.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A DataFrame containing word-level information (word, start time, end time),
        and a NumPy array of corresponding word-level embeddings.
    """
    transcript_path = os.path.join(
        data_params.data_root, "stimuli/gpt2-xl/transcript.tsv"
    )

    # Load transcript
    df_contextual = pd.read_csv(transcript_path, sep="\t", index_col=0)

    if data_params.embedding_type == "gpt-2xl":
        aligned_embeddings = embeddings.get_gpt_2xl_embeddings(
            df_contextual, data_params
        )

    # Group sub-tokens together into words.
    df_word = df_contextual.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )

    if data_params.embedding_type == "gpt-2xl":
        df_word["embedding"] = list(aligned_embeddings)
    if data_params.embedding_type == "glove":
        df_word = embeddings.get_glove_embeddings(df_word, data_params)
    elif data_params.embedding_type == "arbitrary":
        df_word = embeddings.get_arbitrary_embeddings(df_word, data_params)

    if data_params.embedding_pca_dim:
        pca = PCA(n_components=data_params.embedding_pca_dim, svd_solver="auto")
        df_word.target = list(pca.fit_transform(df_word.embedding.tolist()))

    return df_word
