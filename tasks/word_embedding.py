import os
import re

import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import h5py
import numpy as np

from core.config import DataParams
from core import registry


@registry.register_task_data_getter()
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
        and a NumPy array of corresponding word-level embeddings under the header target.
    """
    import nltk
    from nltk.stem import WordNetLemmatizer as wl

    try:
        nltk.data.find("corpora/wordnet")
        print("WordNet already downloaded")
    except LookupError:
        print("Downloading WordNet...")
        nltk.download("wordnet")
        nltk.download("wordnet")

    transcript_path = os.path.join(
        data_params.data_root, "stimuli/gpt2-xl/transcript.tsv"
    )

    # Load transcript
    df_contextual = pd.read_csv(transcript_path, sep="\t", index_col=0)

    if data_params.embedding_type == "gpt-2xl":
        aligned_embeddings = get_gpt_2xl_embeddings(df_contextual, data_params)

    # Group sub-tokens together into words.
    df_word = df_contextual.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )
    df_word["norm_word"] = df_word.word.str.lower().str.replace(
        r"^[^\w\s]+|[^\w\s]+$", "", regex=True
    )
    df_word["lemmatized_word"] = df_word.norm_word.apply(lambda x: wl().lemmatize(x))

    if data_params.embedding_type == "gpt-2xl":
        df_word["target"] = list(aligned_embeddings)
    if data_params.embedding_type == "glove":
        df_word = get_glove_embeddings(df_word, data_params)
    elif data_params.embedding_type == "arbitrary":
        df_word = get_arbitrary_embeddings(df_word, data_params)

    if data_params.embedding_pca_dim:
        pca = PCA(n_components=data_params.embedding_pca_dim, svd_solver="auto")
        df_word.target = list(pca.fit_transform(df_word.target.tolist()))

    return df_word


# TODO: If we decide that we want to add support for more embeddings later it could be worth
# making this more generic with a function registry. For now I don't see a reason to make this
# more complicated though.
def get_gpt_2xl_embeddings(df_contextual, data_params: DataParams):
    """
    Loads GPT-2 XL contextual embeddings and aligns them to word-level units.

    This function:
    1. Loads sub-token-level GPT-2 XL embeddings from a specified HDF5 file.
    2. Groups the embeddings according to word indices provided in the contextual DataFrame.
    3. Averages sub-token embeddings to produce a single embedding vector per word.

    Args:
        df_contextual (pd.DataFrame): DataFrame containing token-level data, including `word_idx` for grouping.
        data_params (config.DataParams): Configuration object with data paths and the GPT-2 layer to extract.

    Returns:
        np.ndarray: A 2D array of shape (num_words, embedding_dim), where each row is a word-level embedding.
    """
    embedding_path = os.path.join(
        data_params.data_root, "stimuli/gpt2-xl/features.hdf5"
    )

    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{data_params.embedding_layer}"][...]

    # Group embeddings for each word (some are sub-tokenized).
    aligned_embeddings = []
    for _, group in df_contextual.groupby("word_idx"):  # group by word index
        indices = group.index.to_numpy()
        average_emb = contextual_embeddings[indices].mean(0)  # average features
        aligned_embeddings.append(average_emb)
    aligned_embeddings = np.stack(aligned_embeddings)

    return aligned_embeddings


def get_glove_embeddings(df_word, data_params: DataParams):
    """
    Retrieves GloVe embeddings for each word in the input DataFrame.

    This function:
    1. Loads GloVe word vectors from a specified text file.
    2. Preprocesses each word by lowercasing and removing surrounding punctuation (while preserving internal characters like apostrophes).
    3. Checks whether each preprocessed word exists in the GloVe vocabulary.
    4. Collects embeddings for words found in GloVe and flags presence for each word.
    5. Adds a new column `in_glove` to the DataFrame indicating if a word was successfully matched.

    Args:
        df_word (pd.DataFrame): DataFrame with a 'word' column containing word-level text entries.
        data_params (config.DataParams): Configuration object specifying the data root and GloVe file path.

    Returns:
        pd.DataFrame:
            - DataFrame with GloVe embeddings in embedding column.
    """
    glove_file = os.path.join(data_params.data_root, "glove/glove.6B.50d.txt")
    glove_vectors = KeyedVectors.load_word2vec_format(
        glove_file, binary=False, no_header=True
    )

    def preprocess_word(word):
        # Convert to lowercase
        word = word.lower()
        # Remove punctuation outside of words. i.e. keep apostrophes
        word = re.sub(r"\b[^\w\s]+|[^\w\s]+\b", "", word)
        return word

    words = df_word.word.tolist()
    preprocessed_words = [preprocess_word(word) for word in words]
    in_glove = []
    glove_embeddings = []

    for i, word in enumerate(preprocessed_words):
        if word in glove_vectors:
            glove_embeddings.append(glove_vectors[word])
            in_glove.append(True)
        else:
            in_glove.append(False)

    df_word["in_glove"] = in_glove
    df_word = df_word[df_word["in_glove"]].reset_index()

    glove_embeddings = np.stack(glove_embeddings)
    df_word["target"] = list(glove_embeddings)

    return df_word


def get_arbitrary_embeddings(df_word, data_params: DataParams):
    """
    Generates arbitrary (random) embeddings for each unique word in the input DataFrame.

    Parameters:
    -----------
    df_word : pandas.DataFrame
        A DataFrame containing a column named 'word', representing a list of words.
    data_params : DataParams
        An object containing configuration parameters, specifically `embedding_pca_dim`,
        which defines the dimensionality of the final embeddings.

    Returns:
    --------
    pd.DataFrame:
        df_word with arbitrary embeddings in embedding column

    Notes:
    ------
    - Embeddings are randomly sampled from a uniform distribution in the range [-1.0, 1.0]
      with an initial dimensionality of 50, then truncated or padded to match
      `embedding_pca_dim`.
    - Useful as a placeholder or for testing models where real word embeddings are not required.
    """
    words = df_word[data_params.word_column].tolist()
    unique_words = list(set(words))
    word_to_idx = {}
    for i, word in enumerate(words):
        if word not in word_to_idx:
            word_to_idx[word] = []
        word_to_idx[word].append(i)

    arbitrary_embeddings_per_word = np.random.uniform(
        low=-1.0, high=1.0, size=(len(unique_words), data_params.embedding_pca_dim)
    )
    arbitrary_embeddings = np.zeros((len(words), data_params.embedding_pca_dim))
    for i, word in enumerate(unique_words):
        for idx in word_to_idx[word]:
            arbitrary_embeddings[idx] = arbitrary_embeddings_per_word[i]

    df_word["target"] = list(arbitrary_embeddings)

    return df_word
