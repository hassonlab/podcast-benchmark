import os

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from config import DataParams
import embeddings
import registry


@registry.register_task_data_getter()
def content_noncontent_task(data_params: DataParams):
    """
    Binary classification dataset for content vs non-content word classification.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 for content, 0.0 non-content

    Notes:
      - Uses `data_params.content_noncontent_path` if provided; else defaults to
        `<data_root>/podcast-benchmark/df_word_onset_with_pos_class.csv.
      
    """
    # Pull task-specific params from data_params.task_params with local defaults
    tp = getattr(data_params, "task_params", {}) or {}

    # Resolve CSV path
    default_csv = os.path.join(os.getcwd(), "df_word_onset_with_pos_class.csv")
    csv_path = tp.get("content_noncontent_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df=pd.DataFrame()
    df['start']=df1['onset']  # convert samples to seconds
    df['target']=df1['is_content']

    print(f"\n=== Content Non-content words DATASET ===")
    print(f"Total examples: {len(df)}")
    print(f"Positives: {np.sum(df.target==1)}")
    print(f"Negatives: {len(df) - np.sum(df.target==1)}")

    return df

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
    default_csv = os.path.join(os.getcwd(), "df_word_onset_with_pos_class.csv")
    csv_path = tp.get("pos_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df=pd.DataFrame()
    df['start']=df1['onset']  # convert samples to seconds
    df['target']=df1['pos_class']

    print(f"\n=== Parts of Speech DATASET ===")
    print(f"Total examples: {len(df)}")
    print(f"Noun: {np.sum(df.target==0)}")
    print(f"Verb: {np.sum(df.target==1)}")
    print(f"Adjective: {np.sum(df.target==2)}")
    print(f"Adverb: {np.sum(df.target==3)}")
    print(f"Other: {np.sum(df.target==4)}")

    return df




@registry.register_task_data_getter()
def sentence_onset_task(data_params: DataParams):
    """
    Binary classification dataset for sentence onset detection.

    Returns a DataFrame with columns:
      - start: time (in seconds) to center the neural window
      - target: 1.0 for sentence onset, 0.0 for negatives sampled away from onsets

    Notes:
      - Uses `data_params.sentence_csv_path` if provided; else defaults to
        `<data_root>/all_sentences_podcast.csv`.
      - Negative examples are sampled within each sentence, at least
        `negative_margin_s` seconds after onset and at least one window width
        before the sentence end.
    """
    # Pull task-specific params from data_params.task_params with local defaults
    tp = getattr(data_params, "task_params", {}) or {}

    # Resolve CSV path
    default_csv = os.path.join(data_params.data_root, "all_sentences_podcast.csv")
    csv_path = tp.get("sentence_csv_path", default_csv)

    df = pd.read_csv(csv_path, index_col=0)

    # Convert ms -> seconds if needed (CSV stores ms; pipeline expects seconds)
    # Heuristic: if values are large (e.g., >1e4), treat as ms.
    if df["sentence_onset"].max() > 1e4 or df["sentence_offset"].max() > 1e4:
        df["sentence_onset"] = df["sentence_onset"] / 1000.0
        df["sentence_offset"] = df["sentence_offset"] / 1000.0

    # Expect columns: sentence_onset, sentence_offset
    if not {"sentence_onset", "sentence_offset"}.issubset(df.columns):
        raise ValueError(
            "Expected columns 'sentence_onset' and 'sentence_offset' in sentence CSV"
        )

    onsets = df["sentence_onset"].to_numpy()
    offsets = df["sentence_offset"].to_numpy()

    # Positives
    pos = pd.DataFrame({"start": onsets, "target": 1.0})

    # Negatives: sample away from onsets within the same sentence when possible
    import numpy as np

    window = getattr(data_params, "window_width", 0.625) or 0.625
    negatives_per_positive = int(tp.get("negatives_per_positive", 1))
    negative_margin_s = float(tp.get("negative_margin_s", 2.0))

    rng = np.random.default_rng(0)
    neg_starts = []
    for onset, offset in zip(onsets, offsets):
        # Start sampling after a margin to avoid including the onset in the window
        left = onset + negative_margin_s
        # Ensure we can still place a full window without touching sentence end
        right = max(left, offset - window - 1e-3)
        if right > left:
            samples = rng.uniform(left, right, size=max(0, negatives_per_positive))
            neg_starts.extend(samples.tolist())

    neg = pd.DataFrame({"start": neg_starts, "target": 0.0})

    df_out = (
        pd.concat([pos, neg], ignore_index=True)
        .sort_values("start")
        .reset_index(drop=True)
    )

    # Print dataset summary for inspection
    print(f"\n=== SENTENCE ONSET DATASET ===")
    print(f"Total examples: {len(df_out)}")
    print(f"Positives: {len(pos)} ({len(pos)/len(df_out)*100:.1f}%)")
    print(f"Negatives: {len(neg)} ({len(neg)/len(df_out)*100:.1f}%)")
    print(f"Time range: {df_out['start'].min():.2f}s - {df_out['start'].max():.2f}s")
    print(f"First 10 examples:")
    print(df_out.head(10))
    print("=" * 50)

    return df_out

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
    default_csv = os.path.join(os.getcwd(), "df_word_onset_with_pos_class.csv")
    csv_path = tp.get("content_noncontent_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df=pd.DataFrame()
    df['start']=df1['onset']  # convert samples to seconds
    df['target']=df1['surprise']

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
    default_csv = os.path.join(os.getcwd(), "df_word_onset_with_pos_class.csv")
    csv_path = tp.get("content_noncontent_path", default_csv)

    df1 = pd.read_csv(csv_path, index_col=0)

    df=pd.DataFrame()
    df['start']=df1['onset']  # convert samples to seconds
    df['target']=df1['surprise_class']

    return df


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
        aligned_embeddings = embeddings.get_gpt_2xl_embeddings(
            df_contextual, data_params
        )

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
        df_word = embeddings.get_glove_embeddings(df_word, data_params)
    elif data_params.embedding_type == "arbitrary":
        df_word = embeddings.get_arbitrary_embeddings(df_word, data_params)

    if data_params.embedding_pca_dim:
        pca = PCA(n_components=data_params.embedding_pca_dim, svd_solver="auto")
        df_word.target = list(pca.fit_transform(df_word.target.tolist()))

    return df_word


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
