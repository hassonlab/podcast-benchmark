import math
import os
import warnings

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.signal import butter, hilbert, resample_poly, sosfilt, sosfiltfilt
from sklearn.decomposition import PCA

from config import DataParams
import embeddings
import registry
from vol_lvl_ridge_utils import compute_window_hop, sliding_window_rms


@registry.register_task_data_getter()
def volume_level_encoding_task(data_params: DataParams):
    """Prepare continuous audio-intensity targets for decoding.

      1. Load the podcast waveform from disk.
      2. Compute the Hilbert envelope and apply a Butterworth low-pass filter.
      3. Resample the envelope to match neural sampling rate expectations.
      4. Log-compress the envelope to produce perceptual loudness values.

    Optional sliding-window aggregation can be enabled via ``task_params`` by
    specifying window and hop sizes (ms). Targets are timestamped at the
    window centers, and each window is reduced to a single RMS value.

    Args:
        data_params (DataParams): Configuration object containing data paths,
            neural sampling rate, and task-specific parameters.

    Returns:
        pd.DataFrame: Continuous targets with columns ``start`` (seconds) and
        ``target`` (log-amplitude or windowed representation) ready for the
        decoding pipeline.
    """
    
    tp = getattr(data_params, "task_params", {}) or {}

    audio_rel_path = tp.get("audio_path", os.path.join("stimuli", "podcast.wav"))
    target_sr = int(tp.get("target_sr", 512))
    expected_audio_sr = int(tp.get("audio_sr", 44100))
    cutoff_hz = float(tp.get("cutoff_hz", 8.0))
    butter_order = int(tp.get("butter_order", 4))
    zero_phase = bool(tp.get("zero_phase", True))
    log_eps = tp.get("log_eps")
    allow_audio_resample = bool(tp.get("allow_resample_audio", False))
    window_size_ms = tp.get("window_size")
    hop_size_ms = tp.get("hop_size")

    audio_path = audio_rel_path if os.path.isabs(audio_rel_path) else os.path.join(
        data_params.data_root, audio_rel_path
    )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at '{audio_path}'.")

    sr, waveform = wavfile.read(audio_path)

    if waveform.size == 0:
        raise ValueError(f"Loaded empty audio file from '{audio_path}'.")

    if sr != expected_audio_sr:
        if allow_audio_resample:
            warnings.warn(
                f"Audio sample rate {sr} Hz does not match expected {expected_audio_sr} Hz. "
                "Continuing with the actual sample rate.",
                RuntimeWarning,
            )
        else:
            raise ValueError(
                f"Expected audio sampled at {expected_audio_sr} Hz, got {sr} Hz. "
                "Provide a file with the expected rate or enable 'allow_resample_audio'."
            )

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        max_abs = max(abs(info.min), abs(info.max)) or 1
        waveform = waveform.astype(np.float32) / float(max_abs)
    else:
        waveform = waveform.astype(np.float32)

    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)

    if cutoff_hz <= 0:
        raise ValueError("'cutoff_hz' must be positive.")

    nyquist = 0.5 * sr
    if cutoff_hz >= nyquist:
        raise ValueError(f"'cutoff_hz' ({cutoff_hz}) must be below Nyquist ({nyquist}).")

    sos = butter(butter_order, cutoff_hz / nyquist, btype="low", output="sos")

    if zero_phase:
        try:
            smoothed = sosfiltfilt(sos, envelope)
        except ValueError as exc:
            warnings.warn(
                f"Zero-phase filtering failed ({exc}); falling back to causal filtering.",
                RuntimeWarning,
            )
            smoothed = sosfilt(sos, envelope)
    else:
        smoothed = sosfilt(sos, envelope)

    if target_sr <= 0:
        raise ValueError("'target_sr' must be positive.")

    g = math.gcd(target_sr, sr)
    up = target_sr // g
    down = sr // g
    envelope_ds = resample_poly(smoothed, up, down)

    # Keep linear envelope non-negative
    envelope_ds = np.clip(envelope_ds, 0.0, None)

    n_samples = envelope_ds.shape[0]

    # If no windowing requested, convert the per-sample linear envelope to dB
    if window_size_ms is None:
        if log_eps is None:
            peak = float(envelope_ds.max()) if envelope_ds.size else 0.0
            log_eps = max(1e-12, peak * 1e-6)
        env_db = 20.0 * np.log10(envelope_ds + float(log_eps))

        times = np.arange(n_samples, dtype=np.float32) / float(target_sr)
        df = pd.DataFrame(
            {
                "start": times.astype(np.float32),
                "target": env_db.astype(np.float32),
            }
        )
        df.attrs["window_params"] = None
        return df

    try:
        width_ms = float(window_size_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("'window_size' must be convertible to milliseconds.") from exc
    if width_ms <= 0:
        raise ValueError("'window_size' must be > 0 milliseconds.")

    stride_ms = float(hop_size_ms) if hop_size_ms is not None else width_ms
    if stride_ms <= 0:
        raise ValueError("'hop_size' must be > 0 milliseconds.")

    width = width_ms / 1000.0
    stride = stride_ms / 1000.0

    window_samples, hop_samples, effective_sr = compute_window_hop(
        target_sr, width_ms, stride_ms
    )

    if window_samples > n_samples:
        raise ValueError(
            f"Requested window of {window_samples} samples exceeds envelope length {n_samples}."
        )
    
    # Compute RMS over the linear envelope, then convert each window RMS to dB.
    targets_linear = sliding_window_rms(envelope_ds, window_samples, hop_samples)

    if log_eps is None:
        peak = float(targets_linear.max()) if targets_linear.size else 0.0
        log_eps = max(1e-12, peak * 1e-6)

    targets = 20.0 * np.log10(targets_linear + float(log_eps))

    starts = np.arange(0, n_samples - window_samples + 1, hop_samples, dtype=int)
    if starts.size == 0:
        raise ValueError(
            "hop_size/window_size combination produced zero windows; adjust parameters."
        )

    centers = (starts + (window_samples - 1) / 2.0) / float(target_sr)

    if targets.ndim == 1:
        target_column = targets
    else:
        target_column = [window for window in targets]

    df = pd.DataFrame(
        {
            "start": centers.astype(np.float32),
            "target": target_column,
        }
    )

    df.attrs["window_params"] = {
        # Notebook-compatible keys
        "mode": "rms",
        "window_ms": width_ms,
        "hop_ms": stride_ms,
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "effective_sr": effective_sr,
        # Backwards-compatibility aliases used in earlier iterations of the codebase
        "window_size_ms": width_ms,
        "hop_size_ms": stride_ms,
        "window_size_s": width,
        "hop_size_s": stride,
        "window_reduction": "rms",
        # dB conversion was applied after RMS computation
        "db_after_rms": True,
    }

    return df



@registry.register_config_setter(name="volume_level_config_setter")
def volume_level_config_setter(experiment_config, raws, df_word):
    """Align experiment config to volume-level task outputs.

    This setter will:
      - Set data_params.window_width (seconds) from task_params.window_size (ms)
        so that neural windows align with audio windows by default.
      - Set the data preprocessing function to 'window_rms' if not already set,
        so each neural window is reduced to RMS amplitudes like the audio.

    The function is defensive: it will only set window_width if it is unset
    (<= 0 or falsy) and will not overwrite an explicitly provided preprocessing
    function unless none is present.
    """

    # Ensure nested objects exist
    dp = experiment_config.data_params
    tp = getattr(dp, "task_params", {}) or {}

    # If the task defines a window size in ms, set the neural window width (s)
    window_size_ms = tp.get("window_size")
    if window_size_ms is not None:
        try:
            width_ms = float(window_size_ms)
            if width_ms > 0:
                # Only override if not already set to a positive value
                if not getattr(dp, "window_width", None) or dp.window_width <= 0:
                    dp.window_width = width_ms / 1000.0
        except (TypeError, ValueError):
            # Ignore invalid values and leave window_width unchanged
            pass

    if not dp.preprocessing_fn_name:
        model_ctor = getattr(experiment_config, "model_constructor_name", None)
        # If no model specified, default to window RMS preprocessing for neural data.
        # Additionally, for ridge-like constructors we also want RMS inputs so detect
        # common ridge constructor names (e.g., 'ridge' in the name) and enable RMS.
        if not model_ctor:
            dp.preprocessing_fn_name = "window_rms"
        else:
            try:
                ctor_name = str(model_ctor).lower()
            except Exception:
                ctor_name = ""
            if "ridge" in ctor_name or "torch_ridge" in ctor_name:
                dp.preprocessing_fn_name = "window_rms"

    # Auto-fill model_params.input_channels when missing by summing channels from raws
    mp = getattr(experiment_config, "model_params", None) or {}
    if mp.get("input_channels") in (None, 0, "", False):
        try:
            total_ch = sum(len(getattr(r, "ch_names", getattr(r, "info", {}).get("ch_names", []))) for r in raws) if raws else None
            # Fallback to info['nchan'] if available
            if (not total_ch or total_ch == 0) and raws:
                total_ch = sum(getattr(r, "info", {}).get("nchan", 0) for r in raws)
        except Exception:
            total_ch = None

        if total_ch and total_ch > 0:
            mp["input_channels"] = int(total_ch)
            experiment_config.model_params = mp

    # No change to experiment_config identity, return for convenience
    return experiment_config


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
