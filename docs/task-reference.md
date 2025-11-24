# Task Reference

Complete reference for all available tasks in the podcast benchmark framework.

## Overview

Tasks define what you want to decode from neural data. Each task provides a DataFrame with timestamps (`start`) and targets (`target`) that serve as training labels for your models.

All tasks are located in the `tasks/` directory and must be registered using the `@registry.register_task_data_getter()` decorator.

---

## Task List

- [word_embedding_decoding_task](#word_embedding_decoding_task)
- [sentence_onset_task](#sentence_onset_task)
- [content_noncontent_task](#content_noncontent_task)
- [pos_task](#pos_task)
- [gpt_surprise_task](#gpt_surprise_task)
- [gpt_surprise_multiclass_task](#gpt_surprise_multiclass_task)
- [volume_level_decoding_task](#volume_level_decoding_task)

---

## word_embedding_decoding_task

**File**: `tasks/word_embedding.py`

**Description**: Decode high-dimensional word embeddings from neural data. Supports GPT-2 XL contextual embeddings, GloVe static embeddings, or custom embeddings.

**Task Type**: Regression (high-dimensional continuous targets)

**Output**:
- `start`: Word start time in seconds
- `target`: Word embedding vector (list or array)

### Configuration Parameters

Configured via `WordEmbeddingConfig` in `task_specific_config`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_type` | string | `"gpt-2xl"` | Embedding type: `"gpt-2xl"`, `"glove"`, or `"arbitrary"` |
| `embedding_layer` | int | `None` | GPT-2 layer to extract (0-47 for GPT-2 XL) |
| `embedding_pca_dim` | int | `None` | Optional: reduce dimensionality with PCA |

### Embedding Types

**`gpt-2xl`**: Contextual embeddings from GPT-2 XL
- Requires transcript at `{data_root}/stimuli/gpt2-xl/transcript.tsv`
- Extracts embeddings from specified layer
- Handles sub-word tokenization automatically

**`glove`**: Static word embeddings (GloVe)
- Requires implementation in `tasks/word_embedding.py`
- Uses lemmatized word forms
- Fixed vectors per word type

**`arbitrary`**: Custom embedding implementation
- Requires implementation in `utils/word_embedding.py`
- Flexible for any embedding type

### Word Processing

The task automatically:
1. Groups sub-word tokens into full words using `word_idx`
2. Normalizes words (lowercase, remove punctuation)
3. Lemmatizes words using NLTK WordNet
4. Aligns embeddings to word boundaries

### Example Config

```yaml
task_config:
  task_name: word_embedding_decoding_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    embedding_type: gpt-2xl
    embedding_layer: 24
    embedding_pca_dim: 50  # Optional: reduce from 1600 to 50 dims
```

---

## volume_level_decoding_task

**File**: `tasks/volume_level.py`

**Description**: Continuous audio intensity decoding task. Extracts perceptual loudness (in dB) from the podcast audio using Hilbert envelope extraction, low-pass filtering, and optional sliding-window aggregation.

**Task Type**: Regression (continuous targets)

**Output**:
- `start`: Timestamp in seconds
- `target`: Log-amplitude (dB) representing perceptual loudness

### Configuration Parameters

Configured via `VolumeLevelConfig` in `task_specific_config`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_path` | string | `"stimuli/podcast.wav"` | Path to audio file (relative to `data_root` or absolute) |
| `target_sr` | int | `512` | Target sampling rate for envelope (Hz) |
| `audio_sr` | int | `44100` | Expected audio sampling rate (Hz) |
| `cutoff_hz` | float | `8.0` | Low-pass filter cutoff frequency (Hz) |
| `butter_order` | int | `4` | Butterworth filter order |
| `zero_phase` | bool | `true` | Use zero-phase filtering (filtfilt) vs causal (filt) |
| `log_eps` | float | auto | Epsilon for log compression (auto: peak * 1e-6) |
| `allow_resample_audio` | bool | `false` | Allow audio with different sample rate than expected |
| `window_size` | float | None | Optional: sliding window width in milliseconds |
| `hop_size` | float | `window_size` | Optional: sliding window hop size in milliseconds |

### Windowing Behavior

**Without windowing** (`window_size=None`):
- Returns per-sample dB values
- Timestamps are evenly spaced at `1/target_sr` intervals
- Formula: `20 * log10(envelope + log_eps)`

**With windowing**:
- Applies sliding RMS windows to the envelope
- Converts each RMS window to dB
- Timestamps are at window centers
- More robust to noise, better aligned with neural integration windows

### Example Config

```yaml
task_config:
  task_name: volume_level_decoding_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.2
  task_specific_config:
    audio_path: stimuli/podcast.wav
    target_sr: 512
    cutoff_hz: 8.0
    window_size: 200.0
    hop_size: 25.0
```

---

## content_noncontent_task

**File**: `tasks/content_noncontent.py`

**Description**: Binary classification of content words (nouns, verbs, adjectives, adverbs) vs non-content words (determiners, prepositions, etc.).

**Task Type**: Binary classification

**Output**:
- `start`: Word onset time in seconds
- `target`: `1.0` for content words, `0.0` for non-content words

### Configuration Parameters

Configured via `ContentNonContentConfig`:

| Parameter | Type | Default |
|-----------|------|---------|
| `content_noncontent_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` |

### Example Config

```yaml
task_config:
  task_name: content_noncontent_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    content_noncontent_path: processed_data/df_word_onset_with_pos_class.csv
```

---

## pos_task

**File**: `tasks/pos_task.py`

**Description**: Multi-class part-of-speech classification for words.

**Task Type**: Multi-class classification (5 classes: Noun, Verb, Adjective, Adverb, Other)

**Output**:
- `start`: Word onset time in seconds
- `target`: Class label (0-4)

### Configuration Parameters

Configured via `PosTaskConfig`:

| Parameter | Type | Default |
|-----------|------|---------|
| `pos_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` |

### Example Config

```yaml
task_config:
  task_name: pos_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    pos_path: processed_data/df_word_onset_with_pos_class.csv
```

---

## sentence_onset_task

**File**: `tasks/sentence_onset.py`

**Description**: Binary classification for detecting sentence onsets with negative sampling.

**Task Type**: Binary classification

**Output**:
- `start`: Time in seconds
- `target`: `1.0` for sentence onset, `0.0` for negative examples

### Configuration Parameters

Configured via `SentenceOnsetConfig`:

| Parameter | Type | Default |
|-----------|------|---------|
| `sentence_csv_path` | string | `"processed_data/all_sentences_podcast.csv"` |
| `negatives_per_positive` | int | `1` |
| `negative_margin_s` | float | `2.0` |

### Example Config

```yaml
task_config:
  task_name: sentence_onset_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    sentence_csv_path: processed_data/all_sentences_podcast.csv
    negatives_per_positive: 5
    negative_margin_s: 0.75
```

---

## gpt_surprise_task

**File**: `tasks/gpt_surprise.py`

**Description**: Regression task predicting GPT-2 XL surprise values.

**Task Type**: Regression (continuous targets)

**Output**:
- `start`: Word onset time in seconds
- `target`: GPT-2 XL surprise value

### Configuration Parameters

Configured via `GptSurpriseConfig`:

| Parameter | Type | Default |
|-----------|------|---------|
| `content_noncontent_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` |

### Example Config

```yaml
task_config:
  task_name: gpt_surprise_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    content_noncontent_path: processed_data/df_word_onset_with_pos_class.csv
```

---

## gpt_surprise_multiclass_task

**File**: `tasks/gpt_surprise.py`

**Description**: Multi-class classification of GPT-2 XL surprise levels.

**Task Type**: Multi-class classification (3 classes: Low, Medium, High surprise)

**Output**:
- `start`: Word onset time in seconds
- `target`: Class label (0-2)

### Configuration Parameters

Configured via `GptSurpriseConfig`:

| Parameter | Type | Default |
|-----------|------|---------|
| `content_noncontent_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` |

### Example Config

```yaml
task_config:
  task_name: gpt_surprise_multiclass_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    content_noncontent_path: processed_data/df_word_onset_with_pos_class.csv
```

---

## See Also

- [Configuration Guide](configuration.md): Full configuration reference
- [Adding a Task](adding-task.md): Step-by-step guide for implementing tasks
- [API Reference](api-reference.md): Detailed API documentation
