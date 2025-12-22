# Task Reference

Complete reference for all available tasks in the podcast benchmark framework.

## Overview

Tasks define what you want to decode from neural data. Each task provides a DataFrame with timestamps (`start`) and targets (`target`) that serve as training labels for your models.

All tasks are located in the `tasks/` directory and must be registered using the `@registry.register_task_data_getter()` decorator.

For performance benchmarks on each task, see [Baseline Results](baseline-results.md).

---

## Task List

- [word_embedding_decoding_task](#word_embedding_decoding_task)
- [volume_level_decoding_task](#volume_level_decoding_task)
- [content_noncontent_task](#content_noncontent_task)
- [pos_task](#pos_task)
- [sentence_onset_task](#sentence_onset_task)
- [gpt_surprise_task](#gpt_surprise_task)
- [gpt_surprise_multiclass_task](#gpt_surprise_multiclass_task)
- [placeholder_task](#placeholder_task)

---

## word_embedding_decoding_task

**File**: `tasks/word_embedding.py`

**Description**: Decode high-dimensional word embeddings from neural data. Supports GPT-2 XL contextual embeddings, GloVe static embeddings, or custom embeddings.

**Task Type**: Regression (high-dimensional continuous targets)

**Output**:
- `start`: Word start time in seconds
- `target`: Word embedding vector (list or array)

### Configuration Parameters

Parameters are specified in `data_params` (not `task_params`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_type` | string | Required | Embedding type: `"gpt-2xl"`, `"glove"`, or `"arbitrary"` |
| `embedding_layer` | int | 24 | GPT-2 layer to extract (0-47 for GPT-2 XL) |
| `embedding_pca_dim` | int | None | Optional: reduce dimensionality with PCA |

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
data_params:
  task_name: word_embedding_decoding_task
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

All parameters are specified in `data_params.task_params`:

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
data_params:
  task_name: volume_level_decoding_task
  task_params:
    audio_path: "stimuli/podcast.wav"
    target_sr: 512
    cutoff_hz: 8.0
    window_size: 625    # 625ms windows
    hop_size: 100       # 100ms hops
    zero_phase: true
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content_noncontent_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` | Path to CSV with word annotations (relative to cwd or absolute) |

### CSV Format

Expected columns:
- `onset`: Word onset time in seconds
- `is_content`: Binary label (1=content, 0=non-content)

### Example Config

```yaml
data_params:
  task_name: content_noncontent_task
  task_params:
    content_noncontent_path: "processed_data/df_word_onset_with_pos_class.csv"
```

---

## pos_task

**File**: `tasks/pos_task.py`

**Description**: Multi-class part-of-speech classification for words.

**Task Type**: Multi-class classification (5 classes)

**Classes**:
- `0`: Noun
- `1`: Verb
- `2`: Adjective
- `3`: Adverb
- `4`: Other

**Output**:
- `start`: Word onset time in seconds
- `target`: Class label (0-4)

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` | Path to CSV with POS annotations (relative to cwd or absolute) |

### CSV Format

Expected columns:
- `onset`: Word onset time in seconds
- `pos_class`: Integer class label (0-4)

### Example Config

```yaml
data_params:
  task_name: pos_task
  task_params:
    pos_path: "processed_data/df_word_onset_with_pos_class.csv"
```

---

## sentence_onset_task

**File**: `tasks/sentence_onset.py`

**Description**: Binary classification for detecting sentence onsets. Includes positive examples at sentence starts and negative examples sampled away from onsets.

**Task Type**: Binary classification

**Output**:
- `start`: Time in seconds
- `target`: `1.0` for sentence onset, `0.0` for negative examples

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence_csv_path` | string | `"processed_data/all_sentences_podcast.csv"` | Path to CSV with sentence boundaries (relative to cwd or absolute) |
| `negatives_per_positive` | int | `1` | Number of negative examples to sample per positive |
| `negative_margin_s` | float | `2.0` | Minimum time (seconds) after onset before sampling negatives |

### CSV Format

Expected columns:
- `sentence_onset`: Sentence start time in seconds
- `sentence_offset`: Sentence end time in seconds

### Negative Sampling Strategy

For each sentence:
1. Sample negatives between `onset + negative_margin_s` and `offset - window_width`
2. This ensures negatives don't overlap with the actual onset window
3. Uses `window_width` from `data_params` to avoid sampling too close to sentence end

### Example Config

```yaml
data_params:
  task_name: sentence_onset_task
  window_width: 0.625  # Used for negative sampling
  task_params:
    sentence_csv_path: "processed_data/all_sentences_podcast.csv"
    negatives_per_positive: 2
    negative_margin_s: 2.0
```

---

## gpt_surprise_task

**File**: `tasks/gpt_surprise.py`

**Description**: Regression task predicting GPT-2 XL surprise values (negative log probability) for each word.

**Task Type**: Regression (continuous targets)

**Output**:
- `start`: Word onset time in seconds
- `target`: GPT-2 XL surprise value (higher = more surprising/unpredictable)

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content_noncontent_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` | Path to CSV with word annotations (relative to cwd or absolute) |

### CSV Format

Expected columns:
- `onset`: Word onset time in seconds
- `surprise`: GPT-2 XL surprise value

### Example Config

```yaml
data_params:
  task_name: gpt_surprise_task
  task_params:
    content_noncontent_path: "processed_data/df_word_onset_with_pos_class.csv"
```

---

## gpt_surprise_multiclass_task

**File**: `tasks/gpt_surprise.py`

**Description**: Multi-class classification of GPT-2 XL surprise levels. Surprise values are binned based on mean and standard deviation.

**Task Type**: Multi-class classification (3 classes)

**Classes**:
- `0`: Low surprise (< mean - std)
- `1`: Medium surprise (within std of mean)
- `2`: High surprise (> mean + std)

**Output**:
- `start`: Word onset time in seconds
- `target`: Class label (0-2)

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content_noncontent_path` | string | `"processed_data/df_word_onset_with_pos_class.csv"` | Path to CSV with word annotations (relative to cwd or absolute) |

### CSV Format

Expected columns:
- `onset`: Word onset time in seconds
- `surprise_class`: Integer class label (0-2)

### Example Config

```yaml
data_params:
  task_name: gpt_surprise_multiclass_task
  task_params:
    content_noncontent_path: "processed_data/df_word_onset_with_pos_class.csv"
```

---

## placeholder_task

**File**: `tasks/placeholder_task.py`

**Description**: Minimal example task for testing. Returns constant targets (always 1.0).

**Task Type**: Regression (trivial)

**Output**:
- `start`: Word start time in seconds
- `target`: `1.0` (constant)

### Configuration Parameters

None. This task takes no parameters.

### Purpose

This is a template showing the minimum requirements for a task:
1. Register with `@registry.register_task_data_getter()`
2. Accept `data_params: DataParams` argument
3. Return DataFrame with `start` and `target` columns

Models will learn to always predict 1.0, making this useful only for testing infrastructure.

### Example Config

```yaml
data_params:
  task_name: placeholder_task
```

---

## See Also

- [Baseline Results](baseline-results.md): Performance benchmarks for all tasks
- [Configuration Guide](configuration.md): Full configuration reference
- [Adding a Task](adding-task.md): Step-by-step guide for implementing tasks
- [API Reference](api-reference.md): Detailed API documentation
