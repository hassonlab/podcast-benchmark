# Adding a New Task

Guide to implementing custom decoding tasks beyond word embedding prediction.

## Overview

A **task** defines what you're trying to decode from neural data. The default task is word embedding decoding, but you can create tasks for any prediction target aligned with temporal events in your data:
- Phoneme prediction
- Sentiment classification
- Grammatical role prediction
- Part-of-speech tagging
- Syllable-level features
- Any other prediction target with associated timing information

## Quick Reference

To add a new task:

1. [Create a task data getter function](#1-create-task-data-getter)
2. [Register the function](#2-register-the-function)
3. [Update your config](#3-update-config)
4. [Optional: Add custom metrics](#4-optional-custom-metrics)

---

## 1. Create Task Data Getter

Create a new file in tasks/ with a function that loads and processes your task-specific data.

### Function Signature

```python
from core.config import DataParams

def my_task_data_getter(data_params: DataParams) -> pd.DataFrame:
    """
    Load task-specific data.

    Args:
        data_params: DataParams object with configuration

    Returns:
        DataFrame with required columns:
        - start: Time to center neural data around (seconds)
        - target: Target variable for prediction
        - word: (Optional) The text/label for this event
    """
    pass
```

**Required DataFrame Columns**:
- `start` (float): Timestamp to center the neural data window around (in seconds)
- `target` (any): The prediction target (can be embeddings, labels, scalars, etc.)

**Optional Columns**:
- `word` (str): Text/label for the event (useful for zero-shot folds)
- Any other metadata you want to track

### Minimal Example

```python
from core.config import DataParams
import pandas as pd
import core.registry as registry

@registry.register_task_data_getter()
def constant_prediction_task(data_params: DataParams):
    """Simple task: predict a constant value."""
    # Load timing data
    transcript_path = os.path.join(
        data_params.data_root,
        "stimuli/gpt2-xl/transcript.tsv"
    )
    df = pd.read_csv(transcript_path, sep="\t")

    # Group tokens into words and get start times
    df_word = df.groupby("word_idx").agg(dict(start="first"))

    # Set target to constant (model learns to output 1.0)
    df_word["target"] = 1.0

    return df_word
```

---

## 2. Register the Function

Use the `@registry.register_task_data_getter()` decorator:

```python
import core.registry as registry

@registry.register_task_data_getter()
def my_custom_task(data_params: DataParams):
    # Your implementation
    return df_word
```

**Optional**: Specify a custom name:
```python
@registry.register_task_data_getter('custom_name')
def my_function(data_params: DataParams):
    ...
```

---

## 3. Update Config

Set the `task_name` in your config file:

```yaml
# Specify your task
task_name: my_custom_task

# Pass task-specific parameters via task_params
data_params:
  task_params:
    custom_param: value
    another_param: 123
```

Access task parameters in your function:
```python
def my_custom_task(data_params: DataParams):
    custom_value = data_params.task_params['custom_param']
    ...
```

---

## 4. Optional: Custom Metrics

Define metrics specific to your task. Add them to the appropriate file in the `metrics/` package based on the metric type:

- **Regression metrics** → `metrics/regression_metrics.py`
- **Classification metrics** → `metrics/classification_metrics.py`
- **Embedding metrics** → `metrics/embedding_metrics.py`
- **Utility functions** → `metrics/utils.py`

Example:

```python
import torch
from core.registry import register_metric

@register_metric('my_accuracy')
def my_accuracy_metric(predicted: torch.Tensor, groundtruth: torch.Tensor):
    """
    Custom metric for your task.

    Args:
        predicted: Model predictions [batch_size, ...]
        groundtruth: Ground truth targets [batch_size, ...]

    Returns:
        Scalar metric value
    """
    correct = (predicted.argmax(dim=1) == groundtruth).float()
    return correct.mean()
```

Then add to your config:
```yaml
training_params:
  losses: [cross_entropy]
  metrics: [my_accuracy, cosine_sim]
```

The metrics are automatically registered when the package is imported.

---

## Examples

See the `tasks/` directory for complete examples:
- `tasks/word_embedding.py` - Default task for decoding word embeddings
- `tasks/placeholder_task.py` - Minimal example showing required structure
- `tasks/content_noncontent.py` - Binary classification example
- `tasks/pos_task.py` - Part-of-speech tagging example
- `tasks/sentence_onset.py` - Sentence onset detection
- `tasks/gpt_surprise.py` - Regression and multiclass classification examples
- `tasks/volume_level.py` - Audio feature prediction with custom config setter

For detailed documentation on all available tasks and their configuration parameters, see the [Task Reference](task-reference.md)

---

## Task Parameters

Pass task-specific configuration via `data_params.task_params`:

```yaml
data_params:
  task_params:
    feature_file: "path/to/features.csv"
    threshold: 0.5
    use_lemmas: true
```

Access in your function:
```python
from core.config import DataParams

def my_task(data_params: DataParams):
    feature_file = data_params.task_params['feature_file']
    threshold = data_params.task_params.get('threshold', 0.5)
    ...
```

---

## See Also

- [Task Reference](task-reference.md) - Complete reference for all available tasks
- [Configuration Guide](configuration.md) - How to configure tasks
- [API Reference](api-reference.md) - Task data getter API
- `tasks/` directory - Complete task examples
