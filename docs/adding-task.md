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

1. [Define a task config dataclass](#1-define-task-config-dataclass)
2. [Create a task data getter function](#2-create-task-data-getter)
3. [Register the function](#3-register-the-function)
4. [Update your config](#4-update-config)
5. [Optional: Using input_fields](#5-optional-using-input_fields)
6. [Optional: Add custom metrics](#6-optional-custom-metrics)

---

## 1. Define Task Config Dataclass

Create a dataclass in your task file that defines the task-specific configuration parameters.

```python
from dataclasses import dataclass
from core.config import BaseTaskConfig

@dataclass
class MyTaskConfig(BaseTaskConfig):
    """Configuration for my_task."""
    csv_path: str = "processed_data/my_task_data.csv"
    threshold: float = 0.5
    use_special_mode: bool = False
```

**Important**:
- Must inherit from `BaseTaskConfig`
- Define all task-specific parameters with type hints
- Provide sensible defaults where appropriate
- Do NOT duplicate fields that belong in `DataParams` (like `data_root`, `window_width`, `subject_ids`)
- `BaseTaskConfig` includes an `input_fields` parameter (optional list of column names from your DataFrame to pass as additional model inputs)

---

## 2. Create Task Data Getter

Create a new file in `tasks/` with a function that loads and processes your task-specific data.

### Function Signature

```python
from core.config import TaskConfig
import pandas as pd

def my_task_data_getter(task_config: TaskConfig) -> pd.DataFrame:
    """
    Load task-specific data.

    Args:
        task_config: TaskConfig containing data_params and task_specific_config

    Returns:
        DataFrame with required columns:
        - start: Time to center neural data around (seconds)
        - target: Target variable for prediction
        - word: (Optional) The text/label for this event
    """
    # Access task-specific config
    config: MyTaskConfig = task_config.task_specific_config
    # Access shared data params
    data_params = task_config.data_params

    # Your implementation here
    pass
```

**Required DataFrame Columns**:
- `start` (float): Timestamp to center the neural data window around (in seconds)
- `target` (any): The prediction target (can be embeddings, labels, scalars, etc.)

**Optional Columns**:
- `word` (str): Text/label for the event (useful for zero-shot folds)
- Any columns specified in `input_fields` (will be passed as kwargs to the model)
- Any other metadata you want to track

### Minimal Example

```python
from dataclasses import dataclass
from core.config import BaseTaskConfig, TaskConfig
import pandas as pd
import os
import core.registry as registry

@dataclass
class ConstantPredictionConfig(BaseTaskConfig):
    """Configuration for constant_prediction_task."""
    target_value: float = 1.0

@registry.register_task_data_getter(config_type=ConstantPredictionConfig)
def constant_prediction_task(task_config: TaskConfig):
    """Simple task: predict a constant value."""
    config: ConstantPredictionConfig = task_config.task_specific_config
    data_params = task_config.data_params

    # Load timing data
    transcript_path = os.path.join(
        data_params.data_root,
        "stimuli/gpt2-xl/transcript.tsv"
    )
    df = pd.read_csv(transcript_path, sep="\t")

    # Group tokens into words and get start times
    df_word = df.groupby("word_idx").agg(dict(start="first"))

    # Set target to constant (model learns to output the configured value)
    df_word["target"] = config.target_value

    return df_word
```

---

## 3. Register the Function

Use the `@registry.register_task_data_getter()` decorator with the `config_type` parameter:

```python
import core.registry as registry

@registry.register_task_data_getter(config_type=MyTaskConfig)
def my_custom_task(task_config: TaskConfig):
    config: MyTaskConfig = task_config.task_specific_config
    data_params = task_config.data_params
    # Your implementation
    return df_word
```

**Important**:
- The `config_type` parameter is **required** and must be your task config dataclass
- The function name will be used as the task name in configs (unless you override with `name` parameter)

**Optional**: Specify a custom name:
```python
@registry.register_task_data_getter(name='custom_name', config_type=MyTaskConfig)
def my_function(task_config: TaskConfig):
    ...
```

---

## 4. Update Config

Create a YAML config with the new nested structure:

```yaml
task_config:
  task_name: my_custom_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
    # ... other shared data params
  task_specific_config:
    csv_path: "processed_data/my_task_data.csv"
    threshold: 0.5
    use_special_mode: true

model_constructor_name: my_model
model_params:
  # ... model params

training_params:
  # ... training params
```

The task-specific parameters are now type-safe and validated at runtime!

---

## 5. Optional: Using input_fields

If your model needs additional inputs beyond neural data, use the `input_fields` parameter to specify which DataFrame columns should be passed to your model as kwargs.

### Example: Passing word IDs to a model

```python
@dataclass
class MyTaskConfig(BaseTaskConfig):
    """Configuration for task requiring word IDs."""
    input_fields: Optional[list[str]] = field(default_factory=lambda: ["word_id"])

@registry.register_task_data_getter(config_type=MyTaskConfig)
def my_task(task_config: TaskConfig):
    config: MyTaskConfig = task_config.task_specific_config

    # Create DataFrame with required columns
    df = pd.DataFrame({
        'start': [0.0, 1.0, 2.0],
        'target': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        'word_id': [42, 43, 44]  # This will be passed to model
    })
    return df
```

Your model's forward method should accept these fields as keyword arguments:

```python
def forward(self, neural_data, word_id=None, **kwargs):
    # word_id will be a tensor of shape [batch_size]
    if word_id is not None:
        # Use word_id in your model
        embeddings = self.word_embedding(word_id)
    # ...
```

**Important**:
- All fields in `input_fields` must be columns in the returned DataFrame
- These columns will be converted to tensors and passed as kwargs during training
- Handle None values in your model if the field might not be provided

---

## 6. Optional: Custom Metrics

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
- `tasks/word_embedding.py` - Word embedding decoding (WordEmbeddingConfig)
- `tasks/sentence_onset.py` - Binary classification (SentenceOnsetConfig)
- `tasks/content_noncontent.py` - Binary classification (ContentNonContentConfig)
- `tasks/pos_task.py` - Multi-class classification (PosTaskConfig)
- `tasks/gpt_surprise.py` - Regression task (GptSurpriseConfig)
- `tasks/volume_level.py` - Audio feature prediction (VolumeLevelConfig)

For detailed documentation on all available tasks, see the [Task Reference](task-reference.md). For baseline performance benchmarks, see [Baseline Results](baseline-results.md).

---

## See Also

- [Task Reference](task-reference.md) - Complete reference for all available tasks
- [Baseline Results](baseline-results.md) - Performance benchmarks for all tasks
- [Configuration Guide](configuration.md) - How to configure tasks
- [API Reference](api-reference.md) - Task data getter API
- `tasks/` directory - Complete task examples
