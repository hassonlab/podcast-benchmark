# Registry API Reference

Reference for all registry decorators and their function signatures.

## Overview

The framework uses registries to discover and manage model components. Decorate your functions with the appropriate registry decorator to make them available to the training pipeline.

**Module**: `core/registry.py`

---

## `@register_model_constructor(name=None, required_data_getter=None)`

Register a function that constructs your decoding model.

### Purpose
Creates model instances from config parameters. Called during training setup.

### Function Signature
```python
def model_constructor(params: dict) -> nn.Module
```

### Arguments
- `params` (dict): Parameters from your config's `model_spec.params` section

### Decorator Arguments
- `name` (str, optional): Name to register under. Defaults to function name.
- `required_data_getter` (str, optional): Name of a registered `model_data_getter` that this model requires. When specified, the getter is called automatically before training to add model-specific columns to the task DataFrame.

### Returns
- PyTorch model instance

### Example
```python
@registry.register_model_constructor()
def my_model(params):
    return MyModel(
        input_dim=params['input_dim'],
        output_dim=params['output_dim']
    )
```

**With required data getter**:
```python
@registry.register_model_constructor(required_data_getter="diver_data_info")
def diver_model(params):
    return DiverModel(...)
```

### Usage in Config
```yaml
model_spec:
  constructor_name: my_model
  params:
    input_dim: 256
    output_dim: 50
```

---

## `@register_model_data_getter(name=None)`

Register a function that adds model-specific columns to the task DataFrame.

### Purpose
Some models require additional data beyond neural signals and task targets. Model data getters enrich the task DataFrame with model-specific columns that are automatically passed to the model's `forward()` method as keyword arguments.

### Function Signature
```python
def model_data_getter(
    task_df: pd.DataFrame,
    raws: list[mne.io.Raw],
    model_params: dict
) -> tuple[pd.DataFrame, list[str]]
```

### Arguments
- `task_df` (pd.DataFrame): DataFrame from the task data getter
- `raws` (list[mne.io.Raw]): Loaded neural recordings
- `model_params` (dict): Parameters from your config's `model_spec.params`

### Returns
- Tuple of `(enriched_df, added_column_names)` where:
  - `enriched_df`: The DataFrame with new columns added
  - `added_column_names`: List of column names that were added (these are automatically appended to `input_fields`)

### Example
```python
@registry.register_model_data_getter("diver_data_info")
def get_diver_data_info(task_df, raws, model_params):
    task_df["data_info_list"] = compute_data_info(raws, model_params)
    return task_df, ["data_info_list"]
```

The added columns are named to match the model's `forward()` parameter names so they can be passed automatically.

### Linking to a Model Constructor
Use `required_data_getter` on `@register_model_constructor` to automatically invoke a data getter:
```python
@registry.register_model_constructor(required_data_getter="diver_data_info")
def diver_model(params):
    ...
```

Or override in config:
```yaml
model_spec:
  constructor_name: my_model
  model_data_getter: diver_data_info  # Explicit override
```

---

## `@register_data_preprocessor(name=None)`

Register a function that preprocesses neural data.

### Purpose
Transforms raw neural data into the format your model expects. Called once before training.

### Function Signature
```python
def preprocessor(
    data: np.ndarray,  # [num_events, num_electrodes, timesteps]
    preprocessor_params: dict
) -> np.ndarray  # [num_events, ...]
```

### Arguments
- `data` (np.ndarray): Raw neural data with shape `[num_events, num_electrodes, timesteps]`
- `preprocessor_params` (dict): Parameters from your config's `data_params.preprocessor_params`

### Returns
- Preprocessed data with shape `[num_events, ...]` (any shape your model needs)

### Example
```python
@registry.register_data_preprocessor()
def my_preprocessor(data, preprocessor_params):
    # Average over time
    n_avg = preprocessor_params['num_average_samples']
    return data.reshape(data.shape[0], data.shape[1], -1, n_avg).mean(-1)
```

### Usage in Config
```yaml
data_params:
  preprocessing_fn_name: my_preprocessor
  preprocessor_params:
    num_average_samples: 32
```

---

## `@register_config_setter(name=None)`

Register a function that modifies config at runtime based on loaded data.

### Purpose
Sets config values that depend on the data (e.g., number of channels, model dimensions). Called after data is loaded, before model construction.

### Function Signature
```python
def config_setter(
    experiment_config: ExperimentConfig,
    raws: list[mne.io.Raw],
    task_df: pd.DataFrame
) -> ExperimentConfig
```

### Arguments
- `experiment_config` (ExperimentConfig): Your experiment configuration
- `raws` (list[mne.io.Raw]): Loaded neural recordings
- `task_df` (pd.DataFrame): Task data with event timings and targets (columns: `start`, `target`, etc.)

### Returns
- Modified `ExperimentConfig`

### Example
```python
@registry.register_config_setter()
def my_config_setter(experiment_config, raws, task_df):
    # Set input channels based on loaded data
    num_channels = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_spec.params['input_channels'] = num_channels
    return experiment_config
```

### Usage in Config
```yaml
config_setter_name: my_config_setter
```

---

## `@register_metric(name=None)`

Register a metric or loss function.

### Purpose
Defines objectives for training (losses) or evaluation (metrics). Called during each training step.

### Function Signature
```python
def metric(
    predicted: torch.Tensor,
    groundtruth: torch.Tensor
) -> float
```

### Arguments
- `predicted` (torch.Tensor): Model predictions `[batch_size, ...]`
- `groundtruth` (torch.Tensor): Ground truth targets `[batch_size, ...]`

### Returns
- Scalar metric value (float or torch scalar)

### Example
```python
@registry.register_metric()
def my_loss(predicted, groundtruth):
    return F.mse_loss(predicted, groundtruth)
```

### Usage in Config
```yaml
training_params:
  losses: [my_loss, mse]
  loss_weights: [0.5, 0.5]
  metrics: [cosine_sim]
```

---

## `@register_task_data_getter(name=None, config_type=None)`

Register a function that loads task-specific data.

### Purpose
Loads event timings and targets for your decoding task. Called once at the start of training.

### Function Signature
```python
def task_data_getter(task_config: TaskConfig) -> pd.DataFrame
```

### Decorator Arguments
- `name` (str, optional): Name to register under. Defaults to function name.
- `config_type` (type, **required**): The dataclass type for this task's configuration. Must be a subclass of `BaseTaskConfig`.

### Arguments
- `task_config` (TaskConfig): Task configuration containing `task_name`, `data_params`, and `task_specific_config`

### Returns
- DataFrame with required columns:
  - `start` (float): Event onset time in seconds
  - `target` (any): Prediction target (embeddings, labels, etc.)
  - `word` (str, optional): Event label (for zero-shot folds)
  - Any columns listed in `input_fields` (will be passed to model as kwargs)

### Example
```python
from dataclasses import dataclass
from core.config import BaseTaskConfig, TaskConfig

@dataclass
class MyTaskConfig(BaseTaskConfig):
    data_file: str = "processed_data/my_data.csv"

@registry.register_task_data_getter(config_type=MyTaskConfig)
def my_task(task_config: TaskConfig):
    config: MyTaskConfig = task_config.task_specific_config
    data_params = task_config.data_params

    df = pd.read_csv(os.path.join(data_params.data_root, config.data_file))
    df['start'] = df['onset_time']
    df['target'] = df['label'].values

    return df[['start', 'target']]
```

### Usage in Config
```yaml
task_config:
  task_name: my_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
  task_specific_config:
    data_file: processed_data/my_data.csv
```

---

## Built-in Registered Functions

### Models
See `models/neural_conv_decoder/decoder_model.py` and `models/example_foundation_model/integration.py` for examples. Additional foundation model integrations are available in:
- `models/diver/integration.py` - DIVER foundation model
- `models/popt/integration.py` - POPT foundation model
- `models/brainbert/integration.py` - BrainBERT foundation model

### Preprocessors
- `window_average_neural_data` - Temporal averaging (models/neural_conv_decoder)
- `foundation_model_preprocessing_fn` - Extract frozen foundation model features
- `foundation_model_finetune_mlp` - Prepare data for foundation model finetuning

### Metrics

The metrics package is organized by task type:

**Regression Metrics** (`metrics/regression_metrics.py`):
- `mse` - Mean squared error
- `corr` - Pearson correlation coefficient
- `r2` - R² score (coefficient of determination)

**Embedding Metrics** (`metrics/embedding_metrics.py`):
- `cosine_sim` - Cosine similarity
- `cosine_dist` - Cosine distance
- `nll_embedding` - Contrastive NLL
- `similarity_entropy` - Similarity distribution entropy

**Classification Metrics** (`metrics/classification_metrics.py`):
- `bce` - Binary cross-entropy (weighted, expects probabilities)
- `bce_with_logits` - Binary cross-entropy with logits (expects raw logits)
- `cross_entropy` - Multi-class cross-entropy (supports sequence prediction and -100 ignore index)
- `weighted_cross_entropy` - Weighted cross-entropy with automatic class balancing
- `roc_auc` - ROC-AUC for binary classification
- `roc_auc_multiclass` - ROC-AUC for multi-class classification
- `f1` - F1 score (binary and multiclass)
- `acc` - Accuracy (binary and multiclass)
- `sensitivity` - Sensitivity (recall/TPR)
- `precision` - Precision
- `specificity` - Specificity (TNR)
- `confusion_matrix` - Confusion matrix

**Utility Functions** (`metrics/utils.py`):
- `compute_cosine_distances` - Cosine distance computation with ensemble support
- `compute_class_scores` - Convert distances to class probabilities
- `calculate_auc_roc` - AUC-ROC with frequency filtering
- `top_k_accuracy` - Top-k accuracy calculation
- `entropy` - Entropy computation for distributions

See the `metrics/` package for complete implementations.

### Tasks
- `word_embedding_decoding_task` - Decode word embeddings (default)
- `content_noncontent_task` - Content vs non-content classification
- `gpt_surprise_task` - GPT surprisal prediction
- `gpt_surprise_multiclass_task` - GPT surprisal multiclass classification
- `pos_task` - Part-of-speech tagging
- `sentence_onset_task` - Sentence onset detection
- `volume_level_decoding_task` - Audio volume level prediction
- `llm_decoding_task` - LLM-based brain-to-text generation
- `llm_embedding_pretraining_task` - Pre-train encoder for LLM decoding

See `tasks/` directory for implementations.

---

## See Also

- [Onboarding a Model](onboarding-model.md) - How to use registries
- [Adding a Task](adding-task.md) - Task data getter details
- [Configuration Guide](configuration.md) - Config structure
