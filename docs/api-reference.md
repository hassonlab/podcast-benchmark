# Registry API Reference

Reference for all registry decorators and their function signatures.

## Overview

The framework uses registries to discover and manage model components. Decorate your functions with the appropriate registry decorator to make them available to the training pipeline.

**Module**: `core/registry.py`

---

## `@register_model_constructor(name=None)`

Register a function that constructs your decoding model.

### Purpose
Creates model instances from config parameters. Called during training setup.

### Function Signature
```python
def model_constructor(model_params: dict) -> nn.Module
```

### Arguments
- `model_params` (dict): Parameters from your config's `model_params` section

### Returns
- PyTorch model instance

### Example
```python
@registry.register_model_constructor()
def my_model(model_params):
    return MyModel(
        input_dim=model_params['input_dim'],
        output_dim=model_params['output_dim']
    )
```

### Usage in Config
```yaml
model_constructor_name: my_model
model_params:
  input_dim: 256
  output_dim: 50
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
    df_word: pd.DataFrame
) -> ExperimentConfig
```

### Arguments
- `experiment_config` (ExperimentConfig): Your experiment configuration
- `raws` (list[mne.io.Raw]): Loaded neural recordings
- `df_word` (pd.DataFrame): Task data with event timings and targets

### Returns
- Modified `ExperimentConfig`

### Example
```python
@registry.register_config_setter()
def my_config_setter(experiment_config, raws, df_word):
    # Set input channels based on loaded data
    num_channels = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_params['input_channels'] = num_channels
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

## `@register_task_data_getter(name=None)`

Register a function that loads task-specific data.

### Purpose
Loads event timings and targets for your decoding task. Called once at the start of training.

### Function Signature
```python
def task_data_getter(data_params: DataParams) -> pd.DataFrame
```

### Arguments
- `data_params` (DataParams): Data configuration from your config file

### Returns
- DataFrame with required columns:
  - `start` (float): Event onset time in seconds
  - `target` (any): Prediction target (embeddings, labels, etc.)
  - `word` (str, optional): Event label (for zero-shot folds)

### Example
```python
@registry.register_task_data_getter()
def my_task(data_params):
    # Load timing data
    df = pd.read_csv(data_params.task_params['data_file'])

    # Create required columns
    df['start'] = df['onset_time']
    df['target'] = df['label'].values

    return df[['start', 'target']]
```

### Usage in Config
```yaml
task_name: my_task
data_params:
  task_params:
    data_file: path/to/data.csv
```

---

## Built-in Registered Functions

### Models
See `models/neural_conv_decoder/decoder_model.py` and `models/example_foundation_model/integration.py` for examples.

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
- `bce` - Binary cross-entropy (weighted)
- `cross_entropy` - Multi-class cross-entropy
- `roc_auc` - ROC-AUC for binary classification
- `roc_auc_multiclass` - ROC-AUC for multi-class classification
- `f1` - F1 score
- `sensitivity` - Sensitivity (recall/TPR)
- `precision` - Precision
- `specificity` - Specificity (TNR)
- `confusion_matrix` - Confusion matrix
- `perplexity` - Perplexity (for LLM evaluation)

**Utility Functions** (`metrics/utils.py`):
- `compute_cosine_distances` - Cosine distance computation with ensemble support
- `compute_class_scores` - Convert distances to class probabilities
- `calculate_auc_roc` - AUC-ROC with frequency filtering
- `top_k_accuracy` - Top-k accuracy calculation
- `entropy` - Entropy computation for distributions

See the `metrics/` package for complete implementations.

### Tasks
- `word_embedding_decoding_task` - Decode word embeddings (default)
- `placeholder_task` - Minimal example
- `content_noncontent_task` - Content vs non-content classification
- `gpt_surprise_task` - GPT surprisal prediction
- `gpt_surprise_multiclass_task` - GPT surprisal multiclass classification
- `pos_task` - Part-of-speech tagging
- `sentence_onset_task` - Sentence onset detection
- `volume_level_encoding_task` - Audio volume level prediction

See `tasks/` directory for implementations.

---

## See Also

- [Onboarding a Model](onboarding-model.md) - How to use registries
- [Adding a Task](adding-task.md) - Task data getter details
- [Configuration Guide](configuration.md) - Config structure
