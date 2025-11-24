# Configuration Guide

Complete guide to configuring experiments in the podcast benchmark framework.

## Overview

All experiments are configured via YAML files in the `configs/` directory. Each config file has four main sections that control different aspects of your experiment.

---

## Configuration Structure

```yaml
# Model specification (supports nested sub-models)
model_spec:
  constructor_name: my_model
  params:
    # Model-specific parameters (passed to constructor)
  sub_models:
    # Optional: nested models passed as constructor arguments
    # encoder_model: {...}

config_setter_name: my_config_setter  # Optional

# Task configuration (nested structure)
task_config:
  task_name: word_embedding_decoding_task
  data_params:
    # Shared data parameters (subjects, electrodes, window size, etc.)
  task_specific_config:
    # Task-specific parameters (type-safe, defined per task)

# How to train
training_params:
  # Batch size, learning rate, losses, metrics, etc.

# Where to save results
output_dir: results
model_dir: models
trial_name: my_experiment
```

---

## Model Specification

**Purpose**: Define your model architecture with support for nested sub-models.

The `model_spec` section specifies how to build your model:
- **constructor_name**: Registered model constructor function name
- **params**: Parameters passed to the constructor (fully customizable)
- **sub_models**: Dictionary of nested models to build and pass as constructor arguments

**Simple Example**:
```yaml
model_spec:
  constructor_name: pitom_model
  params:
    input_channels: 64
    output_dim: 768
    conv_filters: 128
    dropout: 0.2
  sub_models: {}
```

**Nested Model Example** (e.g., GPT2Brain with encoder):
```yaml
model_spec:
  constructor_name: gpt2_brain
  params:
    freeze_lm: true
    device: cuda
  sub_models:
    encoder_model:
      constructor_name: pitom_model
      params:
        input_channels: 64
        output_dim: 768
        conv_filters: 128
        dropout: 0.2
      sub_models: {}
```

In this example:
1. The `pitom_model` encoder is built first with the specified params
2. The built encoder is then passed to `gpt2_brain` as the `encoder_model` argument
3. This allows you to train different encoders at each lag while using the same parent model

---

## Training Parameters

**Purpose**: Control the training loop, optimization, and evaluation.

**Key concepts**:
- **Losses and metrics**: What to optimize and track
- **Cross-validation**: How to split data into folds
- **Early stopping**: When to stop training
- **Time lags**: Temporal relationship between neural data and events

**Particularly useful fields**:

```yaml
training_params:
  # Loss configuration
  losses: [mse, cosine_dist]    # Can combine multiple losses
  loss_weights: [0.7, 0.3]      # Weight for each loss
  metrics: [cosine_sim]         # Additional metrics to track (not in loss)

  # Early stopping
  early_stopping_metric: cosine_sim  # What metric to monitor
  smaller_is_better: false          # false for accuracy/similarity, true for error

  # Cross-validation strategy
  fold_type: sequential_folds   # or "zero_shot_folds" for words not in training
  n_folds: 5

  # Time lags - find optimal temporal offset
  min_lag: -500      # Start 500ms before word onset
  max_lag: 1000      # End 1000ms after word onset
  lag_step_size: 100 # Test every 100ms

  # Baseline models
  linear_regression_baseline: false    # Train and evaluate linear regression baseline
  logistic_regression_baseline: false  # Train and evaluate logistic regression baseline
```

See `core/config.py:TrainingParams` for all available fields.

---

## Task Configuration

**Purpose**: Specify the task, data parameters, and task-specific settings.

The `task_config` section has three parts:
1. **task_name**: Which task to run (e.g., `word_embedding_decoding_task`)
2. **data_params**: Shared parameters (subjects, electrodes, preprocessing)
3. **task_specific_config**: Task-specific parameters (type-safe, validated)

**Example structure**:

```yaml
task_config:
  task_name: sentence_onset_task
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625
    preprocessing_fn_name: window_average_neural_data
  task_specific_config:
    sentence_csv_path: processed_data/sentences.csv
    negatives_per_positive: 5
    negative_margin_s: 0.75
```

### Data Parameters

**Shared fields used across all tasks**:

#### Electrode Selection

```yaml
task_config:
  data_params:
    # Option 1: Regular expression
    channel_reg_ex: "LG[AB]*"

    # Option 2: CSV file
    electrode_file_path: configs/significant_electrodes.csv

    # Option 3: Per-subject dictionary
    per_subject_electrodes:
      1: [LGA1, LGA2, LGA3]
      2: [LGB1, LGB2]
```

#### Other Key Fields

```yaml
task_config:
  data_params:
    data_root: data
    subject_ids: [1, 2, 3]
    window_width: 0.625           # Width of neural data window (seconds)
    word_column: lemmatized_word  # For zero-shot folds

    # Preprocessing
    preprocessing_fn_name: my_preprocessor
    preprocessor_params:
      param1: value1
```

### Task-Specific Config

Each task defines its own config dataclass with type-safe parameters. See [Task Reference](task-reference.md) for details on each task's configuration options.

**Example for word_embedding_decoding_task**:
```yaml
task_config:
  task_name: word_embedding_decoding_task
  task_specific_config:
    embedding_type: gpt-2xl
    embedding_layer: 24
    embedding_pca_dim: 50
```

**Example for sentence_onset_task**:
```yaml
task_config:
  task_name: sentence_onset_task
  task_specific_config:
    sentence_csv_path: processed_data/sentences.csv
    negatives_per_positive: 5
    negative_margin_s: 0.75
```

---

## Output Configuration

**Purpose**: Name your experiment and specify where results are saved.

```yaml
# Dynamic trial naming with formatting
trial_name: "model_{}_lr={}_bs={}"
format_fields:
  - model_spec.params.model_name
  - training_params.learning_rate
  - training_params.batch_size

# Output directories
output_dir: results           # CSV files with metrics
model_dir: models            # Saved model checkpoints
tensorboard_dir: event_logs  # TensorBoard logs
```

---

## Common Patterns

### Pattern 1: Selecting Significant Electrodes Only

```yaml
data_params:
  electrode_file_path: configs/significant_electrodes.csv
```

### Pattern 2: Multi-Loss Training

```yaml
training_params:
  losses: [mse, cosine_dist]
  loss_weights: [0.7, 0.3]
  early_stopping_metric: cosine_sim  # Can use a metric not in losses
```

### Pattern 3: Finding Optimal Time Lag

```yaml
training_params:
  min_lag: -1000    # 1 second before word
  max_lag: 2000     # 2 seconds after word
  lag_step_size: 100

# Results saved to {output_dir}/lag_performance.csv
```

### Pattern 4: Zero-Shot Evaluation

Test on words never seen during training:

```yaml
training_params:
  fold_type: zero_shot_folds

data_params:
  word_column: lemmatized_word
```

### Pattern 5: Quick Debugging Run

```yaml
data_params:
  subject_ids: [1]  # Single subject

training_params:
  epochs: 10
  n_folds: 2
  lag: 0  # Single lag instead of sweep
```

---

## Batch Training with Training Matrix

The `training_matrix.yaml` file enables running multiple experiments at once. Define model/task/config combinations:

```yaml
neural_conv_decoder:
  word_embedding_decoding_task:
    - neural_conv_decoder_base.yml
    - neural_conv_decoder_binary.yml
```

**Usage**:
```bash
make train-all                                    # Run all configs
make train-all MODELS=neural_conv_decoder         # Filter by model
make train-all TASKS=sentence_onset_task          # Filter by task
make train-all MODELS=model1,model2 TASKS=task1   # Combine filters
```

**Adding New Model/Task Combinations**:

Edit `training_matrix.yaml` to add your experiments:

```yaml
your_new_model:
  your_new_task:
    - config_file_1.yml
    - config_file_2.yml
    - config_file_3.yml
```

---

## See Also

- **`core/config.py`**: Source code with all available fields and defaults
- **`training_matrix.yaml`**: Batch experiment configuration
- [Task Reference](task-reference.md): Complete reference for all available tasks
- [Onboarding a Model](onboarding-model.md): How to use configs with your models
- [API Reference](api-reference.md): Detailed API documentation
