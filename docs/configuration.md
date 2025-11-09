# Configuration Guide

Complete guide to configuring experiments in the podcast benchmark framework.

## Overview

All experiments are configured via YAML files in the `configs/` directory. Each config file has four main sections that control different aspects of your experiment.

---

## Configuration Structure

```yaml
# Which model and task to use
model_constructor_name: my_model
config_setter_name: my_config_setter  # Optional
task_name: word_embedding_decoding_task  # Optional

# Model-specific parameters
model_params:
  # Fully customizable - passed to your model constructor

# How to train
training_params:
  # Batch size, learning rate, losses, metrics, etc.

# What data to use
data_params:
  # Subjects, electrodes, embeddings, preprocessing

# Where to save results
output_dir: results
model_dir: models
trial_name: my_experiment
```

---

## Model Parameters

**Purpose**: Define architecture and hyperparameters for your specific model.

This section is **completely customizable** - whatever you put here gets passed directly to your model constructor function. The framework doesn't impose any specific fields.

**Example**:
```yaml
model_params:
  hidden_dim: 512
  num_layers: 3
  dropout: 0.2
  # Anything your model needs
```

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

## Data Parameters

**Purpose**: Specify which subjects, electrodes, and data to use.

**Key concepts**:
- **Subject selection**: Which participants to include
- **Electrode selection**: Which brain regions to decode from
- **Embeddings**: What target representations to predict
- **Preprocessing**: How to transform raw neural data

**Particularly useful fields**:

### Electrode Selection

You have three options for choosing which electrodes to use:

```yaml
data_params:
  # Option 1: Regular expression (most flexible)
  channel_reg_ex: "LG[AB]*"  # Channels starting with LGA or LGB
  # Examples:
  #   "^G([1-9]|[1-5][0-9]|6[0-4])$"  - Channels G1-G64
  #   ".*"  - All channels

  # Option 2: CSV file with electrode list
  electrode_file_path: configs/significant_electrodes.csv
  # Format: subject,elec

  # Option 3: Per-subject dictionary
  per_subject_electrodes:
    1: [LGA1, LGA2, LGA3]
    2: [LGB1, LGB2]
```

### Other Key Fields

```yaml
data_params:
  subject_ids: [1, 2, 3]        # Which subjects to include

  # Target embeddings
  embedding_type: gpt-2xl       # "gpt-2xl", "glove", or "arbitrary"
  embedding_layer: 24           # Which layer to extract (for GPT-2)
  embedding_pca_dim: 50         # Optional dimensionality reduction

  # Neural data window
  window_width: 0.625           # Width of data window in seconds

  # Preprocessing
  preprocessing_fn_name: my_preprocessor  # Your registered function
  preprocessor_params:          # Custom params for your preprocessor
    param1: value1

  # Task configuration
  task_name: my_custom_task     # Optional, defaults to word embeddings
  word_column: lemmatized_word  # For zero-shot folds
  task_params:                  # Task-specific parameters
    param1: value1
```

See `core/config.py:DataParams` for all available fields.

---

## Output Configuration

**Purpose**: Name your experiment and specify where results are saved.

```yaml
# Dynamic trial naming with formatting
trial_name: "model_{}_lr={}_bs={}"
format_fields:
  - model_params.model_name
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
