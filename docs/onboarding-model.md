# Onboarding a New Model

Complete guide to adding your own decoding model to the framework.

## Quick Reference

To set up a new model (e.g., BrainBERT), you need to:

1. [Create a new folder for your model code](#1-create-a-new-folder)
2. [Define a decoding model and constructor function](#2-define-decoding-model-and-constructor)
3. [Define a data preprocessing function](#3-define-data-preprocessing-function)
4. [Create a config file](#4-create-config-file)
5. [Optional: Define a config setter function](#5-optional-define-config-setter)
6. [Import your module in main.py](#6-import-module)
7. [Optional: Update the Makefile](#7-optional-update-makefile)
8. [Run your training code](#8-run-training)

---

## 1. Create a New Folder

Organize all code for your model in its own directory inside the `models/` folder:

```bash
mkdir models/my_model
```

Write all model-specific code in this folder.

---

## 2. Define Decoding Model and Constructor

### Define Your Model

Create your PyTorch model in `models/my_model/model.py`. For example:

```python
import torch.nn as nn

class MyDecodingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### Create a Constructor Function

Define a constructor that takes a params dict (which may include sub-models) from your config:

```python
import core.registry as registry

@registry.register_model_constructor()
def my_model_constructor(params):
    return MyDecodingModel(
        input_dim=params['input_dim'],
        output_dim=params['output_dim']
    )
```

**Important**:
- Use the `@registry.register_model_constructor()` decorator
- The function must have signature: `constructor_fn(params: dict) -> Model`
- The `params` dict contains both regular parameters and any built sub-models
- By default, the registered name is the function name (can override with `@registry.register_model_constructor('custom_name')`)

### Examples

**Neural Conv Decoder** (ensemble model):
```python
@registry.register_model_constructor()
def ensemble_pitom_model(params):
    return EnsemblePitomModel(
        num_models=params['num_models'],
        input_channels=params['input_channels'],
        output_dim=params['embedding_dim'],
        conv_filters=params['conv_filters'],
        dropout=params['dropout']
    )
```

**Model with Nested Sub-Model** (e.g., GPT2Brain with encoder):
```python
@registry.register_model_constructor()
def gpt2_brain(params):
    # params contains both regular params and built sub-models
    return GPT2Brain(
        lm_model=params['lm_model'],
        tokenizer=params['tokenizer'],
        encoder_model=params['encoder_model'],  # This is a pre-built model
        device=params.get('device', 'cpu'),
        freeze_lm=params.get('freeze_lm', True)
    )
```

**Foundation Model with Finetuning**:

When finetuning a foundation model, you include it as part of your decoder class:

```python
class FoundationModelMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_layer_sizes,
        model_dir=None,
        finetune=False,
        foundation_model_config=None,
        freeze_foundation_model=False,
        num_unfrozen_blocks=0,
    ):
        super().__init__()
        self.finetune = finetune

        # Include foundation model as part of decoder if finetuning
        if finetune:
            self.foundation_model = create_and_freeze_foundation_model(
                foundation_model_config,
                model_dir,
                freeze_foundation_model,
                num_unfrozen_blocks,
            )

        self.embedding_norm = nn.BatchNorm1d(input_dim)
        self.mlp = MLP(input_dim, mlp_layer_sizes)

    def forward(self, x):
        # Pass through foundation model if finetuning
        if self.finetune:
            x = self.foundation_model(x, forward_features=True)

        x = self.embedding_norm(x)
        return self.mlp(x)


@registry.register_model_constructor()
def foundation_model_finetune_mlp(params):
    return FoundationModelMLP(
        params["model_dim"],
        params["mlp_layer_sizes"],
        model_dir=params.get("model_dir"),
        foundation_model_config=params["foundation_model_config"],
        finetune=True,
        freeze_foundation_model=params.get("freeze_foundation_model", False),
        num_unfrozen_blocks=params.get("num_unfrozen_blocks", 0),
    )
```

**Key Points for Finetuning**:
- Your decoder model includes the foundation model as a submodule
- The foundation model is loaded with pretrained weights in `__init__`
- You can optionally freeze parts of the foundation model
- The `forward()` method runs data through both the foundation model and your decoder head

---

## 3. Define Data Preprocessing Function

Create a function to transform neural data for your model.

```python
import core.registry as registry

@registry.register_data_preprocessor()
def my_preprocessing_fn(data, preprocessor_params):
    # data shape: [num_words, num_electrodes, timesteps]
    # Return shape: [num_words, ...] (any shape your model expects)

    # Example: average over time
    return data.mean(axis=-1)
```

**Function Signature**:
```python
import numpy as np

def preprocessing_fn(
    data: np.array,  # [num_words, num_electrodes, timesteps]
    preprocessor_params: dict
) -> np.array  # [num_words, ...]
```

### Examples

**Neural Conv Decoder** (temporal averaging):
```python
@registry.register_data_preprocessor()
def window_average_neural_data(data, preprocessor_params):
    # Average over num_average_samples to reduce sample rate
    return data.reshape(
        data.shape[0],
        data.shape[1],
        -1,
        preprocessor_params['num_average_samples']
    ).mean(-1)
```

**Foundation Model with Finetuning** (prepare for model input):

When finetuning, your preprocessing function prepares the data in the format your foundation model expects:

```python
@registry.register_data_preprocessor("foundation_model_finetune_mlp")
def prepare_data_for_finetuning(data, preprocessor_params):
    """Prepare neural data for foundation model input."""
    data_config = preprocessor_params["ecog_data_config"]

    # Downsample temporal resolution
    data = data.reshape(
        data.shape[0],
        data.shape[1],
        -1,
        data_config.original_fs // data_config.new_fs
    )
    data = data.mean(-1)

    # Pad to expected electrode grid (e.g., 64 channels)
    for i in range(64):
        channel = "G" + str(i + 1)
        if channel not in preprocessor_params['ch_names']:
            # Insert NaN for missing channels
            data = np.insert(data, i, np.nan, axis=1)

    # Reshape to spatial grid: [num_examples, bands, time, height, width]
    data = np.einsum('bet->bte', data).reshape(data.shape[0], data.shape[2], 8, 8)
    data = np.expand_dims(data, axis=1)

    return data
```

**Key Points**:
- When **not finetuning**: Extract frozen representations in preprocessing, return embeddings
- When **finetuning**: Format raw data for model input, let the model extract features during training

---

## 4. Create Config File

Create a YAML config file in `configs/my_model/config.yml`.

See [Configuration Guide](configuration.md) for detailed documentation on all config options.

### Basic Example

```yaml
# Model specification
model_spec:
  constructor_name: my_model_constructor
  params:
    input_dim: 256
    output_dim: 50
  sub_models: {}

# Optional: config setter function name
config_setter_name: my_config_setter

# Task configuration
task_config:
  task_name: word_embedding_decoding_task
  data_params:
    data_root: data
    window_width: 0.625
    preprocessing_fn_name: my_preprocessing_fn
    subject_ids: [1, 2, 3]
    preprocessor_params:
      custom_param: value
  task_specific_config:
    embedding_type: gpt-2xl
    embedding_layer: 24

# Training parameters
training_params:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  n_folds: 5
  losses: [mse]
  metrics: [cosine_sim]
  early_stopping_metric: cosine_sim

# Trial identifier
trial_name: my_model_v1
```

### Nested Model Example (e.g., GPT2Brain with encoder)

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
      checkpoint_path: "checkpoints/encoder/lag_{lag}/best_model_fold{fold}.pt"

config_setter_name: set_input_channels

# This allows training different encoders at each lag
# while reusing the same parent GPT2Brain model

task_config:
  task_name: word_embedding_decoding_task
  data_params:
    data_root: data
    preprocessing_fn_name: foundation_model_finetune_mlp
    subject_ids: [1, 2, 3]
    window_width: 0.625
  task_specific_config:
    embedding_type: gpt-2xl
    embedding_layer: 24
    embedding_pca_dim: 50

training_params:
  batch_size: 64
  learning_rate: 0.001
  losses: [mse]
  metrics: [cosine_sim, nll_embedding]
  early_stopping_metric: cosine_sim

trial_name: foundation_finetune_v1
```

---

## 5. Optional: Define Config Setter

Sometimes you need to set config values at runtime based on the loaded data.

```python
import core.registry as registry

@registry.register_config_setter('my_model')
def my_config_setter(experiment_config, raws, df_word):
    # Set values based on data
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_spec.params['input_channels'] = num_electrodes
    return experiment_config
```

**Function Signature**:
```python
from core.config import ExperimentConfig

def config_setter(
    experiment_config: ExperimentConfig,
    raws: list[mne.io.Raw],
    df_word: pd.DataFrame
) -> ExperimentConfig
```

**Multiple Config Setters**: You can apply multiple setters in sequence:
```yaml
# Single setter
config_setter_name: my_model

# Multiple setters (applied in order)
config_setter_name: [set_input_channels, set_embedding_dim, initialize_model]
```

This is useful for:
- Applying task-specific setters from `task_specific_config.required_config_setter_names`
- Following up with model-specific setters
- Chaining multiple config transformations

### Examples

**Neural Conv** (set number of input channels):
```python
@registry.register_config_setter('neural_conv')
def set_config_input_channels(experiment_config, raws, _df_word):
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_spec.params['input_channels'] = num_electrodes
    return experiment_config
```

**Foundation Model Finetuning** (load foundation config and set dimensions):
```python
@registry.register_config_setter("foundation_model_finetune_mlp")
def foundation_model_mlp_finetune_config_setter(
    experiment_config, raws, _df_word
):
    # Add channel names for preprocessing
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    experiment_config.task_config.data_params.preprocessor_params = {"ch_names": ch_names}

    # Load foundation model config
    config_path = os.path.join(
        experiment_config.model_spec.params["model_dir"],
        "experiment_config.yml"
    )
    foundation_config = load_config(config_path)

    # Set dimensions and window width from foundation model
    experiment_config.model_spec.params["foundation_model_config"] = foundation_config
    experiment_config.model_spec.params["model_dim"] = foundation_config.vit_config.dim
    experiment_config.task_config.data_params.window_width = foundation_config.sample_length
    experiment_config.task_config.data_params.preprocessor_params["ecog_data_config"] = (
        foundation_config.ecog_data_config
    )

    return experiment_config
```

---

## 6. Import Module

Your module will be automatically imported! The framework recursively imports all models from the `models/` directory:

```python
# Import all models from the models/ directory (recursively imports all subpackages)
import_all_from_package("models", recursive=True)
```

As long as your model is in `models/my_model/`, it will be automatically discovered and loaded at runtime.

**Critical**: Make sure you've added the `@registry` decorators to your functions!

---

## 7. Optional: Update Makefile

Add a convenient make rule for your model:

```makefile
my-model:
	mkdir -p logs
	$(CMD) main.py \
		--config configs/my_model/config.yml
```

Now you can run with:
```bash
make my-model
```

---

## 8. Run Training

Run your model:

```bash
make my-model
```

Or directly:
```bash
python main.py --config configs/my_model/config.yml
```

Results will be saved to:
- `results/` - Performance metrics
- `checkpoints/` - Model checkpoints
- `event_logs/` - TensorBoard logs (if enabled)

### Debugging

If you encounter errors:

1. Check that all `@registry` decorators are present
2. Verify your module is imported in `main.py`
3. Ensure function names match between config and registered functions
4. Look at logs in `logs/` for SLURM jobs

---

---

## Complete Working Example

See **`models/example_foundation_model/`** for a complete, self-contained example demonstrating:

- Simple transformer foundation model implementation
- Both integration patterns (feature extraction + finetuning)
- Model directory structure with config and checkpoint
- Full documentation and runnable examples

This example shows exactly how all the pieces fit together for foundation models.

```bash
# Run feature extraction example
python main.py --config configs/example_foundation_model/feature_extraction.yaml

# Run finetuning example
python main.py --config configs/example_foundation_model/finetuning.yaml
```

See `models/example_foundation_model/README.md` for details.

---

## See Also

- [Configuration Guide](configuration.md) - Detailed config options and patterns
- [Task Reference](task-reference.md) - Complete reference for all available tasks
- [Adding a Task](adding-task.md) - Create custom decoding tasks
- [API Reference](api-reference.md) - Complete API documentation
