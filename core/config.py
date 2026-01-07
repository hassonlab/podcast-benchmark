from dataclasses import is_dataclass, dataclass, fields, field
from typing import Optional, Union, Dict, Any
from abc import ABC

# These classes exist to document all of the model-agnostic fields for data collection and model training.


# ============================================================================
# Task-Specific Configuration Base Class
# ============================================================================


@dataclass
class BaseTaskConfig(ABC):
    """Base class for all task-specific configurations.

    Each task should define its own config dataclass in its task file that inherits from this.
    """

    # An optional list of inputs which will be passed into the decoder model as additional inputs as
    # kwargs. All fields must be present in the DataFrame returned by the task data getter.
    input_fields: Optional[list[str]] = None
    # An optional list of config setter names which are specific to your task.
    # These will be applied in order before any file configured config setters.
    required_config_setter_names: Optional[list[str]] = None


@dataclass
class DataParams:
    # The width of neural data to gather around each word onset in seconds.
    window_width: float = -1
    # The name of your model's preprocessing function. Your function must be registered using the register_data_preprocessor
    # decorator and imported into main.py. See registry.py for details. Can provide either a single value or a list of names to apply in order.
    # Should align with desired parameters in preprocessor_params.
    preprocessing_fn_name: Optional[str | list[str]] = None
    # The subject id's to include in your analysis. For the podcast data they must all be in the range [1, 9]
    subject_ids: list[int] = field(default_factory=lambda: [])
    # Root of data folder.
    data_root: str = "data"
    # CSV file with columns subject (subject integer id) and elec (string name of electrode).
    # If not set then defaults to configured subject_ids and channel_reg_ex.
    electrode_file_path: Optional[str] = None
    # TODO: Transition to making this dictionary the only field that needs to be set for specifying electrodes.
    # Per-subject electrode list, should be a dictionary from subject id's to a list of electrode names.
    # Optional and can instead use channel_reg_ex if not set.
    per_subject_electrodes: Optional[dict[int, list[str]]] = None
    # A regular expression to pick which channels you are interested in.
    # (i.e. "LG[AB]*" will select channels that start with "LGA" or "LGB")
    channel_reg_ex: Optional[str] = None
    # A user defined configuration for their specific models preprocessor function.  Can provide either a single value or a
    # list of parameters to apply in order. Should align with desired function in preprocessing_fn_name.
    preprocessor_params: Optional[dict | list[dict]] = None
    # The name of the column in the DataFrame returned by the task data getter that specifies the word for each
    # example. Optional if your task does not involve words.
    word_column: Optional[str] = None


@dataclass
class TrainingParams:
    # The batch size to train our decoder with.
    batch_size: int = 32
    # The maximum number of epochs to train over each fold with.
    epochs: int = 100
    # The learning rate to use when training.
    learning_rate: float = 0.001
    # The amount of weight decay to use as regularization in our optimizer.
    weight_decay: float = 0.0001
    # Learning rate scheduler 
    use_lr_scheduler: bool = False
    scheduler_params: Optional[dict] = None
    # If cosine similarity between our predicted embeddings and the actual embeddings do not improve after this many steps
    # stop training for this fold early.
    early_stopping_patience: int = 10
    # Number of folds to train over per-lag.
    n_folds: int = 5
    # If lag is specified then will train over just this lag relative to word onset. In ms.
    lag: Optional[int] = None
    # Otherwise models will be trained over all lags in range(min_lag, max_lag, lag_step_size). In ms.
    min_lag: int = -10_000
    max_lag: int = 10_000
    lag_step_size: int = 1000
    # Type of fold generation to use. One of "sequential_folds" or "zero_shot_folds"
    fold_type: str = "sequential_folds"
    # For backwards compatability if you want to provide just one loss.
    loss_name: Optional[str] = None
    # Losses to use, by default use mse. Should be decorated with @register_metric.
    losses: list[str] = field(default_factory=lambda: ["mse"])
    # Weight to assign to each loss. Should be parallel array with losses.
    loss_weights: list[float] = field(default_factory=lambda: [1.0])
    # Metrics to track during training. Should be decorated with @register_metric.
    metrics: list[str] = field(default_factory=lambda: ["nll_embedding", "cosine_sim"])
    # Metric to use for early stopping over validation set. Must be either the loss or in metrics.
    early_stopping_metric: str = "cosine_sim"
    # Whether or not a smaller value is better for early_stopping_metric. Should be False for metrics you
    # want to increase (i.e. cosine similarity) but True for ones you want to decrease (i.e. MSE).
    smaller_is_better: bool = False
    # Number of gradient accumulation steps.
    grad_accumulation_steps: int = 1
    # TODO: Generalize parameters to metrics based on config. So we don't need to have these last few.
    # Minimum number of occurences of a word in training set to be used for ROC-AUC calculation.
    min_train_freq_auc: int = 5
    # Minimum number of occurences of a word in test set to be used for ROC-AUC calculation.
    min_test_freq_auc: int = -1
    # Sets the k we use in top-k metrics.
    top_k_thresholds: list[int] = field(default_factory=lambda: [1, 5, 10])
    # Random seed to use for training.
    random_seed: int = 42
    # Whether to set cudnn to be deterministic. Will slow down training, but useful for maximizing reproducibility.
    cudnn_deterministic: bool = False
    # Whether to visualize fold class distribution before training.
    visualize_fold_distribution: bool = False
    # If true writes training logs to Tensorboard.
    tensorboard_logging: bool = True
    # If true trains and evaluates a linear regression baseline.
    linear_regression_baseline: bool = False
    # If true trains and evaluates a ridge regression baseline.
    ridge_regression_baseline: bool = False
    # Regularization strength (alpha) for ridge regression baseline.
    ridge_alpha: float = 1.0
    # If true trains and evaluates a logistic regression baseline.
    logistic_regression_baseline: bool = False
    # If true, normalizes targets (Y) to zero mean and unit variance using training set statistics.
    normalize_targets: bool = False
    # If true, shuffles targets to create a sanity check baseline (should break model performance).
    shuffle_targets: bool = False


@dataclass
class TaskConfig:
    """Configuration for a specific task, including data params and task-specific config."""

    task_name: str = "word_embedding_decoding_task"
    data_params: DataParams = field(default_factory=lambda: DataParams())
    task_specific_config: BaseTaskConfig = field(
        default_factory=lambda: BaseTaskConfig()
    )


@dataclass
class ModelSpec:
    """Specification for building a model, including support for nested sub-models.

    This allows for hierarchical model construction where a parent model can receive
    pre-built sub-models as constructor arguments. For example, a GPT2Brain model
    could receive an encoder model built from its own ModelSpec.

    Attributes:
        constructor_name: Name of the registered model constructor function
        params: Dictionary of parameters to pass to the constructor (excluding sub-models)
        sub_models: Dictionary mapping parameter names to ModelSpec objects.
                   The keys indicate the keyword argument names that will receive
                   the built sub-models when constructing the parent model.
        checkpoint_path: Optional path to checkpoint for initialization. Supports
                        dynamic formatting with {lag} and {fold} placeholders.

    Example:
        # Nested encoder model inside GPT2Brain
        ModelSpec(
            constructor_name="gpt2_brain",
            params={"lm_model": "gpt2", "freeze_lm": True},
            sub_models={
                "encoder_model": ModelSpec(
                    constructor_name="pitom_model",
                    params={"input_channels": 64, "output_dim": 768},
                    sub_models={},
                    checkpoint_path="checkpoints/encoder/lag_{lag}/fold_{fold}/best_model.pt"
                )
            }
        )
    """

    constructor_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    sub_models: Dict[str, "ModelSpec"] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    # Model specification with support for nested sub-models
    model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec(constructor_name="")
    )
    # Config setter function name. Must be registered using @registry.register_config_setter(). Can provide a list of names to apply multiple setters in order.
    config_setter_name: Optional[str | list[str]] = None
    # Task configuration including task name, data params, and task-specific config
    # Note: task_specific_config will be set based on the task_name at runtime
    task_config: TaskConfig = field(default_factory=lambda: TaskConfig())
    # Parameters for training.
    training_params: TrainingParams = field(default_factory=lambda: TrainingParams())
    # Name for trial. Will be used for separating results in storage. Can use format strings such as
    # %s, %d, etc and provide which config values you want to fill them in format_fields.
    trial_name: str = ""
    # Path to fields of config to be formatted into trial_name. For example if your trial_name is
    # "decoding_dim={}_lr={:.2f}" you could set format_fields to ["model_spec.params.dim", "training_params.learning_rate"]
    format_fields: Optional[list[str]] = None
    # Base directory to output results to.
    output_dir: str = "results"
    # Base directory to write checkpoints to.
    checkpoint_dir: str = "checkpoints"
    # Base directory to write Tensorboard logs to.
    tensorboard_dir: str = "event_logs"


@dataclass
class MultiTaskConfig:
    """Configuration for running multiple tasks sequentially.

    This allows combining multiple training stages (e.g., pretraining + finetuning)
    into a single config file. Tasks are executed in order, and checkpoint directories
    from previous tasks can be referenced using {prev_checkpoint_dir} placeholder.

    Attributes:
        tasks: List of ExperimentConfig objects to run in sequence
        shared_params: Optional dict mapping parameter paths (e.g., "training_params.n_folds")
                      to values that should override the same field across all tasks.
                      Applied after config setters.

    Example:
        tasks:
          - trial_name: pretrain
            model_spec: {...}
            ...
          - trial_name: finetune
            model_spec:
              sub_models:
                encoder:
                  checkpoint_path: "{prev_checkpoint_dir}/lag_{lag}/best_model_fold{fold}.pt"
            ...
        shared_params:
          training_params.n_folds: 5
          training_params.min_lag: 0
    """

    tasks: list[ExperimentConfig] = field(default_factory=list)
    shared_params: Optional[Dict[str, Any]] = None


def dict_to_config(d: dict, config_class):
    """Recursively convert a dict d to an instance of config_class."""
    import typing

    init_kwargs = {}
    for field_info in fields(config_class):
        field_name = field_info.name
        field_type = field_info.type
        if field_name not in d:
            continue
        field_value = d[field_name]

        # Handle nested dataclasses
        if is_dataclass(field_type) and isinstance(field_value, dict):
            init_kwargs[field_name] = dict_to_config(field_value, field_type)
        # Handle Dict[str, ModelSpec] for sub_models
        elif typing.get_origin(field_type) is dict and isinstance(field_value, dict):
            type_args = typing.get_args(field_type)
            if len(type_args) == 2:
                value_type = type_args[1]
                # Resolve ForwardRef if needed
                if isinstance(value_type, typing.ForwardRef):
                    value_type = ModelSpec
                # Recursively convert dict values to the specified type
                if is_dataclass(value_type):
                    init_kwargs[field_name] = {
                        k: dict_to_config(v, value_type) if isinstance(v, dict) else v
                        for k, v in field_value.items()
                    }
                else:
                    init_kwargs[field_name] = field_value
            else:
                init_kwargs[field_name] = field_value
        else:
            init_kwargs[field_name] = field_value

    return config_class(**init_kwargs)
