from dataclasses import is_dataclass, dataclass, fields, field
from typing import Optional

# These classes exist to document all of the model-agnostic fields for data collection and model training.


@dataclass
class DataParams:
    # The name of the embeddings to use. Currently supports gpt-2xl, glove, and arbitrary.
    embedding_type: str = "gpt-2xl"
    # The width of neural data to gather around each word onset in seconds.
    window_width: float = -1
    # The name of your model's preprocessing function. Your function msut be registered using the register_data_preprocessor
    # decorator and imported into main.py. See registry.py for details.
    preprocessing_fn_name: Optional[str] = None
    # The subject id's to include in your analysis. For the podcast data they must all be in the range [1, 9]
    subject_ids: list[int] = field(default_factory=lambda: [])
    # Root of data folder.
    data_root: str = "data"
    # Number of embeddings to reduce the embeddings to using pca. If None, don't run PCA.
    embedding_pca_dim: Optional[int] = None
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
    # Layer of model to gather embeddings from. Required if using gpt2-xl.
    embedding_layer: Optional[int] = None
    # A user defined configuration for their specific models preprocessor function.
    preprocessor_params: Optional[dict] = None
    # Name of word column in dataframe to use. Optional.
    word_column: Optional[str] = None
    # Dictionary of parameters to pass to your specific task_data_getter if needed.
    task_params: dict = field(default_factory=lambda: {})


@dataclass
class TrainingParams:
    # The batch size to train our decoder with.
    batch_size: int = 32
    # The maximum number of epochs to train over each fold with.
    epochs: int = 100
    # The learning rate to use when training. TODO: currently staic lr, could use a scheduler in the future.
    learning_rate: float = 0.001
    # The amount of weight decay to use as regularization in our optimizer.
    weight_decay: float = 0.0001
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


@dataclass
class ExperimentConfig:
    # Model constructor function name. Must be registered using @registry.register_model_constructor()
    model_constructor_name: str = ""
    # Config setter function name. Must be registered using @registry.register_config_setter()
    config_setter_name: Optional[str] = None
    # Task to run for decoding. Must have a function registered using @registry.register_task_data_getter(). Defaults to decoding word embeddings.
    task_name: str = "word_embedding_decoding_task"
    # Parameters for this model. Can be any user-defined dictionary.
    model_params: dict = field(default_factory=lambda: {})
    # Parameters for training.
    training_params: TrainingParams = field(default_factory=lambda: TrainingParams())
    # Parameters for data loading and preprocessing. Sub-field preprocessor_params can be set for your use-case.
    data_params: DataParams = field(default_factory=lambda: DataParams())
    # Name for trial. Will be used for separating results in storage. Can use format strings such as
    # %s, %d, etc and provide which config values you want to fill them in format_fields.
    trial_name: str = ""
    # Path to fields of config to be formatted into trial_name. For example if your trial_name is
    # "decoding_dim={}_lr={:.2f}" you could set format_fields to ["model_params.dim", "training_params.learning_rate"]
    format_fields: Optional[list[str]] = None
    # Base directory to output results to.
    output_dir: str = "results"
    # Base directory to write models to.
    model_dir: str = "models"
    # Base directory to write Tensorboard logs to.
    tensorboard_dir: str = "event_logs"


def dict_to_config(d: dict, config_class):
    """Recursively convert a dict d to an instance of config_class."""
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
        else:
            init_kwargs[field_name] = field_value

    return config_class(**init_kwargs)
