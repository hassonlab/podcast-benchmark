from dataclasses import is_dataclass, dataclass, fields, field
from typing import Optional

# These classes exist to document all of the model-agnostic fields for data collection and model training.

@dataclass
class DataParams:
    # The name of the embeddings to use. Currently supports gpt-2xl, glove, and arbitrary.
    embedding_type: str = 'gpt-2xl'
    # The width of neural data to gather around each word onset in seconds.
    window_width: float = -1
    # The name of your model's preprocessing function. Your function msut be registered using the register_data_preprocessor
    # decorator and imported into main.py. See registry.py for details.
    preprocessing_fn_name: Optional[str] = None
    # The subject id's to include in your analysis. For the podcast data they must all be in the range [1, 9]
    subject_ids: list[int] = field(default_factory=lambda: [])
    # Root of data folder.
    data_root: str = 'data'
    # Number of embeddings to reduce the embeddings to using pca. If None, don't run PCA.
    embedding_pca_dim: Optional[int] = None
    # A regular expression to pick which channels you are interested in.
    # (i.e. "LG[AB]*" will select channels that start with "LGA" or "LGB")
    channel_reg_ex: Optional[str] = None
    # Layer of model to gather embeddings from. Required if using gpt2-xl.
    embedding_layer: Optional[int] = None
    # A user defined configuration for their specific models preprocessor function.
    preprocessor_params: Optional[dict] = None


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
    # Models will be trained over all lags in range(min_lag, max_lag, lag_step_size). In ms.
    min_lag: int = -10_000
    max_lag: int = 10_000
    lag_step_size: int = 1000


@dataclass
class ExperimentConfig:
    # Model constructor function name. Must be registered using @registry.register_model_constructor()
    model_constructor_name: str = ''
    # Config setter function name. Must be registered using @registry.register_config_setter()
    config_setter_name: Optional[str] = None
    # Parameters for this model. Can be any user-defined dictionary.
    model_params: dict = field(default_factory=lambda: {})
    # Parameters for training.
    training_params: TrainingParams = field(default_factory=lambda: TrainingParams())
    # Parameters for data loading and preprocessing. Sub-field preprocessor_params can be set for your use-case.
    data_params: DataParams = field(default_factory=lambda: DataParams())
    # Name for trial. Will be used for separating results in storage.
    trial_name: str = ''
    # Base directory to output results to.
    output_dir: str = 'results'
    # Base directory to write models to.
    model_dir: str = 'models'


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

