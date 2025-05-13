# Registry of functions for creating decoding models.
model_constructor_registry = {}
# Registry of functions for preprocessing data.
data_preprocessor_registry = {}
# Registry of functions for altering config values after data outputs.
config_setter_registry = {}
# Registry of functions for metric calculations.
metric_registry = {}


def register_model_constructor(name=None):
    """
    Decorator to register a model constructor function used for decoding models.

    The decorated function must follow the signature:
        model_constructor(model_params: dict) -> Model

    Where:
        - model_params (dict): A user-defined dictionary of model parameters.
        - Model: A user-defined decoding model instance constructed using the given parameters.

    Args:
        name (str, optional): Optional name to register the constructor under.
                              Defaults to the function's __name__.

    Returns:
        function: A decorator that registers the model constructor in
                  model_constructor_registry.
    """

    def decorator(fn):
        model_name = name or fn.__name__
        model_constructor_registry[model_name] = fn
        return fn

    return decorator


def register_data_preprocessor(name=None):
    """
    Decorator to register a data preprocessing function.

    The decorated function must follow the signature:
        data_preprocessor(data: np.ndarray, preprocessor_params) -> np.ndarray

    Where:
        - data: A NumPy array of shape [num_words, num_electrodes, timesteps]
        - preprocessor_params: User-defined parameters for preprocessing
        - Returns: A NumPy array whose first dimension is still `num_words`,
                   but the remaining shape is arbitrary (e.g., features for model input)

    This allows for preprocessing steps like feature extraction, downsampling, or reshaping
    prior to model input.

    Args:
        name (str, optional): Optional name to register the preprocessor under.
                              Defaults to the function's __name__.

    Returns:
        function: A decorator that registers the preprocessor in
                  data_preprocessor_registry.
    """

    def decorator(fn):
        preprocessor_name = name or fn.__name__
        data_preprocessor_registry[preprocessor_name] = fn
        return fn

    return decorator


def register_config_setter(name=None):
    """
    Decorator to register a configuration setter function used to modify
    the experiment configuration based on the loaded data.

    The decorated function must follow the signature:
        config_setter(
            experiment_config: ExperimentConfig,
            raws: list[mne.io.Raw],
            df_word: pd.DataFrame,
            word_embeddings: np.ndarray
        ) -> ExperimentConfig

    Where:
        - experiment_config: ExperimentConfig dataclass
        - raws: List of MNE Raw objects (continuous iEEG/EEG data)
        - df_word: DataFrame containing word-level metadata (e.g., onset times, labels)
        - word_embeddings: NumPy array of embeddings corresponding to the words

    This function allows updating or enriching the config dynamically based on data contents.

    Args:
        name (str, optional): Optional name to register the config setter under.
                              Defaults to the function's __name__.

    Returns:
        function: A decorator that registers the config setter in
                  config_setter_registry.
    """

    def decorator(fn):
        config_setter_name = name or fn.__name__
        config_setter_registry[config_setter_name] = fn
        return fn

    return decorator


def register_metric(name=None):
    """
    Decorator to register a metric function.

    The decorated function must follow the signature:
        model_constructor(predicted: torch.Tensor, groundtruth: torch.Tensor) -> metric (float)

    Where:
        - predicted (torch.Tensor): Model outputs
        - groundtruth (torch.Tensor): Groundtruth outputs
        - metric: Metric value.

    Args:
        name (str, optional): Optional name to register the function under.
                              Defaults to the function's __name__.

    Returns:
        function: A decorator that registers the metric in
                  metric_registry.
    """

    def decorator(fn):
        model_name = name or fn.__name__
        metric_registry[model_name] = fn
        return fn

    return decorator
