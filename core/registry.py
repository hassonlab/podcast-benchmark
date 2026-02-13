# Registry of functions for creating decoding models.
# Structure: {name: {"constructor": fn, "required_data_getter": str|None}}
model_constructor_registry = {}
# Registry of functions for preprocessing data.
data_preprocessor_registry = {}
# Registry of functions for altering config values after data outputs.
config_setter_registry = {}
# Registry of functions for metric calculations.
metric_registry = {}
# Registry of task information including data getter functions and config types.
# Structure: {task_name: {"getter": function, "config_type": ConfigClass}}
task_registry = {}
# Registry of model data getter functions that add model-specific columns to task DataFrames.
# Structure: {name: function}
model_data_getter_registry = {}


def register_model_constructor(name=None, required_data_getter=None):
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
        required_data_getter (str, optional): Name of the model_data_getter that this model
                              requires. When specified, the data getter will be called
                              automatically before training to add model-specific columns
                              to the task DataFrame. Defaults to None.

    Returns:
        function: A decorator that registers the model constructor in
                  model_constructor_registry with structure:
                  {name: {"constructor": fn, "required_data_getter": str|None}}
    """

    def decorator(fn):
        model_name = name or fn.__name__
        model_constructor_registry[model_name] = {
            "constructor": fn,
            "required_data_getter": required_data_getter,
        }
        return fn

    return decorator


def register_model_data_getter(name=None):
    """
    Decorator to register a model data getter function that adds model-specific
    columns to the task DataFrame.

    The decorated function must follow the signature:
        model_data_getter(task_df: pd.DataFrame, raws: list[mne.io.Raw], model_params: dict)
            -> tuple[pd.DataFrame, list[str]]

    Where:
        - task_df: DataFrame containing task-specific data
        - raws: List of MNE Raw objects (continuous iEEG/EEG data)
        - model_params: Dictionary of model parameters from ModelSpec
        - Returns: Tuple of (enriched_df, list of added column names)

    The added columns should be named to match the model's forward() parameter names
    exactly, so they can be passed automatically via **inputs_dict.

    Args:
        name (str, optional): Optional name to register the getter under.
                              Defaults to the function's __name__.

    Returns:
        function: A decorator that registers the model data getter in
                  model_data_getter_registry.

    Example:
        @register_model_data_getter("diver_data_info")
        def get_diver_data_info(task_df, raws, model_params):
            # Add data_info_list column for DIVER model
            task_df["data_info_list"] = ...
            return task_df, ["data_info_list"]
    """

    def decorator(fn):
        getter_name = name or fn.__name__
        model_data_getter_registry[getter_name] = fn
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
            task_df: pd.DataFrame,
            word_embeddings: np.ndarray
        ) -> ExperimentConfig

    Where:
        - experiment_config: ExperimentConfig dataclass
        - raws: List of MNE Raw objects (continuous iEEG/EEG data)
        - task_df: DataFrame containing task-specific data. Should have at least columns 'start' and 'target'.
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


def register_task_data_getter(name=None, config_type=None):
    """
    Decorator to register a task data getter function that gathers the data relevant for your task.

    The decorated function must follow the signature:
        task_data_getter(task_config: TaskConfig) -> pd.DataFrame

    Where:
        - task_config: TaskConfig object containing task name, data params, and task-specific config
        - Returns: A pandas DataFrame with required columns: 'start', 'target'
                  - start: Time to center the neural data example around. Most likely the start time of the word/token but can vary depending on task.
                  - word: The actual word/token text (Optional)
                  - target: The target variable for prediction tasks

    This function provides a way to load task-specific word-level data that follows
    the expected format for downstream processing while allowing customization of
    data sources and preprocessing steps.

    Args:
        name (str, optional): Optional name to register the task data getter under.
                              Defaults to the function's __name__.
        config_type (type, required): The dataclass type for this task's configuration.
                                      Must be a subclass of BaseTaskConfig.

    Returns:
        function: A decorator that registers the task data getter and config type in
                  task_registry.
    """

    def decorator(fn):
        task_name = name or fn.__name__
        if config_type is None:
            raise ValueError(f"config_type is required when registering task '{task_name}'")
        task_registry[task_name] = {
            "getter": fn,
            "config_type": config_type
        }
        return fn

    return decorator
