import numpy as np

from core import registry
from core.config import ExperimentConfig
from models.shared_config_setters import set_input_channels


@registry.register_config_setter("neural_conv")
def neural_conv_config_setter(
    experiment_config: ExperimentConfig, raws, task_df
) -> ExperimentConfig:
    experiment_config = set_input_channels(experiment_config, raws, task_df)

    experiment_config.model_params["input_timesteps"] = np.floor(
        experiment_config.data_params.window_width * 512
    ) / experiment_config.data_params.preprocessor_params.get("num_average_samples")

    return experiment_config
