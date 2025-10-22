import registry
from config import ExperimentConfig
import numpy as np


@registry.register_data_preprocessor()
def preprocess_neural_data(data, preprocessor_params):
    return data.reshape(
        data.shape[0], data.shape[1], -1, preprocessor_params["num_average_samples"]
    ).mean(-1)


@registry.register_config_setter("neural_conv")
def set_config_input_channels(
    experiment_config: ExperimentConfig, raws, _df_word
) -> ExperimentConfig:
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_params["input_channels"] = num_electrodes

    if experiment_config.task_name == "content_noncontent_task":
        experiment_config.model_params["logistic_regression"] = True
    else:
        experiment_config.model_params["logistic_regression"] = False

    experiment_config.model_params["input_timesteps"] = np.floor(
        experiment_config.data_params.window_width * 512)/ experiment_config.data_params.preprocessor_params.num_average_samples
            


    return experiment_config
