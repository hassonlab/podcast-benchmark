import numpy as np

from core import registry
from core.config import ExperimentConfig
from models.shared_config_setters import set_input_channels, set_model_spec_fields


@registry.register_config_setter("neural_conv")
def neural_conv_config_setter(
    experiment_config: ExperimentConfig, raws, task_df
) -> ExperimentConfig:
    relevant_model_constructors = ["pitom_model", "ensemble_pitom_model", "decoder_mlp"]
    experiment_config = set_input_channels(
        experiment_config,
        raws,
        task_df,
        relevant_model_constructors,
    )

    pp = experiment_config.task_config.data_params.preprocessor_params
    if isinstance(pp, list):
        num_average_samples = next(
            (p.get("num_average_samples") for p in pp if isinstance(p, dict) and "num_average_samples" in p),
            None,
        )
    else:
        num_average_samples = pp.get("num_average_samples") if pp else None

    set_model_spec_fields(
        experiment_config.model_spec,
        {
            "input_timesteps": np.floor(
                experiment_config.task_config.data_params.window_width * 512
            )
            // num_average_samples
        },
        relevant_model_constructors,
    )

    return experiment_config
