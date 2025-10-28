from core import registry
from core.config import ExperimentConfig


@registry.register_config_setter()
def set_input_channels(
    experiment_config: ExperimentConfig, raws, _task_df
) -> ExperimentConfig:
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_params["input_channels"] = num_electrodes
    return experiment_config
