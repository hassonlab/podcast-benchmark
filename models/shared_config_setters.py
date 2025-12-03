from typing import Optional

from core import registry
from core.config import ExperimentConfig, ModelSpec


@registry.register_config_setter()
def set_input_channels(
    experiment_config: ExperimentConfig,
    raws,
    _task_df,
    model_spec_constructor_names: Optional[list[str]] = None,
) -> ExperimentConfig:
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    set_input_channels = _set_model_spec_input_channels(
        experiment_config.model_spec,
        num_electrodes,
    )
    if not set_input_channels:
        raise ValueError(
            f"Could not set input channels for model_spec_constructor: {model_spec_constructor_names}"
        )
    return experiment_config


def _set_model_spec_input_channels(
    model_spec: ModelSpec,
    num_electrodes: int,
    model_spec_constructor_names: Optional[list[str]] = None,
):
    if model_spec_constructor_names is None:
        value_set = True
        model_spec.params["input_channels"] = num_electrodes
    else:
        value_set = False
        for sub_model_spec in model_spec.sub_models.values():
            if sub_model_spec.constructor_name in model_spec_constructor_names:
                sub_model_spec.params["input_channels"] = num_electrodes
                value_set = True
        for sub_model_spec in model_spec.sub_models.values():
            value_set |= _set_model_spec_input_channels(
                sub_model_spec, num_electrodes, model_spec_constructor_names
            )

    return value_set
