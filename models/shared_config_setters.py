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
    set_input_channels = set_model_spec_fields(
        experiment_config.model_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names,
    )
    if not set_input_channels:
        raise ValueError(
            f"Could not set input channels for model_spec_constructor: {model_spec_constructor_names}"
        )
    return experiment_config


def set_model_spec_fields(
    model_spec: ModelSpec,
    dict_update: dict,
    model_spec_constructor_names: Optional[list[str]] = None,
):
    if model_spec_constructor_names is None:
        value_set = True
        model_spec.params.update(dict_update)
    else:
        value_set = False
        if model_spec.constructor_name in model_spec_constructor_names:
            model_spec.params.update(dict_update)
            value_set = True
        for sub_model_spec in model_spec.sub_models.values():
            value_set |= set_model_spec_fields(
                sub_model_spec, dict_update, model_spec_constructor_names
            )

    return value_set
