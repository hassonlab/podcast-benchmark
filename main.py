import os
from datetime import datetime

import numpy as np

from config import ExperimentConfig
import data_utils
import decoding_utils
import task_utils
from loader import import_all_from_package
import registry
from config_utils import parse_known_args, load_config_with_overrides, get_nested_value

# Import modules which define registry functions. REQUIRED FOR ANY NEW MODELS.
import_all_from_package("neural_conv_decoder")
import_all_from_package("foundation_model")

# Add your model import here!


def main():
    args, overrides = parse_known_args()
    experiment_config = load_config_with_overrides(args.config, overrides)

    # Load all data.
    raws = data_utils.load_raws(experiment_config.data_params)
    df_word = registry.task_data_getter_registry[experiment_config.task_name](
        experiment_config.data_params
    )

    # Allow user defined function to alter config if necessary for their model.
    if experiment_config.config_setter_name:
        config_setter_fn = registry.config_setter_registry[
            experiment_config.config_setter_name
        ]
        experiment_config = config_setter_fn(experiment_config, raws, df_word)

    # User defined preprocessing function.
    preprocessing_fn = None
    if experiment_config.data_params.preprocessing_fn_name:
        preprocessing_fn = registry.data_preprocessor_registry[
            experiment_config.data_params.preprocessing_fn_name
        ]

    # User defined model constructor function.
    model_constructor_fn = registry.model_constructor_registry[
        experiment_config.model_constructor_name
    ]

    # Overwrite subject id's and set per-subject electrodes based on file if provided.
    if experiment_config.data_params.electrode_file_path:
        subject_id_map = data_utils.read_subject_mapping(
            "data/participants.tsv", delimiter="\t"
        )
        subject_electrode_map = data_utils.read_electrode_file(
            experiment_config.data_params.electrode_file_path,
            subject_mapping=subject_id_map,
        )
        experiment_config.data_params.subject_ids = list(subject_electrode_map.keys())
        experiment_config.data_params.per_subject_electrodes = subject_electrode_map

    # Generate trial name if user specified format string.
    trial_name = experiment_config.trial_name
    if experiment_config.format_fields:
        format_values = [
            get_nested_value(experiment_config, s)
            for s in experiment_config.format_fields
        ]
        trial_name = trial_name.format(*format_values)
    # Append timestamp to prevent accidental overwriting.
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_name = f"{trial_name}_{timestamp}"
    output_dir = os.path.join(experiment_config.output_dir, trial_name)
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(os.path.join(experiment_config.model_dir, trial_name))
    os.makedirs(model_dir, exist_ok=True)

    tensorboard_dir = os.path.join(
        os.path.join(experiment_config.tensorboard_dir, trial_name)
    )
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Write config to output_dir so it is easy to tell what parameters led to these results.
    import yaml
    from dataclasses import asdict

    with open(os.path.join(output_dir, "config.yml"), "w") as fp:
        yaml.dump(asdict(experiment_config), fp, default_flow_style=False)

    # Decide what lags we need to train over.
    if experiment_config.training_params.lag:
        lags = [experiment_config.training_params.lag]
    else:
        lags = np.arange(
            experiment_config.training_params.min_lag,
            experiment_config.training_params.max_lag,
            experiment_config.training_params.lag_step_size,
        )
    decoding_utils.run_training_over_lags(
        lags,
        raws,
        df_word,
        preprocessing_fn,
        model_constructor_fn,
        experiment_config.task_name,
        model_params=experiment_config.model_params,
        training_params=experiment_config.training_params,
        data_params=experiment_config.data_params,
        output_dir=output_dir,
        model_dir=model_dir,
        tensorboard_dir=tensorboard_dir,
        write_to_tensorboard=True,
    )


if __name__ == "__main__":
    main()
