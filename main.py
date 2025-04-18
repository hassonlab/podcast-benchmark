import argparse
import yaml
import os
import time

import numpy as np

from config import ExperimentConfig, dict_to_config
import data_utils
import decoding_utils
from loader import import_all_from_package
import registry

# Import modules which define registry functions. REQUIRED FOR ANY NEW MODELS.
import_all_from_package('neural_conv_decoder')
import_all_from_package('foundation_model')
# Add your model import here!


def load_config(config_path) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)
    return dict_to_config(experiment_config, ExperimentConfig)


def main():
    parser = argparse.ArgumentParser(description="Run decoding model over lag range")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    experiment_config = load_config(args.config)

    # Load all data.
    raws = data_utils.load_raws(experiment_config.data_params)
    df_word, word_embeddings = data_utils.load_word_data(experiment_config.data_params)

    # Allow user defined function to alter config if necessary for their model.
    if experiment_config.config_setter_name:
        config_setter_fn = registry.config_setter_registry[experiment_config.config_setter_name]
        experiment_config = config_setter_fn(experiment_config, raws, df_word, word_embeddings)

    # User defined preprocessing function.
    preprocessing_fn = None
    if experiment_config.data_params.preprocessing_fn_name:
        preprocessing_fn = registry.data_preprocessor_registry[experiment_config.data_params.preprocessing_fn_name]
        
    # User defined model constructor function.
    model_constructor_fn = registry.model_constructor_registry[experiment_config.model_constructor_nameq]

    # Append epoch seconds to prevent accidental overwriting.
    trial_name = experiment_config.trial_name + '_' + str(int(time.time()))
    output_dir = os.path.join(experiment_config.output_dir, trial_name)
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(os.path.join(experiment_config.model_dir, trial_name))
    os.makedirs(model_dir, exist_ok=True)

    # Write config to output_dir so it is easy to tell what parameters led to these results.
    with open(os.path.join(output_dir, 'config.yml'), 'w') as fp:
        yaml.dump(experiment_config, fp, default_flow_style=False)

    lags = np.arange(experiment_config.training_params.min_lag,
                     experiment_config.training_params.max_lag,
                     experiment_config.training_params.lag_step_size)
    weighted_roc_means = decoding_utils.run_training_over_lags(lags, 
                                                raws,
                                                df_word,
                                                word_embeddings,
                                                preprocessing_fn,
                                                model_constructor_fn,
                                                model_params=experiment_config.model_params,
                                                training_params=experiment_config.training_params,
                                                data_params=experiment_config.data_params,
                                                trial_name=trial_name,
                                                output_dir=output_dir,
                                                model_dir=model_dir)


if __name__ == '__main__':
    main()