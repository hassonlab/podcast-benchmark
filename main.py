import argparse
import yaml
import os
import time

import numpy as np

from config import DataParams, TrainingParams, dict_to_config
import data_utils
import decoding_utils
from loader import import_all_from_package
import registry

# Import modules which define registry functions.
import_all_from_package('neural_conv_decoder')
import_all_from_package('foundation_model')


def load_config(config_path):
    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)
    return experiment_config


def main():
    parser = argparse.ArgumentParser(description="Run decoding model over lag range")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    experiment_config = load_config(args.config)

    # Load all data.
    data_params = dict_to_config(experiment_config['data_params'], DataParams)
    raws = data_utils.load_raws(data_params)
    df_word, word_embeddings = data_utils.load_word_data(data_params)

    # Allow user defined function to alter config if necessary for their model.
    if experiment_config['config_setter_name']:
        config_setter_fn = registry.config_setter_registry[experiment_config['config_setter_name']]
        experiment_config = config_setter_fn(experiment_config, raws, df_word, word_embeddings)
        data_params = dict_to_config(experiment_config['data_params'], DataParams)

    # Add pointer to preprocessing_fn, raw, df_word, and word_embeddings to data_params.
    preprocessing_fn = registry.data_preprocessor_registry[data_params.preprocessing_fn_name]

    model_constructor_fn = registry.model_constructor_registry[experiment_config['model_constructor_name']]

    # Append epoch seconds to prevent accidental overwriting.
    trial_name = experiment_config['trial_name'] + '_' + str(int(time.time()))
    output_dir = os.path.join(experiment_config.get("output_dir", "results/"), trial_name)
    os.makedirs(output_dir, exist_ok=True)

    # Write config to output_dir so it is easy to tell what parameters led to these results.
    with open(os.path.join(output_dir, 'config.yml'), 'w') as fp:
        yaml.dump(experiment_config, fp, default_flow_style=False)

    training_params = dict_to_config(experiment_config['training_params'], TrainingParams)
    lags = np.arange(training_params.min_lag, training_params.max_lag, training_params.lag_step_size)
    weighted_roc_means = decoding_utils.run_training_over_lags(lags, 
                                                raws,
                                                df_word,
                                                word_embeddings,
                                                preprocessing_fn,
                                                model_constructor_fn,
                                                model_params=experiment_config['model_params'],
                                                training_params=training_params,
                                                data_params=data_params,
                                                trial_name=trial_name,
                                                output_dir=output_dir)


if __name__ == '__main__':
    main()