import argparse
from dataclasses import is_dataclass
import yaml
import os
from datetime import datetime
from typing import Any, Union
from copy import deepcopy

import numpy as np

from config import ExperimentConfig, dict_to_config
import data_utils
import decoding_utils
from loader import import_all_from_package
import registry

# Import modules which define registry functions. REQUIRED FOR ANY NEW MODELS.
import_all_from_package("neural_conv_decoder")
# import_all_from_package("foundation_model")  # Not needed for PopulationTransformer
import_all_from_package("population_transformer_module")
# Add your model import here!


def load_config(config_path) -> ExperimentConfig:
    with open(config_path, "r") as f:
        experiment_config = yaml.safe_load(f)
    return dict_to_config(experiment_config, ExperimentConfig)


def get_nested_value(obj: Union[dict, Any], path: str) -> Any:
    fields = path.split(".")
    current = obj
    for field in fields:
        if isinstance(current, dict):
            current = current[field]
        elif is_dataclass(current):
            current = getattr(current, field)
        else:
            raise TypeError(
                f"Cannot access field '{field}' on non-dict, non-dataclass object: {current}"
            )
    return current


def parse_known_args():
    parser = argparse.ArgumentParser(description="Run decoding model over lag range")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args, unknown_args = parser.parse_known_args()
    overrides = parse_override_args(unknown_args)
    return args, overrides


def parse_override_args(unknown_args):
    """
    Parse args like --model_params.checkpoint_dir=some_path into a dictionary.
    """
    overrides = {}
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            overrides[key] = yaml.safe_load(val)  # preserve types like int, float, bool
    return overrides


def set_nested_attr(obj, key_path, value):
    keys = key_path.split(".")
    target = obj
    for key in keys[:-1]:
        if is_dataclass(target):
            target = getattr(target, key)
        elif isinstance(target, dict):
            target = target[key]
        else:
            raise TypeError(
                f"Unsupported type {type(target)} for intermediate key: {key}"
            )

    final_key = keys[-1]
    if is_dataclass(target):
        setattr(target, final_key, value)
    elif isinstance(target, dict):
        target[final_key] = value
    else:
        raise TypeError(f"Unsupported type {type(target)} for final key: {final_key}")


def apply_overrides(config, overrides):
    config = deepcopy(config)  # Avoid mutating original
    for key_path, value in overrides.items():
        set_nested_attr(config, key_path, value)
    return config


def load_config_with_overrides(config_path: str, overrides: dict):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    base_config = dict_to_config(raw_cfg, ExperimentConfig)
    return apply_overrides(base_config, overrides)


def main():
    print("🚀 Starting PopulationTransformer experiment...")
    print("=" * 60)
    
    args, overrides = parse_known_args()
    experiment_config = load_config_with_overrides(args.config, overrides)
    print(f"📋 Loaded config: {experiment_config.trial_name}")

    # Load all data.
    print("📊 Loading neural data...")
    raws = data_utils.load_raws(experiment_config.data_params)
    print(f"   ✅ Loaded {len(raws)} raw data files")
    
    print("📝 Loading word data...")
    df_word = data_utils.load_word_data(experiment_config.data_params)
    print(f"   ✅ Loaded {len(df_word)} word entries")

    # Allow user defined function to alter config if necessary for their model.
    if experiment_config.config_setter_name:
        print(f"⚙️  Running config setter: {experiment_config.config_setter_name}")
        config_setter_fn = registry.config_setter_registry[
            experiment_config.config_setter_name
        ]
        experiment_config = config_setter_fn(experiment_config, raws, df_word)
        print("   ✅ Config setter completed")

    # User defined preprocessing function.
    preprocessing_fn = None
    if experiment_config.data_params.preprocessing_fn_name:
        print(f"🔧 Loading preprocessing function: {experiment_config.data_params.preprocessing_fn_name}")
        preprocessing_fn = registry.data_preprocessor_registry[
            experiment_config.data_params.preprocessing_fn_name
        ]
        print("   ✅ Preprocessing function loaded")

    # User defined model constructor function.
    print(f"🏗️  Loading model constructor: {experiment_config.model_constructor_name}")
    model_constructor_fn = registry.model_constructor_registry[
        experiment_config.model_constructor_name
    ]
    print("   ✅ Model constructor loaded")

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
    print(f"📁 Creating output directories for trial: {trial_name}")
    
    output_dir = os.path.join(experiment_config.output_dir, trial_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"   📂 Output dir: {output_dir}")

    model_dir = os.path.join(os.path.join(experiment_config.model_dir, trial_name))
    os.makedirs(model_dir, exist_ok=True)
    print(f"   📂 Model dir: {model_dir}")

    tensorboard_dir = os.path.join(
        os.path.join(experiment_config.tensorboard_dir, trial_name)
    )
    os.makedirs(tensorboard_dir, exist_ok=True)
    print(f"   📂 Tensorboard dir: {tensorboard_dir}")

    # Write config to output_dir so it is easy to tell what parameters led to these results.
    print("💾 Saving configuration...")
    with open(os.path.join(output_dir, "config.yml"), "w") as fp:
        yaml.dump(experiment_config, fp, default_flow_style=False)
    print("   ✅ Configuration saved")

    lags = np.arange(
        experiment_config.training_params.min_lag,
        experiment_config.training_params.max_lag,
        experiment_config.training_params.lag_step_size,
    )
    print(f"🔄 Starting training over {len(lags)} lags: {lags[0]} to {lags[-1]} ms")
    print("=" * 60)
    
    weighted_roc_means = decoding_utils.run_training_over_lags(
        lags,
        raws,
        df_word,
        preprocessing_fn,
        model_constructor_fn,
        model_params=experiment_config.model_params,
        training_params=experiment_config.training_params,
        data_params=experiment_config.data_params,
        output_dir=output_dir,
        model_dir=model_dir,
        tensorboard_dir=tensorboard_dir,
        write_to_tensorboard=True,
    )
    
    print("=" * 60)
    print("🎉 Experiment completed successfully!")
    print(f"📊 Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
