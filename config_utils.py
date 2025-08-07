"""
Configuration utilities for argument parsing and config overrides.

This module contains the testable configuration logic extracted from main.py.
"""
import argparse
from dataclasses import is_dataclass
import yaml
from typing import Any, Union
from copy import deepcopy

from config import ExperimentConfig, dict_to_config


def parse_known_args():
    """Parse command line arguments, returning known args and overrides."""
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
            # Skip malformed args with empty keys or values
            if key and val:
                overrides[key] = yaml.safe_load(val)  # preserve types like int, float, bool
    return overrides


def get_nested_value(obj: Union[dict, Any], path: str) -> Any:
    """Get nested value from object using dot notation path."""
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


def set_nested_attr(obj, key_path, value):
    """Set nested attribute using dot notation path."""
    keys = key_path.split(".")
    target = obj
    for key in keys[:-1]:
        if is_dataclass(target):
            target = getattr(target, key)
        elif isinstance(target, dict):
            # Create intermediate dictionary if it doesn't exist
            if key not in target:
                target[key] = {}
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
    """Apply override dictionary to config object, returning a deep copy."""
    config = deepcopy(config)  # Avoid mutating original
    for key_path, value in overrides.items():
        set_nested_attr(config, key_path, value)
    return config


def load_config(config_path) -> ExperimentConfig:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        experiment_config = yaml.safe_load(f)
    return dict_to_config(experiment_config, ExperimentConfig)


def load_config_with_overrides(config_path: str, overrides: dict):
    """Load config from YAML file and apply command-line overrides."""
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    base_config = dict_to_config(raw_cfg, ExperimentConfig)
    return apply_overrides(base_config, overrides)