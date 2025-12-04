"""
Configuration utilities for argument parsing and config overrides.

This module contains the testable configuration logic extracted from main.py.
"""

import argparse
from dataclasses import is_dataclass
import yaml
from typing import Any, Union
from copy import deepcopy

from core.config import ExperimentConfig, TaskConfig, DataParams, dict_to_config
from core import registry
from utils import data_utils


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
                overrides[key] = yaml.safe_load(
                    val
                )  # preserve types like int, float, bool
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


def load_experiment_config(
    config_path: str, overrides: dict, subject_mapping_file="data/participants.tsv"
) -> ExperimentConfig:
    """Load experiment config from file and apply overrides. Ensures correct task config is loaded."""
    # Load raw config and apply overrides, but keep task_config as dict
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Apply overrides to raw dict first
    raw_cfg = apply_overrides(raw_cfg, overrides)

    # Keep task_config as dict for now
    task_config_dict = raw_cfg.pop("task_config", {})

    # Convert everything except task_config to ExperimentConfig
    experiment_config = dict_to_config(raw_cfg, ExperimentConfig)

    # Now handle task_config separately
    if isinstance(task_config_dict, dict) and task_config_dict:
        task_name = task_config_dict["task_name"]
        task_info = registry.task_registry[task_name]

        # Instantiate DataParams
        data_params = dict_to_config(task_config_dict["data_params"], DataParams)

        # Instantiate task-specific config
        config_class = task_info["config_type"]
        task_specific_config = dict_to_config(
            task_config_dict.get("task_specific_config", {}), config_class
        )

        # Create TaskConfig
        experiment_config.task_config = TaskConfig(
            task_name=task_name,
            data_params=data_params,
            task_specific_config=task_specific_config,
        )

    # Overwrite subject id's and set per-subject electrodes based on file if provided.
    if experiment_config.task_config.data_params.electrode_file_path:
        subject_id_map = data_utils.read_subject_mapping(
            subject_mapping_file, delimiter="\t"
        )
        subject_electrode_map = data_utils.read_electrode_file(
            experiment_config.task_config.data_params.electrode_file_path,
            subject_mapping=subject_id_map,
        )
        experiment_config.task_config.data_params.subject_ids = list(
            subject_electrode_map.keys()
        )
        experiment_config.task_config.data_params.per_subject_electrodes = (
            subject_electrode_map
        )

    # Allow user defined function to alter config if necessary for their model.
    task_specific_config_setters = (
        experiment_config.task_config.task_specific_config.required_config_setter_names
    )
    if experiment_config.config_setter_name or task_specific_config_setters:
        if experiment_config.config_setter_name and not isinstance(
            experiment_config.config_setter_name, list
        ):
            experiment_config.config_setter_name = [
                experiment_config.config_setter_name
            ]
        if task_specific_config_setters:
            if experiment_config.config_setter_name is None:
                experiment_config.config_setter_name = []
            experiment_config.config_setter_name = (
                task_specific_config_setters + experiment_config.config_setter_name
            )

    return experiment_config
