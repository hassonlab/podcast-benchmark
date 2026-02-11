"""
Configuration utilities for argument parsing and config overrides.

This module contains the testable configuration logic extracted from main.py.
"""

import argparse
from dataclasses import is_dataclass
import re
import yaml
from typing import Any, Union
from copy import deepcopy

from core.config import ExperimentConfig, TaskConfig, DataParams, MultiTaskConfig, dict_to_config
from core import registry
from utils import data_utils


def partial_format(template: str, **kwargs) -> str:
    """Format a string with provided kwargs while preserving unmatched format variables.

    This function allows partial formatting of strings containing format variables.
    Any format variables not provided in kwargs will be preserved for later formatting.

    Example:
        >>> partial_format("{a} and {b}", a="hello")
        "hello and {b}"

        >>> partial_format("{prev_checkpoint_dir}/lag_{lag}/fold_{fold}",
        ...                prev_checkpoint_dir="/path/to/checkpoint")
        "/path/to/checkpoint/lag_{lag}/fold_{fold}"

    Args:
        template: String template with format variables like {var_name}
        **kwargs: Values to substitute for matching format variables

    Returns:
        Formatted string with unmatched variables preserved
    """
    # Find all format variables in the template
    format_vars = re.findall(r'\{(\w+)\}', template)

    # Create a complete format dict: provided kwargs + self-referencing for missing vars
    format_dict = {}
    for var in format_vars:
        if var in kwargs:
            format_dict[var] = kwargs[var]
        else:
            # Preserve the variable by formatting it to itself
            format_dict[var] = f"{{{var}}}"

    return template.format(**format_dict)


def interpolate_prev_checkpoint_dir(model_spec, prev_checkpoint_dir: str):
    """Recursively interpolate {prev_checkpoint_dir} in checkpoint paths of a ModelSpec.

    Creates a deep copy of the model_spec and replaces any {prev_checkpoint_dir} placeholders
    in checkpoint_path fields with the actual directory from the previous task.

    Uses partial_format() with prev_checkpoint_dir parameter, which allows it to be combined
    with other placeholders like {lag} and {fold} that will be formatted later.

    Args:
        model_spec: ModelSpec to process
        prev_checkpoint_dir: Checkpoint directory path from previous task

    Returns:
        New ModelSpec with interpolated paths

    Raises:
        ValueError: If {prev_checkpoint_dir} is found but prev_checkpoint_dir is None or empty

    Example:
        >>> spec = ModelSpec(
        ...     constructor_name="encoder",
        ...     checkpoint_path="{prev_checkpoint_dir}/lag_{lag}/best_model_fold{fold}.pt"
        ... )
        >>> new_spec = interpolate_prev_checkpoint_dir(spec, "checkpoints/pretrain/run_123")
        >>> print(new_spec.checkpoint_path)
        checkpoints/pretrain/run_123/lag_{lag}/best_model_fold{fold}.pt
    """
    # Deep copy to avoid mutating original
    new_spec = deepcopy(model_spec)

    # Interpolate checkpoint_path at this level
    if new_spec.checkpoint_path and "{prev_checkpoint_dir}" in new_spec.checkpoint_path:
        if not prev_checkpoint_dir:
            raise ValueError(
                f"Cannot interpolate {{prev_checkpoint_dir}} in checkpoint path "
                f"'{new_spec.checkpoint_path}' - no previous checkpoint directory available. "
                f"This is likely the first task in a multi-task config."
            )
        # Use partial_format() to replace {prev_checkpoint_dir}, leaving {lag} and {fold} intact
        new_spec.checkpoint_path = partial_format(
            new_spec.checkpoint_path,
            prev_checkpoint_dir=prev_checkpoint_dir
        )

    # Recursively process sub-models
    if new_spec.sub_models:
        for param_name, sub_spec in new_spec.sub_models.items():
            new_spec.sub_models[param_name] = interpolate_prev_checkpoint_dir(
                sub_spec, prev_checkpoint_dir
            )

    return new_spec


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
    """Get nested value from object using dot notation path.

    Supports accessing:
    - Dictionary keys: obj.key1.key2
    - Dataclass attributes: obj.attr1.attr2
    - List indices: obj.list_field.0.nested_field
    """
    fields = path.split(".")
    current = obj
    for field in fields:
        if isinstance(current, dict):
            current = current[field]
        elif is_dataclass(current):
            current = getattr(current, field)
        elif isinstance(current, list):
            # Handle list indexing
            try:
                index = int(field)
                current = current[index]
            except (ValueError, IndexError) as e:
                raise TypeError(
                    f"Cannot access list with key '{field}': {e}"
                )
        else:
            raise TypeError(
                f"Cannot access field '{field}' on non-dict, non-dataclass, non-list object: {current}"
            )
    return current


def set_nested_attr(obj, key_path, value):
    """Set nested attribute using dot notation path.

    Supports accessing:
    - Dictionary keys: obj.key1.key2
    - Dataclass attributes: obj.attr1.attr2
    - List indices: obj.list_field.0.nested_field
    """
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
        elif isinstance(target, list):
            # Handle list indexing
            try:
                index = int(key)
                target = target[index]
            except (ValueError, IndexError) as e:
                raise TypeError(
                    f"Cannot access list with key '{key}': {e}"
                )
        else:
            raise TypeError(
                f"Unsupported type {type(target)} for intermediate key: {key}"
            )

    final_key = keys[-1]
    if is_dataclass(target):
        setattr(target, final_key, value)
    elif isinstance(target, dict):
        target[final_key] = value
    elif isinstance(target, list):
        # Handle list indexing for final key
        try:
            index = int(final_key)
            target[index] = value
        except (ValueError, IndexError) as e:
            raise TypeError(
                f"Cannot set list element with key '{final_key}': {e}"
            )
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

    # Finalize the experiment config (electrode files, config setters)
    experiment_config = _finalize_experiment_config(experiment_config, subject_mapping_file)

    return experiment_config


def validate_multi_task_config(config: MultiTaskConfig):
    """Validate multi-task configuration.

    Args:
        config: MultiTaskConfig to validate

    Raises:
        ValueError: If validation fails
    """
    if not config.tasks:
        raise ValueError("MultiTaskConfig must have at least one task")

    # Validate task names are unique (if provided and non-empty)
    task_names = [t.trial_name for t in config.tasks if t.trial_name]
    if len(task_names) != len(set(task_names)):
        raise ValueError("Task trial_names must be unique")


def _finalize_experiment_config(
    experiment_config: ExperimentConfig,
    subject_mapping_file: str = "data/participants.tsv"
) -> ExperimentConfig:
    """Finalize an experiment config by processing task_config and electrode files.

    Extracted from load_experiment_config to allow reuse for multi-task configs.

    Args:
        experiment_config: ExperimentConfig to finalize
        subject_mapping_file: Path to subject mapping file

    Returns:
        Finalized ExperimentConfig
    """
    # Overwrite subject id's and set per-subject electrodes based on file if provided
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

    # Allow user defined function to alter config if necessary for their model
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


def load_multi_task_config(
    config_path: str, overrides: dict, subject_mapping_file="data/participants.tsv"
) -> MultiTaskConfig:
    """Load multi-task config from file and apply overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Dict of override values to apply
        subject_mapping_file: Path to subject mapping file

    Returns:
        MultiTaskConfig with loaded and finalized tasks

    Raises:
        ValueError: If file doesn't contain 'tasks' field
    """
    # Load raw config
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Check if this is a multi-task config
    if "tasks" not in raw_cfg:
        raise ValueError(
            f"Config file {config_path} does not contain 'tasks' field. "
            "Use load_experiment_config() for single-task configs."
        )

    # Apply overrides to raw dict first
    raw_cfg = apply_overrides(raw_cfg, overrides)

    # Extract shared_params (removed from task processing)
    shared_params = raw_cfg.get("shared_params", None)

    # Load each task as an ExperimentConfig
    tasks = []
    for task_dict in raw_cfg["tasks"]:
        # Keep task_config as dict for now
        task_config_dict = task_dict.pop("task_config", {})

        # Convert everything except task_config to ExperimentConfig
        task_config = dict_to_config(task_dict, ExperimentConfig)

        # Now handle task_config separately (similar to load_experiment_config)
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
            task_config.task_config = TaskConfig(
                task_name=task_name,
                data_params=data_params,
                task_specific_config=task_specific_config,
            )

        # Finalize the task config (electrode files, config setters)
        task_config = _finalize_experiment_config(task_config, subject_mapping_file)

        tasks.append(task_config)

    multi_config = MultiTaskConfig(tasks=tasks, shared_params=shared_params)
    validate_multi_task_config(multi_config)

    return multi_config


def load_config(
    config_path: str, overrides: dict, subject_mapping_file="data/participants.tsv"
) -> Union[ExperimentConfig, MultiTaskConfig]:
    """Auto-detect and load single or multi-task config.

    Args:
        config_path: Path to YAML config file
        overrides: Dict of override values to apply
        subject_mapping_file: Path to subject mapping file

    Returns:
        ExperimentConfig if single-task, MultiTaskConfig if multi-task
    """
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    if "tasks" in raw_cfg:
        return load_multi_task_config(config_path, overrides, subject_mapping_file)
    else:
        return load_experiment_config(config_path, overrides, subject_mapping_file)
