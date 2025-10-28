"""Core framework components for the podcast benchmark."""

from .config import DataParams, TrainingParams, ExperimentConfig, dict_to_config
from .registry import (
    register_model_constructor,
    register_data_preprocessor,
    register_config_setter,
    register_metric,
    register_task_data_getter,
    model_constructor_registry,
    data_preprocessor_registry,
    config_setter_registry,
    metric_registry,
    task_data_getter_registry,
)

__all__ = [
    # Config
    "DataParams",
    "TrainingParams",
    "ExperimentConfig",
    "dict_to_config",
    # Registry decorators
    "register_model_constructor",
    "register_data_preprocessor",
    "register_config_setter",
    "register_metric",
    "register_task_data_getter",
    # Registry dictionaries
    "model_constructor_registry",
    "data_preprocessor_registry",
    "config_setter_registry",
    "metric_registry",
    "task_data_getter_registry",
]
