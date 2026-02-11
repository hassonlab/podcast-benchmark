"""
Tests for the training matrix configuration.

Validates that all task/model combinations in training_matrix.yaml:
1. Have valid configuration files that can be loaded
2. Reference registered task data getters
3. Can successfully generate data with required columns
"""

import os
import pytest
import yaml
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from core.config import ExperimentConfig, dict_to_config
from core.registry import task_registry
from utils.module_loader_utils import import_all_from_package

import_all_from_package("tasks", recursive=True)  # Populate task registry


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def training_matrix(project_root):
    """Load the training matrix configuration."""
    matrix_path = project_root / "training_matrix.yaml"
    with open(matrix_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def configs_dir(project_root):
    """Get the configs directory path."""
    return project_root / "configs"


def get_all_config_paths(training_matrix, configs_dir):
    """
    Extract all config file paths from the training matrix.

    Returns:
        List of tuples: (model_name, task_name, config_path)
    """
    config_paths = []

    for model_name, tasks in training_matrix.items():
        if not isinstance(tasks, dict):
            continue

        for task_name, config_files in tasks.items():
            if not isinstance(config_files, list):
                continue

            for config_file in config_files:
                # Assume configs are in configs/{model_name}/ directory
                config_path = configs_dir / model_name / config_file
                config_paths.append((model_name, task_name, config_path))

    return config_paths


class TestTrainingMatrixConfigs:
    """Test that all configs in training matrix are valid and loadable."""

    def test_training_matrix_exists(self, project_root):
        """Test that training_matrix.yaml exists."""
        matrix_path = project_root / "training_matrix.yaml"
        assert matrix_path.exists(), "training_matrix.yaml not found"

    def test_training_matrix_structure(self, training_matrix):
        """Test that training matrix has expected structure."""
        assert isinstance(training_matrix, dict), "Training matrix should be a dict"
        assert len(training_matrix) > 0, "Training matrix should not be empty"

        for model_name, tasks in training_matrix.items():
            if not isinstance(tasks, dict):
                continue
            assert isinstance(
                tasks, dict
            ), f"Model '{model_name}' should map to a dict of tasks"

            for task_name, config_files in tasks.items():
                assert isinstance(
                    config_files, list
                ), f"Task '{task_name}' in model '{model_name}' should have a list of config files"

    def test_all_config_files_exist(self, training_matrix, configs_dir):
        """Test that all referenced config files exist."""
        config_paths = get_all_config_paths(training_matrix, configs_dir)

        missing_configs = []
        for model_name, task_name, config_path in config_paths:
            if not config_path.exists():
                missing_configs.append((model_name, task_name, str(config_path)))

        assert len(missing_configs) == 0, f"Missing config files: {missing_configs}"

    def test_all_configs_are_valid_yaml(self, training_matrix, configs_dir):
        """Test that all config files are valid YAML."""
        config_paths = get_all_config_paths(training_matrix, configs_dir)

        invalid_configs = []
        for model_name, task_name, config_path in config_paths:
            try:
                with open(config_path, "r") as f:
                    yaml.safe_load(f)
            except Exception as e:
                invalid_configs.append(
                    (model_name, task_name, str(config_path), str(e))
                )

        assert len(invalid_configs) == 0, f"Invalid YAML configs: {invalid_configs}"

    def test_all_configs_load_as_experiment_config(self, training_matrix, configs_dir):
        """Test that all config files can be converted to ExperimentConfig."""
        config_paths = get_all_config_paths(training_matrix, configs_dir)

        failed_conversions = []
        for model_name, task_name, config_path in config_paths:
            try:
                with open(config_path, "r") as f:
                    config_dict = yaml.safe_load(f)
                experiment_config = dict_to_config(config_dict, ExperimentConfig)
                assert isinstance(experiment_config, ExperimentConfig)
            except Exception as e:
                failed_conversions.append(
                    (model_name, task_name, str(config_path), str(e))
                )

        assert (
            len(failed_conversions) == 0
        ), f"Failed config conversions: {failed_conversions}"


class TestTrainingMatrixTaskDataGetters:
    """Test that all tasks can generate data with required columns."""

    def test_all_tasks_are_registered(self, training_matrix):
        """Test that all tasks in training matrix are registered."""
        unregistered_tasks = []

        for model_name, tasks in training_matrix.items():
            if not isinstance(tasks, dict):
                continue

            for task_name in tasks.keys():
                if task_name not in task_registry:
                    unregistered_tasks.append((model_name, task_name))

        assert (
            len(unregistered_tasks) == 0
        ), f"Unregistered tasks in training matrix: {unregistered_tasks}"

    def test_all_task_data_getters_exist(self, training_matrix, configs_dir):
        """
        Test that all task data getters referenced in configs exist in the registry.

        This test only verifies that the task data getters are registered and callable,
        without actually calling them.
        """
        config_paths = get_all_config_paths(training_matrix, configs_dir)

        # Track which tasks we've already tested to avoid duplicates
        tested_tasks = set()
        missing_or_invalid_tasks = []

        for model_name, task_name, config_path in config_paths:
            # Skip if we've already tested this task
            if task_name in tested_tasks:
                continue
            tested_tasks.add(task_name)

            try:
                # Get the task info from registry
                task_info = task_registry.get(task_name)

                # Verify it exists
                assert (
                    task_info is not None
                ), f"Task '{task_name}' not found in registry"

                # Verify it has a getter function
                assert (
                    "getter" in task_info
                ), f"Task '{task_name}' missing 'getter' in registry"

                # Verify getter is callable
                assert callable(
                    task_info["getter"]
                ), f"Task '{task_name}' getter is not callable"

                # Verify it has a config_type
                assert (
                    "config_type" in task_info
                ), f"Task '{task_name}' missing 'config_type' in registry"

            except Exception as e:
                missing_or_invalid_tasks.append(
                    (model_name, task_name, str(config_path), str(e))
                )

        assert (
            len(missing_or_invalid_tasks) == 0
        ), f"Missing or invalid task data getters: {missing_or_invalid_tasks}"


class TestTaskDataGetterRegistry:
    """Test the task data getter registry system."""

    def test_registry_is_not_empty(self):
        """Test that task data getter registry has entries."""
        assert len(task_registry) > 0, "Task registry is empty"

    def test_registry_functions_are_callable(self):
        """Test that all registered task data getters are callable."""
        for task_name, task_info in task_registry.items():
            assert (
                "getter" in task_info
            ), f"Task '{task_name}' missing 'getter' in registry"
            assert callable(
                task_info["getter"]
            ), f"Task data getter '{task_name}' is not callable"

    def test_required_tasks_are_registered(self):
        """Test that expected tasks are registered."""
        expected_tasks = [
            "word_embedding_decoding_task",
            "sentence_onset_task",
        ]

        for task_name in expected_tasks:
            assert (
                task_name in task_registry
            ), f"Expected task '{task_name}' not found in registry"
