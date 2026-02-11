"""
Tests for multi-task configuration functionality in utils/config_utils.py.

Following TDD principles - these tests define the expected behavior
before implementation.
"""

import pytest
import tempfile
import os
from copy import deepcopy

from core.config import ExperimentConfig, MultiTaskConfig, TaskConfig, ModelSpec
from utils.config_utils import (
    apply_overrides,  # Reuse existing function for shared params
    load_multi_task_config,
    load_config,
    validate_multi_task_config,
    parse_override_args,
    set_nested_attr,
    get_nested_value,
    load_experiment_config,
    partial_format,
    interpolate_prev_checkpoint_dir,
)


@pytest.fixture
def override_args_list():
    """Sample command-line override arguments for testing."""
    return [
        "--model_params.learning_rate=0.01",
        "--training_params.batch_size=128",
        "--training_params.epochs=50",
        "--data_params.subject_ids=[1,2,3,4]",
        "--model_params.complex_param={'nested': True, 'value': 42}",
    ]


class TestParseOverrideArgs:
    """Test parsing of command-line override arguments."""

    def test_parse_single_override(self):
        """Test parsing single override argument."""
        args = ["--model_params.learning_rate=0.01"]

        overrides = parse_override_args(args)

        assert overrides == {"model_params.learning_rate": 0.01}

    def test_parse_multiple_overrides(self, override_args_list):
        """Test parsing multiple override arguments."""
        overrides = parse_override_args(override_args_list)

        expected = {
            "model_params.learning_rate": 0.01,
            "training_params.batch_size": 128,
            "training_params.epochs": 50,
            "data_params.subject_ids": [1, 2, 3, 4],
            "model_params.complex_param": {"nested": True, "value": 42},
        }

        assert overrides == expected

    def test_parse_different_types(self):
        """Test that YAML parsing preserves different Python types."""
        args = [
            "--param.int_value=42",
            "--param.float_value=3.14",
            "--param.bool_true=true",
            "--param.bool_false=false",
            "--param.string_value=hello",
            "--param.list_value=[1,2,3]",
            "--param.dict_value={key: value}",
        ]

        overrides = parse_override_args(args)

        assert isinstance(overrides["param.int_value"], int)
        assert overrides["param.int_value"] == 42

        assert isinstance(overrides["param.float_value"], float)
        assert overrides["param.float_value"] == 3.14

        assert isinstance(overrides["param.bool_true"], bool)
        assert overrides["param.bool_true"] == True

        assert isinstance(overrides["param.bool_false"], bool)
        assert overrides["param.bool_false"] == False

        assert isinstance(overrides["param.string_value"], str)
        assert overrides["param.string_value"] == "hello"

        assert isinstance(overrides["param.list_value"], list)
        assert overrides["param.list_value"] == [1, 2, 3]

        assert isinstance(overrides["param.dict_value"], dict)
        assert overrides["param.dict_value"] == {"key": "value"}

    def test_parse_quoted_strings(self):
        """Test parsing quoted string values."""
        args = [
            '--param.quoted="hello world"',
            "--param.single='single quotes'",
            "--param.path=/path/to/file",
        ]

        overrides = parse_override_args(args)

        assert overrides["param.quoted"] == "hello world"
        assert overrides["param.single"] == "single quotes"
        assert overrides["param.path"] == "/path/to/file"

    def test_parse_empty_args(self):
        """Test parsing empty argument list."""
        overrides = parse_override_args([])
        assert overrides == {}

    def test_parse_non_override_args_ignored(self):
        """Test that non-override arguments are ignored."""
        args = [
            "regular_arg",
            "-flag",
            "--model_params.lr=0.01",  # This should be parsed
            "another_arg",
        ]

        overrides = parse_override_args(args)

        # Only the override should be parsed
        assert overrides == {"model_params.lr": 0.01}

    def test_parse_malformed_args(self):
        """Test that malformed arguments are handled gracefully."""
        args = [
            "--no_equals_sign",  # No = sign
            "--model_params.lr=0.01",  # Valid
            "--=value",  # No key
            "--key=",  # No value
        ]

        overrides = parse_override_args(args)

        # Should only parse the valid one
        assert overrides == {"model_params.lr": 0.01}


class TestNestedAttributeHandling:
    """Test nested attribute getting and setting."""

    def test_get_nested_value_dict(self):
        """Test getting nested values from dictionary."""
        data = {"level1": {"level2": {"value": 42}}, "simple": "test"}

        assert get_nested_value(data, "simple") == "test"
        assert get_nested_value(data, "level1.level2.value") == 42

    def test_get_nested_value_dataclass(self, sample_experiment_config):
        """Test getting nested values from dataclass."""
        assert (
            get_nested_value(sample_experiment_config, "model_spec.constructor_name")
            == "test_model"
        )
        assert (
            get_nested_value(sample_experiment_config, "training_params.batch_size")
            == 32
        )
        assert (
            get_nested_value(sample_experiment_config, "model_spec.params.param2") == 42
        )

    def test_get_nested_value_mixed(self, sample_experiment_config):
        """Test getting nested values from mixed dict/dataclass structure."""
        # model_spec.params is a dict inside a dataclass
        assert (
            get_nested_value(sample_experiment_config, "model_spec.params.param1")
            == "value1"
        )

    def test_set_nested_attr_dict(self):
        """Test setting nested attributes in dictionary."""
        data = {"level1": {"level2": {"value": 0}}}

        set_nested_attr(data, "level1.level2.value", 42)

        assert data["level1"]["level2"]["value"] == 42

    def test_set_nested_attr_dataclass(self, sample_experiment_config):
        """Test setting nested attributes in dataclass."""
        original_batch_size = sample_experiment_config.training_params.batch_size

        set_nested_attr(sample_experiment_config, "training_params.batch_size", 128)

        assert sample_experiment_config.training_params.batch_size == 128
        assert (
            sample_experiment_config.training_params.batch_size != original_batch_size
        )

    def test_set_nested_attr_mixed(self, sample_experiment_config):
        """Test setting nested attributes in mixed dict/dataclass."""
        set_nested_attr(
            sample_experiment_config, "model_spec.params.new_param", "new_value"
        )

        assert sample_experiment_config.model_spec.params["new_param"] == "new_value"

    def test_set_nested_attr_create_path(self):
        """Test setting nested attributes creates intermediate dictionaries."""
        data = {}

        set_nested_attr(data, "level1.level2.value", 42)

        assert data["level1"]["level2"]["value"] == 42

    def test_nested_attr_error_handling(self):
        """Test error handling for invalid paths."""
        data = {"key": "not_a_dict"}

        with pytest.raises(TypeError):
            get_nested_value(data, "key.invalid_path")

        with pytest.raises(TypeError):
            set_nested_attr(data, "key.invalid_path", "value")


class TestApplyOverrides:
    """Test applying override dictionaries to config objects."""

    def test_apply_simple_override(self, sample_experiment_config):
        """Test applying simple override."""
        overrides = {"trial_name": "new_trial_name"}

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert modified_config.trial_name == "new_trial_name"
        # Original should be unchanged (deep copy)
        assert sample_experiment_config.trial_name == "test_experiment"

    def test_apply_nested_overrides(self, sample_experiment_config):
        """Test applying nested overrides."""
        overrides = {
            "training_params.batch_size": 256,
            "training_params.learning_rate": 0.001,
            "model_spec.params.new_param": "added_value",
        }

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert modified_config.training_params.batch_size == 256
        assert modified_config.training_params.learning_rate == 0.001
        assert modified_config.model_spec.params["new_param"] == "added_value"

        # Original should be unchanged
        assert sample_experiment_config.training_params.batch_size == 32

    def test_apply_multiple_overrides(self, sample_experiment_config):
        """Test applying multiple overrides at once."""
        overrides = {
            "model_spec.constructor_name": "new_model",
            "training_params.epochs": 200,
            "model_spec.params.param1": "modified_value",
        }

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert modified_config.model_spec.constructor_name == "new_model"
        assert modified_config.training_params.epochs == 200
        assert modified_config.model_spec.params["param1"] == "modified_value"

    def test_apply_empty_overrides(self, sample_experiment_config):
        """Test applying empty overrides dictionary."""
        modified_config = apply_overrides(sample_experiment_config, {})

        # Should be identical but different object
        assert modified_config is not sample_experiment_config
        assert (
            modified_config.model_spec.constructor_name
            == sample_experiment_config.model_spec.constructor_name
        )
        assert (
            modified_config.training_params.batch_size
            == sample_experiment_config.training_params.batch_size
        )

    def test_apply_overrides_preserves_structure(self, sample_experiment_config):
        """Test that applying overrides preserves the config structure."""
        overrides = {"training_params.batch_size": 1024}

        modified_config = apply_overrides(sample_experiment_config, overrides)

        # Structure should be preserved
        assert type(modified_config) == type(sample_experiment_config)
        assert type(modified_config.training_params) == type(
            sample_experiment_config.training_params
        )
        assert type(modified_config.model_spec) == type(
            sample_experiment_config.model_spec
        )

    def test_override_types_preserved(self, sample_experiment_config):
        """Test that override value types are preserved."""
        overrides = {
            "training_params.batch_size": 64,  # int
            "training_params.learning_rate": 0.005,  # float
            "model_spec.params.config": {"nested": True},  # dict
        }

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert isinstance(modified_config.training_params.batch_size, int)
        assert isinstance(modified_config.training_params.learning_rate, float)
        assert isinstance(modified_config.model_spec.params["config"], dict)


class TestLoadExperimentConfig:
    """Test load_experiment_config function for loading and processing complete experiment configs."""

    def test_load_task_specific_config(self, temp_task_config_file, mock_task_registry):
        """Test that task-specific config is correctly loaded and instantiated."""
        overrides = {}

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify task config was converted from dict to TaskConfig
        from core.config import TaskConfig

        assert isinstance(config.task_config, TaskConfig)
        assert config.task_config.task_name == "test_task"
        assert config.task_config.data_params.window_width == 0.5
        assert config.task_config.data_params.subject_ids == [1, 2, 3]

        # Verify task-specific config was instantiated correctly
        from tests.conftest import TestTaskConfig

        assert isinstance(config.task_config.task_specific_config, TestTaskConfig)
        assert config.task_config.task_specific_config.test_param == "test_value"
        assert config.task_config.task_specific_config.input_fields == [
            "field1",
            "field2",
        ]

    def test_electrode_file_overrides_per_subject_electrodes(
        self,
        temp_task_config_with_electrode_file,
        mock_task_registry,
        temp_electrode_file,
        temp_subject_mapping,
    ):
        """Test that electrode_file_path correctly overrides per_subject_electrodes."""
        overrides = {}

        config = load_experiment_config(
            temp_task_config_with_electrode_file,
            overrides,
            subject_mapping_file=temp_subject_mapping,
        )

        # Verify subject_ids were set from electrode file
        assert set(config.task_config.data_params.subject_ids) == {5, 12}

        # Verify per_subject_electrodes was populated correctly
        assert config.task_config.data_params.per_subject_electrodes is not None
        assert 5 in config.task_config.data_params.per_subject_electrodes
        assert 12 in config.task_config.data_params.per_subject_electrodes
        assert config.task_config.data_params.per_subject_electrodes[5] == [
            "A1",
            "A2",
            "B1",
        ]
        assert config.task_config.data_params.per_subject_electrodes[12] == ["C1", "C2"]

    def test_config_setter_name_with_required_setters(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test that config_setter_name correctly combines with required_config_setter_names."""
        overrides = {"config_setter_name": "user_setter"}

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify that required_config_setter_names are prepended to config_setter_name
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == [
            "required_setter1",
            "required_setter2",
            "user_setter",
        ]

    def test_config_setter_name_list_with_required_setters(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test that config_setter_name list correctly combines with required_config_setter_names."""
        overrides = {"config_setter_name": ["user_setter1", "user_setter2"]}

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify that required_config_setter_names are prepended to config_setter_name list
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == [
            "required_setter1",
            "required_setter2",
            "user_setter1",
            "user_setter2",
        ]

    def test_config_setter_name_only_required(
        self, temp_task_config_file_no_setter, mock_task_registry
    ):
        """Test that required_config_setter_names work when no config_setter_name is provided."""
        overrides = {}

        config = load_experiment_config(temp_task_config_file_no_setter, overrides)

        # Verify only required setters are present
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == ["required_setter1", "required_setter2"]

    def test_config_setter_name_no_required(
        self, temp_task_config_file_no_required_setters, mock_task_registry
    ):
        """Test that config_setter_name works when no required_config_setter_names are provided."""
        overrides = {"config_setter_name": "user_setter"}

        config = load_experiment_config(
            temp_task_config_file_no_required_setters, overrides
        )

        # Verify only user setter is present as a list
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == ["user_setter"]

    def test_no_config_setters(
        self, temp_task_config_file_no_setters_at_all, mock_task_registry
    ):
        """Test that config loads correctly when no config setters are provided at all."""
        overrides = {}

        config = load_experiment_config(
            temp_task_config_file_no_setters_at_all, overrides
        )

        # Verify config_setter_name remains None
        assert config.config_setter_name is None

    def test_overrides_applied_before_task_config_loading(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test that overrides are applied before task config is instantiated."""
        overrides = {
            "task_config.data_params.window_width": 1.5,
            "task_config.task_specific_config.test_param": "overridden_value",
        }

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify overrides were applied
        assert config.task_config.data_params.window_width == 1.5
        assert config.task_config.task_specific_config.test_param == "overridden_value"

    def test_nested_model_specs_correctly_set(
        self, temp_nested_model_config_file, mock_task_registry
    ):
        """Test that nested model specs can be correctly set via overrides."""
        # Test overriding nested model spec parameters
        overrides = {
            "model_spec.sub_models.encoder_model.params.input_channels": 128,
            "model_spec.sub_models.encoder_model.params.output_dim": 1024,
            "model_spec.params.freeze_lm": False,
        }

        config = load_experiment_config(temp_nested_model_config_file, overrides)

        # Verify top-level model spec
        assert config.model_spec.constructor_name == "gpt2_brain"
        assert config.model_spec.params["lm_model"] == "gpt2"
        assert config.model_spec.params["freeze_lm"] == False  # Overridden

        # Verify nested sub_model spec
        assert "encoder_model" in config.model_spec.sub_models
        encoder_spec = config.model_spec.sub_models["encoder_model"]
        assert encoder_spec.constructor_name == "pitom_model"
        assert encoder_spec.params["input_channels"] == 128  # Overridden
        assert encoder_spec.params["output_dim"] == 1024  # Overridden


class TestExperimentConfigCLIIntegration:
    """Test integration of load_experiment_config with CLI-style argument parsing."""

    def test_realistic_command_line_scenario(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test a realistic command-line override scenario with task config."""
        # Simulate args like: python main.py --config file.yml --model_spec.params.lr=0.01 --training_params.epochs=100
        unknown_args = [
            "--model_spec.params.learning_rate=0.01",
            "--training_params.epochs=100",
            "--training_params.batch_size=128",
            "--trial_name=command_line_test",
            "--task_config.data_params.window_width=1.0",
        ]

        overrides = parse_override_args(unknown_args)
        config = load_experiment_config(temp_task_config_file, overrides)

        # Should have overridden values from command line
        assert config.model_spec.params["learning_rate"] == 0.01
        assert config.training_params.epochs == 100
        assert config.training_params.batch_size == 128
        assert config.trial_name == "command_line_test"
        assert config.task_config.data_params.window_width == 1.0

        # Should still have original values from file for non-overridden fields
        assert config.model_spec.constructor_name == "test_model"
        assert config.model_spec.params["hidden_dim"] == 256
        assert config.training_params.learning_rate == 0.001
        assert config.task_config.task_name == "test_task"

    def test_override_precedence(self, temp_task_config_file, mock_task_registry):
        """Test that command-line overrides take precedence over file values."""
        # Both file and command line specify batch_size and window_width
        overrides = {
            "training_params.batch_size": 999,
            "task_config.data_params.window_width": 2.5,
        }

        config = load_experiment_config(temp_task_config_file, overrides)

        # Command line should win
        assert config.training_params.batch_size == 999
        assert config.task_config.data_params.window_width == 2.5
        # Other file values should remain
        assert config.training_params.learning_rate == 0.001
        assert config.task_config.data_params.subject_ids == [1, 2, 3]

    def test_override_task_specific_config_fields(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test overriding fields in task-specific config via CLI."""
        unknown_args = [
            "--task_config.task_specific_config.test_param=cli_override",
            "--task_config.task_specific_config.input_fields=[field3,field4,field5]",
        ]

        overrides = parse_override_args(unknown_args)
        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify task-specific config overrides were applied
        assert config.task_config.task_specific_config.test_param == "cli_override"
        assert config.task_config.task_specific_config.input_fields == [
            "field3",
            "field4",
            "field5",
        ]

    def test_override_with_complex_types(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test CLI overrides with complex data types (lists, dicts)."""
        unknown_args = [
            "--model_spec.params.layer_sizes=[512,256,128]",
            "--training_params.metrics=[mse,cosine_sim,nll_embedding]",
            "--task_config.data_params.subject_ids=[5,6,7,8,9]",
        ]

        overrides = parse_override_args(unknown_args)
        config = load_experiment_config(temp_task_config_file, overrides)

        assert config.model_spec.params["layer_sizes"] == [512, 256, 128]
        assert config.training_params.metrics == ["mse", "cosine_sim", "nll_embedding"]
        assert config.task_config.data_params.subject_ids == [5, 6, 7, 8, 9]

    def test_override_config_setter_name_via_cli(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test overriding config_setter_name via CLI arguments."""
        unknown_args = ["--config_setter_name=cli_setter"]

        overrides = parse_override_args(unknown_args)
        config = load_experiment_config(temp_task_config_file, overrides)

        # Should have required setters prepended to CLI-provided setter
        assert config.config_setter_name == [
            "required_setter1",
            "required_setter2",
            "cli_setter",
        ]

    def test_multiple_override_sources(self, temp_task_config_file, mock_task_registry):
        """Test combining overrides from multiple sources (simulating mixed CLI args)."""
        # First set of args
        args_batch1 = [
            "--training_params.batch_size=64",
            "--model_spec.params.dropout=0.3",
        ]
        overrides1 = parse_override_args(args_batch1)

        # Second set of args (simulating user adding more)
        args_batch2 = ["--task_config.data_params.window_width=0.75"]
        overrides2 = parse_override_args(args_batch2)

        # Combine overrides
        combined_overrides = {**overrides1, **overrides2}

        config = load_experiment_config(temp_task_config_file, combined_overrides)

        # All overrides should be applied
        assert config.training_params.batch_size == 64
        assert config.model_spec.params["dropout"] == 0.3
        assert config.task_config.data_params.window_width == 0.75


class TestSharedParamsWithApplyOverrides:
    """Test that apply_overrides works for shared params use case."""

    def test_apply_overrides_as_shared_params(self, sample_experiment_config):
        """Test that apply_overrides can be used for shared params."""
        shared_params = {
            "training_params.n_folds": 5,
            "training_params.min_lag": 0,
            "training_params.max_lag": 400,
        }

        modified_config = apply_overrides(sample_experiment_config, shared_params)

        assert modified_config.training_params.n_folds == 5
        assert modified_config.training_params.min_lag == 0
        assert modified_config.training_params.max_lag == 400


class TestLoadMultiTaskConfig:
    """Test loading multi-task configurations from YAML files."""

    def test_load_basic_multi_task_config(
        self, temp_multi_task_config_file, mock_task_registry
    ):
        """Test loading basic multi-task config with two tasks."""
        overrides = {}

        multi_config = load_multi_task_config(temp_multi_task_config_file, overrides)

        # Should return MultiTaskConfig
        assert isinstance(multi_config, MultiTaskConfig)

        # Should have 2 tasks
        assert len(multi_config.tasks) == 2

        # Each task should be an ExperimentConfig
        assert all(isinstance(task, ExperimentConfig) for task in multi_config.tasks)

        # Verify first task
        assert multi_config.tasks[0].trial_name == "task1_pretrain"
        assert multi_config.tasks[0].training_params.batch_size == 32

        # Verify second task
        assert multi_config.tasks[1].trial_name == "task2_finetune"
        assert multi_config.tasks[1].training_params.batch_size == 16

    def test_load_multi_task_with_shared_params(
        self, temp_multi_task_config_with_shared, mock_task_registry
    ):
        """Test that shared_params are loaded and stored in MultiTaskConfig.

        Note: Shared params are NOT applied during load_multi_task_config - they are
        stored and applied later by main.py when running tasks. This gives main.py
        control over when shared params are applied (after config setters).
        """
        overrides = {}

        multi_config = load_multi_task_config(
            temp_multi_task_config_with_shared, overrides
        )

        # Should have shared_params stored
        assert multi_config.shared_params is not None
        assert multi_config.shared_params["training_params.n_folds"] == 5
        assert multi_config.shared_params["training_params.min_lag"] == 0
        assert multi_config.shared_params["training_params.max_lag"] == 400

        # Shared params should NOT be applied yet - tasks should have original values
        # Task 1 has batch_size: 32 (not overridden yet)
        assert multi_config.tasks[0].training_params.batch_size == 32
        # Task 2 has batch_size: 16 (not overridden yet)
        assert multi_config.tasks[1].training_params.batch_size == 16

    def test_shared_params_can_be_applied_with_apply_overrides(
        self, temp_multi_task_config_with_shared, mock_task_registry
    ):
        """Test that shared_params can be applied to tasks using apply_overrides.

        This demonstrates how main.py will use shared_params after loading.
        """
        overrides = {}

        multi_config = load_multi_task_config(
            temp_multi_task_config_with_shared, overrides
        )

        # Apply shared_params to each task using apply_overrides
        if multi_config.shared_params:
            for i, task in enumerate(multi_config.tasks):
                multi_config.tasks[i] = apply_overrides(
                    task, multi_config.shared_params
                )

        # Now shared params should be applied to all tasks
        assert multi_config.tasks[0].training_params.n_folds == 5
        assert multi_config.tasks[0].training_params.min_lag == 0
        assert multi_config.tasks[0].training_params.max_lag == 400

        assert multi_config.tasks[1].training_params.n_folds == 5
        assert multi_config.tasks[1].training_params.min_lag == 0
        assert multi_config.tasks[1].training_params.max_lag == 400

        # Task-specific values should be preserved
        assert multi_config.tasks[0].training_params.batch_size == 32
        assert multi_config.tasks[1].training_params.batch_size == 16

    def test_load_multi_task_no_shared_params(
        self, temp_multi_task_config_file, mock_task_registry
    ):
        """Test loading multi-task config without shared_params."""
        overrides = {}

        multi_config = load_multi_task_config(temp_multi_task_config_file, overrides)

        # shared_params should be None or empty
        assert multi_config.shared_params is None or multi_config.shared_params == {}

    def test_load_multi_task_applies_cli_overrides(
        self, temp_multi_task_config_file, mock_task_registry
    ):
        """Test that command-line overrides are applied during loading."""
        # Override fields in individual tasks
        overrides = {
            "tasks.0.training_params.batch_size": 64,
            "tasks.1.trial_name": "overridden_name",
        }

        multi_config = load_multi_task_config(temp_multi_task_config_file, overrides)

        # CLI overrides should be applied to raw YAML before conversion
        assert multi_config.tasks[0].training_params.batch_size == 64
        assert multi_config.tasks[1].trial_name == "overridden_name"

    def test_load_multi_task_with_nested_model_specs(
        self, temp_multi_task_nested_models, mock_task_registry
    ):
        """Test loading multi-task config with nested model specs."""
        overrides = {}

        multi_config = load_multi_task_config(temp_multi_task_nested_models, overrides)

        # Second task should have nested model spec with checkpoint_path
        task2 = multi_config.tasks[1]
        assert "encoder_model" in task2.model_spec.sub_models
        encoder = task2.model_spec.sub_models["encoder_model"]
        assert "{prev_checkpoint_dir}" in encoder.checkpoint_path

    def test_load_multi_task_finalizes_task_configs(
        self, temp_multi_task_config_file, mock_task_registry
    ):
        """Test that each task's task_config is properly finalized."""
        overrides = {}

        multi_config = load_multi_task_config(temp_multi_task_config_file, overrides)

        # Each task should have properly typed TaskConfig
        for task in multi_config.tasks:
            assert isinstance(task.task_config, TaskConfig)
            assert task.task_config.task_name == "test_task"
            # Task-specific config should be instantiated from TestTaskConfig
            from tests.conftest import TestTaskConfig

            assert isinstance(task.task_config.task_specific_config, TestTaskConfig)

    def test_load_multi_task_processes_config_setters(
        self, temp_multi_task_with_config_setters, mock_task_registry
    ):
        """Test that config setters are properly combined for each task."""
        overrides = {}

        multi_config = load_multi_task_config(
            temp_multi_task_with_config_setters, overrides
        )

        # Each task should have required_config_setter_names processed
        for task in multi_config.tasks:
            # Should have required setters in the list
            assert task.config_setter_name is not None
            assert isinstance(task.config_setter_name, list)
            assert "required_setter1" in task.config_setter_name


class TestLoadConfig:
    """Test auto-detection of single vs multi-task configs."""

    def test_load_config_detects_single_task(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test that load_config correctly detects and loads single-task config."""
        overrides = {}

        config = load_config(temp_task_config_file, overrides)

        # Should return ExperimentConfig
        assert isinstance(config, ExperimentConfig)
        assert config.trial_name == "test_with_task_config"

    def test_load_config_detects_multi_task(
        self, temp_multi_task_config_file, mock_task_registry
    ):
        """Test that load_config correctly detects and loads multi-task config."""
        overrides = {}

        config = load_config(temp_multi_task_config_file, overrides)

        # Should return MultiTaskConfig
        assert isinstance(config, MultiTaskConfig)
        assert len(config.tasks) == 2

    def test_load_config_preserves_overrides_single(
        self, temp_task_config_file, mock_task_registry
    ):
        """Test that load_config preserves overrides for single-task."""
        overrides = {"trial_name": "cli_override"}

        config = load_config(temp_task_config_file, overrides)

        assert isinstance(config, ExperimentConfig)
        assert config.trial_name == "cli_override"

    def test_load_config_preserves_overrides_multi(
        self, temp_multi_task_config_file, mock_task_registry
    ):
        """Test that load_config preserves overrides for multi-task."""
        overrides = {"tasks.0.trial_name": "multi_override"}

        config = load_config(temp_multi_task_config_file, overrides)

        assert isinstance(config, MultiTaskConfig)
        assert config.tasks[0].trial_name == "multi_override"


class TestMultiTaskConfigValidation:
    """Test validation of MultiTaskConfig."""

    def test_validation_rejects_empty_tasks(self):
        """Test that empty tasks list is rejected."""
        from core.config import MultiTaskConfig

        multi_config = MultiTaskConfig(tasks=[], shared_params=None)

        with pytest.raises(ValueError, match="at least one task"):
            validate_multi_task_config(multi_config)

    def test_validation_rejects_duplicate_trial_names(self, sample_experiment_config):
        """Test that duplicate trial_names are rejected."""
        from core.config import MultiTaskConfig

        task1 = deepcopy(sample_experiment_config)
        task1.trial_name = "duplicate_name"

        task2 = deepcopy(sample_experiment_config)
        task2.trial_name = "duplicate_name"

        multi_config = MultiTaskConfig(tasks=[task1, task2], shared_params=None)

        with pytest.raises(ValueError, match="trial_names must be unique"):
            validate_multi_task_config(multi_config)

    def test_validation_allows_empty_trial_names(self, sample_experiment_config):
        """Test that empty trial names don't trigger duplicate check."""
        from core.config import MultiTaskConfig

        task1 = deepcopy(sample_experiment_config)
        task1.trial_name = ""

        task2 = deepcopy(sample_experiment_config)
        task2.trial_name = ""

        multi_config = MultiTaskConfig(tasks=[task1, task2], shared_params=None)

        # Should not raise - empty names are ignored in uniqueness check
        validate_multi_task_config(multi_config)

    def test_validation_accepts_valid_config(self, sample_experiment_config):
        """Test that valid multi-task config passes validation."""
        from core.config import MultiTaskConfig

        task1 = deepcopy(sample_experiment_config)
        task1.trial_name = "task1"

        task2 = deepcopy(sample_experiment_config)
        task2.trial_name = "task2"

        multi_config = MultiTaskConfig(
            tasks=[task1, task2], shared_params={"training_params.n_folds": 5}
        )

        # Should not raise
        validate_multi_task_config(multi_config)


class TestPartialFormat:
    """Test partial_format function for partial string formatting."""

    def test_format_with_all_variables_provided(self):
        """Test formatting when all variables are provided."""
        template = "{a} and {b}"
        result = partial_format(template, a="hello", b="world")
        assert result == "hello and world"

    def test_format_with_some_variables_missing(self):
        """Test formatting when some variables are missing - they should be preserved."""
        template = "{a} and {b}"
        result = partial_format(template, a="hello")
        assert result == "hello and {b}"

    def test_format_preserves_multiple_missing_variables(self):
        """Test that multiple missing variables are preserved."""
        template = "{prev_checkpoint_dir}/lag_{lag}/fold_{fold}"
        result = partial_format(template, prev_checkpoint_dir="/path/to/checkpoint")
        assert result == "/path/to/checkpoint/lag_{lag}/fold_{fold}"

    def test_format_with_no_variables_provided(self):
        """Test formatting when no variables are provided - all should be preserved."""
        template = "{a} and {b} and {c}"
        result = partial_format(template)
        assert result == "{a} and {b} and {c}"

    def test_format_with_repeated_variables(self):
        """Test formatting with repeated variable names."""
        template = "{x} plus {x} equals two {x}"
        result = partial_format(template, x="5")
        assert result == "5 plus 5 equals two 5"

    def test_format_empty_string(self):
        """Test formatting an empty string."""
        result = partial_format("")
        assert result == ""

    def test_format_string_with_no_variables(self):
        """Test formatting a string with no format variables."""
        template = "plain text with no variables"
        result = partial_format(template, unused="value")
        assert result == "plain text with no variables"

    def test_format_with_numeric_values(self):
        """Test formatting with numeric values."""
        template = "fold_{fold}_epoch_{epoch}"
        result = partial_format(template, fold=3)
        assert result == "fold_3_epoch_{epoch}"

    def test_format_complex_path(self):
        """Test formatting a complex file path."""
        template = "{base_dir}/experiments/{experiment}/lag_{lag}/fold_{fold}/checkpoint.pt"
        result = partial_format(
            template,
            base_dir="/home/user/data",
            experiment="exp_001"
        )
        assert result == "/home/user/data/experiments/exp_001/lag_{lag}/fold_{fold}/checkpoint.pt"

    def test_format_consecutive_variables(self):
        """Test formatting with consecutive variables."""
        template = "{a}{b}{c}"
        result = partial_format(template, a="x", c="z")
        assert result == "x{b}z"


class TestInterpolatePrevCheckpointDir:
    """Test interpolate_prev_checkpoint_dir function for checkpoint path interpolation."""

    def test_interpolate_simple_checkpoint_path(self):
        """Test interpolating prev_checkpoint_dir in a simple checkpoint path."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="{prev_checkpoint_dir}/model.pt"
        )

        result = interpolate_prev_checkpoint_dir(spec, "/path/to/prev")

        assert result.checkpoint_path == "/path/to/prev/model.pt"
        # Original should be unchanged (deep copy)
        assert spec.checkpoint_path == "{prev_checkpoint_dir}/model.pt"

    def test_interpolate_with_other_variables_preserved(self):
        """Test that other format variables like {lag} and {fold} are preserved."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="{prev_checkpoint_dir}/lag_{lag}/best_model_fold{fold}.pt"
        )

        result = interpolate_prev_checkpoint_dir(spec, "checkpoints/pretrain/run_123")

        assert result.checkpoint_path == "checkpoints/pretrain/run_123/lag_{lag}/best_model_fold{fold}.pt"

    def test_interpolate_without_prev_checkpoint_dir_variable(self):
        """Test that paths without {prev_checkpoint_dir} are unchanged."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="/absolute/path/model.pt"
        )

        result = interpolate_prev_checkpoint_dir(spec, "/path/to/prev")

        assert result.checkpoint_path == "/absolute/path/model.pt"

    def test_interpolate_with_none_checkpoint_path(self):
        """Test handling of None checkpoint_path."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path=None
        )

        result = interpolate_prev_checkpoint_dir(spec, "/path/to/prev")

        assert result.checkpoint_path is None

    def test_interpolate_raises_on_missing_prev_checkpoint_dir(self):
        """Test that ValueError is raised when prev_checkpoint_dir is needed but not provided."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="{prev_checkpoint_dir}/model.pt"
        )

        with pytest.raises(ValueError, match="no previous checkpoint directory available"):
            interpolate_prev_checkpoint_dir(spec, None)

    def test_interpolate_raises_on_empty_prev_checkpoint_dir(self):
        """Test that ValueError is raised when prev_checkpoint_dir is empty string."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="{prev_checkpoint_dir}/model.pt"
        )

        with pytest.raises(ValueError, match="no previous checkpoint directory available"):
            interpolate_prev_checkpoint_dir(spec, "")

    def test_interpolate_recursive_sub_models(self):
        """Test that sub_models are recursively processed."""
        encoder_spec = ModelSpec(
            constructor_name="encoder",
            checkpoint_path="{prev_checkpoint_dir}/encoder.pt"
        )

        main_spec = ModelSpec(
            constructor_name="main_model",
            checkpoint_path="{prev_checkpoint_dir}/main.pt",
            sub_models={"encoder": encoder_spec}
        )

        result = interpolate_prev_checkpoint_dir(main_spec, "/checkpoints/task1")

        assert result.checkpoint_path == "/checkpoints/task1/main.pt"
        assert result.sub_models["encoder"].checkpoint_path == "/checkpoints/task1/encoder.pt"
        # Original should be unchanged
        assert main_spec.sub_models["encoder"].checkpoint_path == "{prev_checkpoint_dir}/encoder.pt"

    def test_interpolate_nested_sub_models_with_mixed_paths(self):
        """Test recursive interpolation with some paths having {prev_checkpoint_dir} and others not."""
        encoder_spec = ModelSpec(
            constructor_name="encoder",
            checkpoint_path="{prev_checkpoint_dir}/lag_{lag}/encoder_fold{fold}.pt"
        )

        decoder_spec = ModelSpec(
            constructor_name="decoder",
            checkpoint_path="/absolute/path/decoder.pt"  # No prev_checkpoint_dir
        )

        main_spec = ModelSpec(
            constructor_name="main_model",
            checkpoint_path="{prev_checkpoint_dir}/main.pt",
            sub_models={"encoder": encoder_spec, "decoder": decoder_spec}
        )

        result = interpolate_prev_checkpoint_dir(main_spec, "/checkpoints/pretrain")

        assert result.checkpoint_path == "/checkpoints/pretrain/main.pt"
        assert result.sub_models["encoder"].checkpoint_path == "/checkpoints/pretrain/lag_{lag}/encoder_fold{fold}.pt"
        assert result.sub_models["decoder"].checkpoint_path == "/absolute/path/decoder.pt"

    def test_interpolate_empty_sub_models(self):
        """Test that empty sub_models dict doesn't cause issues."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="{prev_checkpoint_dir}/model.pt",
            sub_models={}
        )

        result = interpolate_prev_checkpoint_dir(spec, "/checkpoints/prev")

        assert result.checkpoint_path == "/checkpoints/prev/model.pt"
        assert result.sub_models == {}

    def test_interpolate_none_sub_models(self):
        """Test that None sub_models doesn't cause issues."""
        spec = ModelSpec(
            constructor_name="test_model",
            checkpoint_path="{prev_checkpoint_dir}/model.pt",
            sub_models=None
        )

        result = interpolate_prev_checkpoint_dir(spec, "/checkpoints/prev")

        assert result.checkpoint_path == "/checkpoints/prev/model.pt"
        assert result.sub_models is None


# ============================================================================
# Fixtures for Multi-Task Tests
# ============================================================================


@pytest.fixture
def temp_multi_task_config_file():
    """Create a temporary multi-task YAML config file."""
    config_content = """
tasks:
  - trial_name: task1_pretrain
    model_spec:
      constructor_name: test_model
      params:
        hidden_dim: 256
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2, 3]
      task_specific_config:
        test_param: task1_value
    training_params:
      batch_size: 32
      learning_rate: 0.001
      epochs: 20
    output_dir: results/task1
    checkpoint_dir: checkpoints/task1

  - trial_name: task2_finetune
    model_spec:
      constructor_name: test_model
      params:
        hidden_dim: 512
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2, 3]
      task_specific_config:
        test_param: task2_value
    training_params:
      batch_size: 16
      learning_rate: 0.0001
      epochs: 50
    output_dir: results/task2
    checkpoint_dir: checkpoints/task2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_multi_task_config_with_shared():
    """Create a multi-task config with shared_params."""
    config_content = """
tasks:
  - trial_name: task1
    model_spec:
      constructor_name: test_model
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2]
      task_specific_config:
        test_param: value1
    training_params:
      batch_size: 32
    output_dir: results/task1
    checkpoint_dir: checkpoints/task1

  - trial_name: task2
    model_spec:
      constructor_name: test_model
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2]
      task_specific_config:
        test_param: value2
    training_params:
      batch_size: 16
    output_dir: results/task2
    checkpoint_dir: checkpoints/task2

shared_params:
  training_params.n_folds: 5
  training_params.min_lag: 0
  training_params.max_lag: 400
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_multi_task_nested_models():
    """Create a multi-task config with nested model specs."""
    config_content = """
tasks:
  - trial_name: pretrain_encoder
    model_spec:
      constructor_name: encoder_model
      params:
        embedding_dim: 768
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2]
      task_specific_config:
        test_param: encoder_pretrain
    training_params:
      batch_size: 32
    output_dir: results/pretrain
    checkpoint_dir: checkpoints/pretrain

  - trial_name: finetune_full
    model_spec:
      constructor_name: gpt2_brain
      params:
        freeze_lm: true
      sub_models:
        encoder_model:
          constructor_name: encoder_model
          params:
            embedding_dim: 768
          checkpoint_path: "{prev_checkpoint_dir}/lag_{lag}/best_model_fold{fold}.pt"
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2]
      task_specific_config:
        test_param: finetune
    training_params:
      batch_size: 16
    output_dir: results/finetune
    checkpoint_dir: checkpoints/finetune
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_multi_task_with_config_setters():
    """Create a multi-task config where tasks have required_config_setter_names."""
    config_content = """
tasks:
  - trial_name: task1
    model_spec:
      constructor_name: test_model
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2]
      task_specific_config:
        test_param: value1
        required_config_setter_names: [required_setter1, required_setter2]
    training_params:
      batch_size: 32
    output_dir: results/task1
    checkpoint_dir: checkpoints/task1

  - trial_name: task2
    model_spec:
      constructor_name: test_model
    task_config:
      task_name: test_task
      data_params:
        window_width: 0.5
        subject_ids: [1, 2]
      task_specific_config:
        test_param: value2
        required_config_setter_names: [required_setter1]
    training_params:
      batch_size: 16
    output_dir: results/task2
    checkpoint_dir: checkpoints/task2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
