"""
Tests for command-line parsing and config override functionality from main.py.

Tests argument parsing, config override application, and nested attribute handling.
"""

import pytest

from utils.config_utils import (
    parse_override_args,
    apply_overrides,
    set_nested_attr,
    get_nested_value,
    load_experiment_config,
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
        assert get_nested_value(sample_experiment_config, "model_spec.params.param2") == 42

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
        set_nested_attr(sample_experiment_config, "model_spec.params.new_param", "new_value")

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
        assert config.task_config.task_specific_config.input_fields == ["field1", "field2"]

    def test_electrode_file_overrides_per_subject_electrodes(
        self, temp_task_config_with_electrode_file, mock_task_registry,
        temp_electrode_file, temp_subject_mapping
    ):
        """Test that electrode_file_path correctly overrides per_subject_electrodes."""
        overrides = {}

        config = load_experiment_config(
            temp_task_config_with_electrode_file,
            overrides,
            subject_mapping_file=temp_subject_mapping
        )

        # Verify subject_ids were set from electrode file
        assert set(config.task_config.data_params.subject_ids) == {5, 12}

        # Verify per_subject_electrodes was populated correctly
        assert config.task_config.data_params.per_subject_electrodes is not None
        assert 5 in config.task_config.data_params.per_subject_electrodes
        assert 12 in config.task_config.data_params.per_subject_electrodes
        assert config.task_config.data_params.per_subject_electrodes[5] == ["A1", "A2", "B1"]
        assert config.task_config.data_params.per_subject_electrodes[12] == ["C1", "C2"]

    def test_config_setter_name_with_required_setters(self, temp_task_config_file, mock_task_registry):
        """Test that config_setter_name correctly combines with required_config_setter_names."""
        overrides = {"config_setter_name": "user_setter"}

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify that required_config_setter_names are prepended to config_setter_name
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == ["required_setter1", "required_setter2", "user_setter"]

    def test_config_setter_name_list_with_required_setters(self, temp_task_config_file, mock_task_registry):
        """Test that config_setter_name list correctly combines with required_config_setter_names."""
        overrides = {"config_setter_name": ["user_setter1", "user_setter2"]}

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify that required_config_setter_names are prepended to config_setter_name list
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == ["required_setter1", "required_setter2", "user_setter1", "user_setter2"]

    def test_config_setter_name_only_required(self, temp_task_config_file_no_setter, mock_task_registry):
        """Test that required_config_setter_names work when no config_setter_name is provided."""
        overrides = {}

        config = load_experiment_config(temp_task_config_file_no_setter, overrides)

        # Verify only required setters are present
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == ["required_setter1", "required_setter2"]

    def test_config_setter_name_no_required(self, temp_task_config_file_no_required_setters, mock_task_registry):
        """Test that config_setter_name works when no required_config_setter_names are provided."""
        overrides = {"config_setter_name": "user_setter"}

        config = load_experiment_config(temp_task_config_file_no_required_setters, overrides)

        # Verify only user setter is present as a list
        assert isinstance(config.config_setter_name, list)
        assert config.config_setter_name == ["user_setter"]

    def test_no_config_setters(self, temp_task_config_file_no_setters_at_all, mock_task_registry):
        """Test that config loads correctly when no config setters are provided at all."""
        overrides = {}

        config = load_experiment_config(temp_task_config_file_no_setters_at_all, overrides)

        # Verify config_setter_name remains None
        assert config.config_setter_name is None

    def test_overrides_applied_before_task_config_loading(self, temp_task_config_file, mock_task_registry):
        """Test that overrides are applied before task config is instantiated."""
        overrides = {
            "task_config.data_params.window_width": 1.5,
            "task_config.task_specific_config.test_param": "overridden_value"
        }

        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify overrides were applied
        assert config.task_config.data_params.window_width == 1.5
        assert config.task_config.task_specific_config.test_param == "overridden_value"


class TestExperimentConfigCLIIntegration:
    """Test integration of load_experiment_config with CLI-style argument parsing."""

    def test_realistic_command_line_scenario(self, temp_task_config_file, mock_task_registry):
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

    def test_override_task_specific_config_fields(self, temp_task_config_file, mock_task_registry):
        """Test overriding fields in task-specific config via CLI."""
        unknown_args = [
            "--task_config.task_specific_config.test_param=cli_override",
            "--task_config.task_specific_config.input_fields=[field3,field4,field5]",
        ]

        overrides = parse_override_args(unknown_args)
        config = load_experiment_config(temp_task_config_file, overrides)

        # Verify task-specific config overrides were applied
        assert config.task_config.task_specific_config.test_param == "cli_override"
        assert config.task_config.task_specific_config.input_fields == ["field3", "field4", "field5"]

    def test_override_with_complex_types(self, temp_task_config_file, mock_task_registry):
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

    def test_override_config_setter_name_via_cli(self, temp_task_config_file, mock_task_registry):
        """Test overriding config_setter_name via CLI arguments."""
        unknown_args = ["--config_setter_name=cli_setter"]

        overrides = parse_override_args(unknown_args)
        config = load_experiment_config(temp_task_config_file, overrides)

        # Should have required setters prepended to CLI-provided setter
        assert config.config_setter_name == ["required_setter1", "required_setter2", "cli_setter"]

    def test_multiple_override_sources(self, temp_task_config_file, mock_task_registry):
        """Test combining overrides from multiple sources (simulating mixed CLI args)."""
        # First set of args
        args_batch1 = ["--training_params.batch_size=64", "--model_spec.params.dropout=0.3"]
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
