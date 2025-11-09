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
    load_config_with_overrides,
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
            get_nested_value(sample_experiment_config, "model_constructor_name")
            == "test_model"
        )
        assert (
            get_nested_value(sample_experiment_config, "training_params.batch_size")
            == 32
        )
        assert (
            get_nested_value(sample_experiment_config, "data_params.embedding_type")
            == "gpt-2xl"
        )
        assert get_nested_value(sample_experiment_config, "model_params.param2") == 42

    def test_get_nested_value_mixed(self, sample_experiment_config):
        """Test getting nested values from mixed dict/dataclass structure."""
        # model_params is a dict inside a dataclass
        assert (
            get_nested_value(sample_experiment_config, "model_params.param1")
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
        set_nested_attr(sample_experiment_config, "model_params.new_param", "new_value")

        assert sample_experiment_config.model_params["new_param"] == "new_value"

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
            "data_params.embedding_type": "glove",
            "model_params.new_param": "added_value",
        }

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert modified_config.training_params.batch_size == 256
        assert modified_config.training_params.learning_rate == 0.001
        assert modified_config.data_params.embedding_type == "glove"
        assert modified_config.model_params["new_param"] == "added_value"

        # Original should be unchanged
        assert sample_experiment_config.training_params.batch_size == 32
        assert sample_experiment_config.data_params.embedding_type == "gpt-2xl"

    def test_apply_multiple_overrides(self, sample_experiment_config):
        """Test applying multiple overrides at once."""
        overrides = {
            "model_constructor_name": "new_model",
            "training_params.epochs": 200,
            "data_params.subject_ids": [5, 6, 7],
            "model_params.param1": "modified_value",
        }

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert modified_config.model_constructor_name == "new_model"
        assert modified_config.training_params.epochs == 200
        assert modified_config.data_params.subject_ids == [5, 6, 7]
        assert modified_config.model_params["param1"] == "modified_value"

    def test_apply_empty_overrides(self, sample_experiment_config):
        """Test applying empty overrides dictionary."""
        modified_config = apply_overrides(sample_experiment_config, {})

        # Should be identical but different object
        assert modified_config is not sample_experiment_config
        assert (
            modified_config.model_constructor_name
            == sample_experiment_config.model_constructor_name
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
        assert type(modified_config.data_params) == type(
            sample_experiment_config.data_params
        )

    def test_override_types_preserved(self, sample_experiment_config):
        """Test that override value types are preserved."""
        overrides = {
            "training_params.batch_size": 64,  # int
            "training_params.learning_rate": 0.005,  # float
            "data_params.subject_ids": [1, 2, 3, 4, 5],  # list
            "model_params.config": {"nested": True},  # dict
        }

        modified_config = apply_overrides(sample_experiment_config, overrides)

        assert isinstance(modified_config.training_params.batch_size, int)
        assert isinstance(modified_config.training_params.learning_rate, float)
        assert isinstance(modified_config.data_params.subject_ids, list)
        assert isinstance(modified_config.model_params["config"], dict)


class TestLoadConfigWithOverrides:
    """Test loading config from file with overrides applied."""

    def test_load_config_with_overrides(self, temp_config_file):
        """Test loading config file and applying overrides."""
        overrides = {
            "model_constructor_name": "overridden_model",
            "training_params.batch_size": 256,
            "model_params.new_param": "added",
        }

        config = load_config_with_overrides(temp_config_file, overrides)

        # Should have original values from file
        assert config.config_setter_name == "test_setter"
        assert config.trial_name == "temp_test"
        assert config.data_params.embedding_type == "gpt-2xl"

        # Should have overridden values
        assert config.model_constructor_name == "overridden_model"
        assert config.training_params.batch_size == 256
        assert config.model_params["new_param"] == "added"

        # Non-overridden values should remain from file
        assert config.model_params["hidden_dim"] == 256
        assert config.training_params.learning_rate == 0.001

    def test_load_config_no_overrides(self, temp_config_file):
        """Test loading config file without overrides."""
        config = load_config_with_overrides(temp_config_file, {})

        # Should match file contents exactly
        assert config.model_constructor_name == "test_model"
        assert config.config_setter_name == "test_setter"
        assert config.model_params["hidden_dim"] == 256
        assert config.training_params.batch_size == 64
        assert config.trial_name == "temp_test"

    def test_complex_override_scenarios(self, temp_config_file):
        """Test complex override scenarios with various data types."""
        overrides = {
            "model_params.layer_sizes": [512, 256, 128],
            "training_params.metrics": ["mse", "cosine_sim", "nll_embedding"],
            "data_params.preprocessor_params": {"new_param": True, "value": 3.14},
            "format_fields": ["model_params.hidden_dim", "training_params.batch_size"],
        }

        config = load_config_with_overrides(temp_config_file, overrides)

        assert config.model_params["layer_sizes"] == [512, 256, 128]
        assert config.training_params.metrics == ["mse", "cosine_sim", "nll_embedding"]
        assert config.data_params.preprocessor_params["new_param"] == True
        assert config.data_params.preprocessor_params["value"] == 3.14
        assert config.format_fields == [
            "model_params.hidden_dim",
            "training_params.batch_size",
        ]


class TestArgumentParsingIntegration:
    """Test integration with argparse-style argument parsing."""

    def test_realistic_command_line_scenario(self, temp_config_file):
        """Test a realistic command-line override scenario."""
        # Simulate args like: python main.py --config file.yml --model_params.lr=0.01 --training_params.epochs=100
        unknown_args = [
            "--model_params.learning_rate=0.01",
            "--training_params.epochs=100",
            "--training_params.batch_size=128",
            "--data_params.subject_ids=[1,2,3,4]",
            "--trial_name=command_line_test",
        ]

        overrides = parse_override_args(unknown_args)
        config = load_config_with_overrides(temp_config_file, overrides)

        # Should combine file config with command-line overrides
        assert config.model_constructor_name == "test_model"  # from file
        assert config.model_params["learning_rate"] == 0.01  # from command line
        assert (
            config.training_params.epochs == 100
        )  # from command line (overrides file's 20)
        assert (
            config.training_params.batch_size == 128
        )  # from command line (overrides file's 64)
        assert config.data_params.subject_ids == [1, 2, 3, 4]  # from command line
        assert config.trial_name == "command_line_test"  # from command line

    def test_override_precedence(self, temp_config_file):
        """Test that command-line overrides take precedence over file values."""
        # Both file and command line specify batch_size
        overrides = {"training_params.batch_size": 999}

        config = load_config_with_overrides(temp_config_file, overrides)

        # Command line should win
        assert config.training_params.batch_size == 999
        # Other file values should remain
        assert config.training_params.learning_rate == 0.001
