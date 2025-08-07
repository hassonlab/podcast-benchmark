"""
Tests for the configuration system (config.py).

Tests the YAML config loading, dataclass conversion, and field validation.
"""

import pytest
import yaml
from dataclasses import asdict

from config import DataParams, TrainingParams, ExperimentConfig, dict_to_config


@pytest.fixture
def sample_data_params():
    """Sample DataParams configuration for testing."""
    return DataParams(
        embedding_type="gpt-2xl",
        window_width=0.5,
        preprocessing_fn_name="test_preprocessor",
        subject_ids=[1, 2],
        data_root="test_data",
        embedding_pca_dim=50,
        channel_reg_ex=None,
        embedding_layer=24,
        preprocessor_params={"param1": "value1"},
    )


@pytest.fixture
def sample_training_params():
    """Sample TrainingParams configuration for testing."""
    return TrainingParams(
        batch_size=32,
        epochs=10,
        learning_rate=0.001,
        weight_decay=0.0001,
        early_stopping_patience=5,
        n_folds=3,
        min_lag=-1000,
        max_lag=1000,
        lag_step_size=200,
        fold_type="sequential_folds",
        loss_name="mse",
        metrics=["cosine_sim", "nll_embedding"],
        early_stopping_metric="cosine_sim",
        smaller_is_better=False,
        grad_accumulation_steps=1,
    )


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing dict_to_config."""
    return {
        "model_constructor_name": "test_model",
        "model_params": {"lr": 0.001, "hidden_dim": 128},
        "training_params": {
            "batch_size": 16,
            "epochs": 5,
            "learning_rate": 0.01,
            "loss_name": "mse",
            "metrics": ["cosine_sim"],
        },
        "data_params": {
            "embedding_type": "glove",
            "subject_ids": [1, 2, 3],
            "window_width": 0.625,
        },
        "trial_name": "test_trial",
    }


class TestDictToConfig:
    """Test dict_to_config conversion function."""

    def test_simple_dict_conversion(self):
        """Test converting simple dictionary to DataParams."""
        data_dict = {
            "embedding_type": "glove",
            "window_width": 0.5,
            "subject_ids": [1, 2, 3],
            "embedding_pca_dim": 50,
        }

        params = dict_to_config(data_dict, DataParams)

        assert isinstance(params, DataParams)
        assert params.embedding_type == "glove"
        assert params.window_width == 0.5
        assert params.subject_ids == [1, 2, 3]
        assert params.embedding_pca_dim == 50
        # Check that defaults are preserved
        assert params.data_root == "data"

    def test_nested_dict_conversion(self, sample_config_dict):
        """Test converting nested dictionary to ExperimentConfig."""
        config = dict_to_config(sample_config_dict, ExperimentConfig)

        assert isinstance(config, ExperimentConfig)
        assert config.model_constructor_name == "test_model"
        assert config.trial_name == "test_trial"

        # Check nested dataclasses
        assert isinstance(config.training_params, TrainingParams)
        assert config.training_params.batch_size == 16
        assert config.training_params.epochs == 5
        assert config.training_params.learning_rate == 0.01

        assert isinstance(config.data_params, DataParams)
        assert config.data_params.embedding_type == "glove"
        assert config.data_params.subject_ids == [1, 2, 3]
        assert config.data_params.window_width == 0.625

    def test_partial_dict_conversion(self):
        """Test conversion with partial dictionary (missing fields use defaults)."""
        partial_dict = {
            "model_constructor_name": "test_model",
            "training_params": {
                "batch_size": 128
                # Other fields should use defaults
            },
        }

        config = dict_to_config(partial_dict, ExperimentConfig)

        assert config.model_constructor_name == "test_model"
        assert config.training_params.batch_size == 128
        # Check defaults are preserved
        assert config.training_params.epochs == 100  # default
        assert config.training_params.learning_rate == 0.001  # default
        assert config.data_params.embedding_type == "gpt-2xl"  # default

    def test_empty_dict_conversion(self):
        """Test conversion with empty dictionary (all defaults)."""
        config = dict_to_config({}, ExperimentConfig)

        assert isinstance(config, ExperimentConfig)
        assert config.model_constructor_name == ""
        assert config.training_params.batch_size == 32
        assert config.data_params.embedding_type == "gpt-2xl"

    def test_invalid_field_ignored(self):
        """Test that invalid fields in dictionary are ignored."""
        data_dict = {
            "embedding_type": "glove",
            "invalid_field": "should_be_ignored",
            "another_invalid": 42,
        }

        params = dict_to_config(data_dict, DataParams)

        assert params.embedding_type == "glove"
        assert not hasattr(params, "invalid_field")
        assert not hasattr(params, "another_invalid")


class TestYAMLIntegration:
    """Test YAML file loading and conversion."""

    def test_yaml_to_config_conversion(self, temp_config_file):
        """Test loading YAML file and converting to ExperimentConfig."""
        with open(temp_config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        config = dict_to_config(config_dict, ExperimentConfig)

        assert isinstance(config, ExperimentConfig)
        assert config.model_constructor_name == "test_model"
        assert config.config_setter_name == "test_setter"
        assert config.model_params["hidden_dim"] == 256
        assert config.model_params["dropout"] == 0.1
        assert config.training_params.batch_size == 64
        assert config.training_params.learning_rate == 0.001
        assert config.training_params.epochs == 20
        assert config.data_params.embedding_type == "gpt-2xl"
        assert config.data_params.embedding_layer == 24
        assert config.data_params.subject_ids == [1, 2, 3]
        assert config.trial_name == "temp_test"

    def test_roundtrip_conversion(self, sample_experiment_config):
        """Test that config can be converted to dict and back without loss."""
        # Convert to dict (as would be done for YAML serialization)
        config_dict = asdict(sample_experiment_config)

        # Convert back to config
        recovered_config = dict_to_config(config_dict, ExperimentConfig)

        # Check key fields are preserved
        assert (
            recovered_config.model_constructor_name
            == sample_experiment_config.model_constructor_name
        )
        assert recovered_config.trial_name == sample_experiment_config.trial_name
        assert (
            recovered_config.training_params.batch_size
            == sample_experiment_config.training_params.batch_size
        )
        assert (
            recovered_config.data_params.embedding_type
            == sample_experiment_config.data_params.embedding_type
        )
