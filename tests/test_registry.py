"""
Tests for the registry system (registry.py).

Tests decorator registration, function lookup, and registry management.
"""

import pytest
import torch
import numpy as np

import registry


@pytest.fixture
def mock_model_constructor():
    """Mock model constructor function for testing registry."""

    def constructor(model_params):
        # Return a simple mock object with the params
        return type("MockModel", (), {"params": model_params})()

    return constructor


@pytest.fixture
def mock_data_preprocessor():
    """Mock data preprocessor function for testing registry."""

    def preprocessor(data, preprocessor_params):
        # Simple preprocessing - just reshape
        return data.reshape(data.shape[0], -1)

    return preprocessor


@pytest.fixture
def mock_config_setter():
    """Mock config setter function for testing registry."""

    def setter(experiment_config, raws, df_word):
        # Modify some config field
        experiment_config.model_params["modified"] = True
        return experiment_config

    return setter


@pytest.fixture
def mock_metric():
    """Mock metric function for testing registry."""

    def metric(predicted, groundtruth):
        # Simple mock metric - return random value
        return torch.tensor(0.5)

    return metric


class TestModelConstructorRegistry:
    """Test model constructor registration and lookup."""

    def test_register_model_constructor_decorator(self, mock_model_constructor):
        """Test that @register_model_constructor adds function to registry."""
        # Use the decorator
        decorated_fn = registry.register_model_constructor("test_model")(
            mock_model_constructor
        )

        # Check it was added to registry
        assert "test_model" in registry.model_constructor_registry
        assert registry.model_constructor_registry["test_model"] == decorated_fn
        assert decorated_fn == mock_model_constructor  # Should return original function

    def test_register_model_constructor_default_name(self, mock_model_constructor):
        """Test that decorator uses function name when no name provided."""
        # Rename the function for this test
        mock_model_constructor.__name__ = "my_custom_model"

        decorated_fn = registry.register_model_constructor()(mock_model_constructor)

        assert "my_custom_model" in registry.model_constructor_registry
        assert registry.model_constructor_registry["my_custom_model"] == decorated_fn

    def test_model_constructor_functionality(self, mock_model_constructor):
        """Test that registered model constructor works correctly."""
        registry.register_model_constructor("test_model")(mock_model_constructor)

        constructor = registry.model_constructor_registry["test_model"]
        test_params = {"param1": "value1", "param2": 42}

        model = constructor(test_params)

        assert hasattr(model, "params")
        assert model.params == test_params

    def test_multiple_model_constructors(self, mock_model_constructor):
        """Test registering multiple model constructors."""

        def constructor1(params):
            return {"type": "model1", "params": params}

        def constructor2(params):
            return {"type": "model2", "params": params}

        registry.register_model_constructor("model1")(constructor1)
        registry.register_model_constructor("model2")(constructor2)

        assert len(registry.model_constructor_registry) == 2
        assert "model1" in registry.model_constructor_registry
        assert "model2" in registry.model_constructor_registry

        # Test they work independently
        result1 = registry.model_constructor_registry["model1"]({"test": 1})
        result2 = registry.model_constructor_registry["model2"]({"test": 2})

        assert result1["type"] == "model1"
        assert result2["type"] == "model2"


class TestDataPreprocessorRegistry:
    """Test data preprocessor registration and lookup."""

    def test_register_data_preprocessor_decorator(self, mock_data_preprocessor):
        """Test that @register_data_preprocessor adds function to registry."""
        decorated_fn = registry.register_data_preprocessor("test_preprocessor")(
            mock_data_preprocessor
        )

        assert "test_preprocessor" in registry.data_preprocessor_registry
        assert registry.data_preprocessor_registry["test_preprocessor"] == decorated_fn

    def test_data_preprocessor_functionality(self, mock_data_preprocessor):
        """Test that registered preprocessor works correctly."""
        registry.register_data_preprocessor("test_preprocessor")(mock_data_preprocessor)

        preprocessor = registry.data_preprocessor_registry["test_preprocessor"]

        # Create test data: [batch, channels, time]
        test_data = np.random.randn(10, 32, 100)
        test_params = {"param1": "value1"}

        result = preprocessor(test_data, test_params)

        # Should be reshaped to [batch, features]
        assert result.shape[0] == test_data.shape[0]
        assert len(result.shape) == 2  # Should be flattened

    def test_preprocessor_default_name(self):
        """Test preprocessor registration with default name."""

        def my_preprocessor(data, params):
            return data * 2

        decorated_fn = registry.register_data_preprocessor()(my_preprocessor)

        assert "my_preprocessor" in registry.data_preprocessor_registry
        assert registry.data_preprocessor_registry["my_preprocessor"] == decorated_fn


class TestConfigSetterRegistry:
    """Test config setter registration and lookup."""

    def test_register_config_setter_decorator(self, mock_config_setter):
        """Test that @register_config_setter adds function to registry."""
        decorated_fn = registry.register_config_setter("test_setter")(
            mock_config_setter
        )

        assert "test_setter" in registry.config_setter_registry
        assert registry.config_setter_registry["test_setter"] == decorated_fn

    def test_config_setter_functionality(
        self, mock_config_setter, sample_experiment_config
    ):
        """Test that registered config setter works correctly."""
        registry.register_config_setter("test_setter")(mock_config_setter)

        setter = registry.config_setter_registry["test_setter"]

        # Mock raw data and df_word (these would normally be MNE objects and DataFrame)
        mock_raws = [{"info": {"ch_names": ["ch1", "ch2"]}}]
        mock_df_word = {"words": ["word1", "word2"]}

        # Should modify the config
        original_params = sample_experiment_config.model_params.copy()
        modified_config = setter(sample_experiment_config, mock_raws, mock_df_word)

        # Check that modification was made
        assert "modified" in modified_config.model_params
        assert modified_config.model_params["modified"] == True

        # Should return the same config object
        assert modified_config is sample_experiment_config


class TestMetricRegistry:
    """Test metric registration and lookup."""

    def test_register_metric_decorator(self, mock_metric):
        """Test that @register_metric adds function to registry."""
        decorated_fn = registry.register_metric("test_metric")(mock_metric)

        assert "test_metric" in registry.metric_registry
        assert registry.metric_registry["test_metric"] == decorated_fn

    def test_metric_functionality(self, mock_metric):
        """Test that registered metric works correctly."""
        registry.register_metric("test_metric")(mock_metric)

        metric_fn = registry.metric_registry["test_metric"]

        # Create test tensors
        predicted = torch.randn(10, 50)
        groundtruth = torch.randn(10, 50)

        result = metric_fn(predicted, groundtruth)

        assert torch.is_tensor(result)
        assert result.item() == 0.5  # Mock metric returns 0.5

    def test_multiple_metrics(self):
        """Test registering multiple metrics."""

        def metric1(pred, true):
            return torch.tensor(1.0)

        def metric2(pred, true):
            return torch.tensor(2.0)

        registry.register_metric("metric1")(metric1)
        registry.register_metric("metric2")(metric2)

        assert len(registry.metric_registry) == 2

        # Test they work independently
        pred = torch.zeros(5, 10)
        true = torch.zeros(5, 10)

        result1 = registry.metric_registry["metric1"](pred, true)
        result2 = registry.metric_registry["metric2"](pred, true)

        assert result1.item() == 1.0
        assert result2.item() == 2.0


class TestRegistryManagement:
    """Test registry management functionality."""

    def test_registry_isolation(self):
        """Test that different registries are independent."""

        def test_fn():
            pass

        registry.register_model_constructor("test_name")(test_fn)
        registry.register_data_preprocessor("test_name")(test_fn)
        registry.register_config_setter("test_name")(test_fn)
        registry.register_metric("test_name")(test_fn)

        # All should be independent
        assert len(registry.model_constructor_registry) == 1
        assert len(registry.data_preprocessor_registry) == 1
        assert len(registry.config_setter_registry) == 1
        assert len(registry.metric_registry) == 1

        # Same name in different registries should not conflict
        assert "test_name" in registry.model_constructor_registry
        assert "test_name" in registry.data_preprocessor_registry
        assert "test_name" in registry.config_setter_registry
        assert "test_name" in registry.metric_registry

    def test_registry_overwrite(self):
        """Test that registering same name twice overwrites."""

        def fn1():
            return "first"

        def fn2():
            return "second"

        # Register first function
        registry.register_model_constructor("test_overwrite")(fn1)
        assert registry.model_constructor_registry["test_overwrite"]() == "first"

        # Register second function with same name
        registry.register_model_constructor("test_overwrite")(fn2)
        assert registry.model_constructor_registry["test_overwrite"]() == "second"

        # Should only have one entry
        assert (
            len(
                [
                    k
                    for k in registry.model_constructor_registry.keys()
                    if k == "test_overwrite"
                ]
            )
            == 1
        )

    def test_registry_persistence(self):
        """Test that registry persists across multiple registrations."""

        def fn1():
            return 1

        def fn2():
            return 2

        registry.register_model_constructor("func1")(fn1)
        assert len(registry.model_constructor_registry) == 1

        registry.register_model_constructor("func2")(fn2)
        assert len(registry.model_constructor_registry) == 2

        # Both should still be accessible
        assert registry.model_constructor_registry["func1"]() == 1
        assert registry.model_constructor_registry["func2"]() == 2
