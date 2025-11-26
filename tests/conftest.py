"""
Shared fixtures and configuration for pytest tests.
"""

import pytest
import tempfile
import os
from core.config import ExperimentConfig, DataParams, TrainingParams, ModelSpec


@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file for testing."""
    config_content = """
model_spec:
  constructor_name: test_model
  params:
    hidden_dim: 256
    dropout: 0.1
config_setter_name: test_setter
training_params:
  batch_size: 64
  learning_rate: 0.001
  epochs: 20
  loss_name: mse
  metrics: [cosine_sim]
trial_name: temp_test
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_experiment_config():
    """Create a sample ExperimentConfig for testing."""
    return ExperimentConfig(
        model_spec=ModelSpec(
            constructor_name="test_model",
            params={
                "hidden_dim": 256,
                "dropout": 0.1,
                "param1": "value1",
                "param2": 42,
            },
        ),
        config_setter_name="test_setter",
        training_params=TrainingParams(
            batch_size=32,
            learning_rate=0.01,
            epochs=20,
            loss_name="mse",
            metrics=["cosine_sim"],
        ),
        trial_name="test_experiment",
        output_dir="test_results",
        checkpoint_dir="test_checkpoints",
    )
