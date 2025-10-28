"""
Shared fixtures and configuration for pytest tests.
"""

import pytest
import tempfile
import os
from core.config import ExperimentConfig, DataParams, TrainingParams


@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file for testing."""
    config_content = """
model_constructor_name: test_model
config_setter_name: test_setter
model_params:
  hidden_dim: 256
  dropout: 0.1
training_params:
  batch_size: 64
  learning_rate: 0.001
  epochs: 20
  loss_name: mse
  metrics: [cosine_sim]
data_params:
  embedding_type: gpt-2xl
  embedding_layer: 24
  subject_ids: [1, 2, 3]
  window_width: 0.5
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
        model_constructor_name="test_model",
        config_setter_name="test_setter",
        model_params={
            "hidden_dim": 256,
            "dropout": 0.1,
            "param1": "value1",
            "param2": 42,
        },
        training_params=TrainingParams(
            batch_size=32,
            learning_rate=0.01,
            epochs=20,
            loss_name="mse",
            metrics=["cosine_sim"],
        ),
        data_params=DataParams(
            embedding_type="gpt-2xl",
            embedding_layer=24,
            subject_ids=[1, 2, 3],
            window_width=0.5,
        ),
        trial_name="test_experiment",
        output_dir="test_results",
        checkpoint_dir="test_checkpoints",
    )
