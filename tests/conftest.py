"""
Shared fixtures and configuration for pytest tests.
"""

import pytest
import tempfile
import os
from dataclasses import dataclass
from core.config import ExperimentConfig, DataParams, TrainingParams, ModelSpec, BaseTaskConfig
from core import registry


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


# Test fixtures for load_experiment_config tests

@dataclass
class TestTaskConfig(BaseTaskConfig):
    """Test task-specific configuration."""
    test_param: str = "default_value"
    input_fields: list = None
    required_config_setter_names: list = None


@pytest.fixture
def mock_task_registry(monkeypatch):
    """Mock the task registry with a test task."""
    test_registry = {
        "test_task": {
            "getter": lambda x: None,  # Dummy getter
            "config_type": TestTaskConfig
        }
    }
    monkeypatch.setattr(registry, "task_registry", test_registry)
    return test_registry


@pytest.fixture
def temp_task_config_file():
    """Create a temporary YAML config file with task config as dict."""
    config_content = """
model_spec:
  constructor_name: test_model
  params:
    hidden_dim: 256
config_setter_name: null
task_config:
  task_name: test_task
  data_params:
    window_width: 0.5
    subject_ids: [1, 2, 3]
  task_specific_config:
    test_param: test_value
    input_fields: [field1, field2]
    required_config_setter_names: [required_setter1, required_setter2]
training_params:
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
  loss_name: mse
  metrics: [cosine_sim]
trial_name: test_with_task_config
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_electrode_file():
    """Create a temporary electrode CSV file."""
    electrode_content = """subject,elec
661,A1
661,A2
661,B1
717,C1
717,C2"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(electrode_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_subject_mapping():
    """Create a temporary subject mapping TSV file."""
    mapping_content = """nyu_id\tparticipant_id
661\tsub-05
717\tsub-12"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(mapping_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_task_config_with_electrode_file(temp_electrode_file):
    """Create a temporary config file with electrode_file_path set."""
    config_content = f"""
model_spec:
  constructor_name: test_model
  params:
    hidden_dim: 256
task_config:
  task_name: test_task
  data_params:
    window_width: 0.5
    subject_ids: []
    electrode_file_path: {temp_electrode_file}
  task_specific_config:
    test_param: test_value
    required_config_setter_names: [required_setter1]
training_params:
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
  loss_name: mse
  metrics: [cosine_sim]
trial_name: test_with_electrode_file
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_task_config_file_no_setter():
    """Create a config file without config_setter_name but with required_config_setter_names."""
    config_content = """
model_spec:
  constructor_name: test_model
  params:
    hidden_dim: 256
config_setter_name: null
task_config:
  task_name: test_task
  data_params:
    window_width: 0.5
    subject_ids: [1, 2, 3]
  task_specific_config:
    test_param: test_value
    required_config_setter_names: [required_setter1, required_setter2]
training_params:
  batch_size: 32
trial_name: test_no_setter
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_task_config_file_no_required_setters():
    """Create a config file with config_setter_name but no required_config_setter_names."""
    config_content = """
model_spec:
  constructor_name: test_model
  params:
    hidden_dim: 256
config_setter_name: null
task_config:
  task_name: test_task
  data_params:
    window_width: 0.5
    subject_ids: [1, 2, 3]
  task_specific_config:
    test_param: test_value
    required_config_setter_names: null
training_params:
  batch_size: 32
trial_name: test_no_required
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_task_config_file_no_setters_at_all():
    """Create a config file with no config setters at all."""
    config_content = """
model_spec:
  constructor_name: test_model
  params:
    hidden_dim: 256
config_setter_name: null
task_config:
  task_name: test_task
  data_params:
    window_width: 0.5
    subject_ids: [1, 2, 3]
  task_specific_config:
    test_param: test_value
    required_config_setter_names: null
training_params:
  batch_size: 32
trial_name: test_no_setters
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_nested_model_config_file():
    """Create a config file with nested model specs (sub_models)."""
    config_content = """
model_spec:
  constructor_name: gpt2_brain
  params:
    lm_model: gpt2
    freeze_lm: true
  sub_models:
    encoder_model:
      constructor_name: pitom_model
      params:
        input_channels: 64
        output_dim: 768
      sub_models: {}
config_setter_name: null
task_config:
  task_name: test_task
  data_params:
    window_width: 0.5
    subject_ids: [1, 2, 3]
  task_specific_config:
    test_param: test_value
    required_config_setter_names: null
training_params:
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
  loss_name: mse
  metrics: [cosine_sim]
trial_name: test_nested_models
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
