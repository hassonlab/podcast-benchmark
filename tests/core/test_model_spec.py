"""Tests for ModelSpec and model building utilities."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from core.config import ModelSpec, dict_to_config
from utils.model_utils import build_model_from_spec
from core.registry import register_model_constructor


# Test models
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.linear(x)


class EncoderModel(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, embedding_dim, kernel_size=3)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.conv(x)


class ParentModel(nn.Module):
    def __init__(self, encoder_model, num_classes):
        super().__init__()
        self.encoder = encoder_model
        self.classifier = nn.Linear(encoder_model.embedding_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features.mean(dim=-1))


# Register test model constructors
@register_model_constructor("simple_test_model")
def simple_test_model(params):
    return SimpleModel(
        input_dim=params["input_dim"],
        output_dim=params["output_dim"]
    )


@register_model_constructor("encoder_test_model")
def encoder_test_model(params):
    return EncoderModel(
        input_channels=params["input_channels"],
        embedding_dim=params["embedding_dim"]
    )


@register_model_constructor("parent_test_model")
def parent_test_model(params):
    return ParentModel(
        encoder_model=params["encoder_model"],
        num_classes=params["num_classes"]
    )


def test_build_simple_model():
    """Test building a simple model from ModelSpec."""
    spec = ModelSpec(
        constructor_name="simple_test_model",
        params={"input_dim": 10, "output_dim": 5}
    )

    model = build_model_from_spec(spec)

    assert isinstance(model, SimpleModel)
    assert model.input_dim == 10
    assert model.output_dim == 5


def test_build_nested_model():
    """Test building a model with a nested sub-model."""
    encoder_spec = ModelSpec(
        constructor_name="encoder_test_model",
        params={"input_channels": 64, "embedding_dim": 128}
    )

    parent_spec = ModelSpec(
        constructor_name="parent_test_model",
        params={"num_classes": 10},
        sub_models={"encoder_model": encoder_spec}
    )

    model = build_model_from_spec(parent_spec)

    assert isinstance(model, ParentModel)
    assert isinstance(model.encoder, EncoderModel)
    assert model.encoder.embedding_dim == 128
    assert model.classifier.out_features == 10


def test_dict_to_model_spec():
    """Test converting dict to ModelSpec using dict_to_config."""
    spec_dict = {
        "constructor_name": "simple_test_model",
        "params": {"input_dim": 10, "output_dim": 5},
        "sub_models": {}
    }

    spec = dict_to_config(spec_dict, ModelSpec)

    assert isinstance(spec, ModelSpec)
    assert spec.constructor_name == "simple_test_model"
    assert spec.params == {"input_dim": 10, "output_dim": 5}


def test_model_forward_pass():
    """Test that built models can perform forward passes."""
    spec = ModelSpec(
        constructor_name="simple_test_model",
        params={"input_dim": 10, "output_dim": 5}
    )

    model = build_model_from_spec(spec)
    x = torch.randn(32, 10)
    output = model(x)

    assert output.shape == (32, 5)


def test_checkpoint_loading_nonexistent():
    """Test that FileNotFoundError is raised when checkpoint doesn't exist."""
    spec = ModelSpec(
        constructor_name="simple_test_model",
        params={"input_dim": 10, "output_dim": 5},
        checkpoint_path="/nonexistent/path/checkpoint.pt"
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        build_model_from_spec(spec)

    assert "Checkpoint not found" in str(exc_info.value)
    assert "/nonexistent/path/checkpoint.pt" in str(exc_info.value)


def test_checkpoint_loading_valid():
    """Test that checkpoint is loaded when it exists."""
    # Create a model and save its checkpoint
    spec = ModelSpec(
        constructor_name="simple_test_model",
        params={"input_dim": 10, "output_dim": 5}
    )

    original_model = build_model_from_spec(spec)

    # Modify weights to make them distinctive
    with torch.no_grad():
        original_model.linear.weight.fill_(42.0)
        original_model.linear.bias.fill_(7.0)

    # Save checkpoint to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        checkpoint_path = f.name
        torch.save(original_model.state_dict(), checkpoint_path)

    try:
        # Create spec with checkpoint path
        spec_with_checkpoint = ModelSpec(
            constructor_name="simple_test_model",
            params={"input_dim": 10, "output_dim": 5},
            checkpoint_path=checkpoint_path
        )

        # Build model with checkpoint
        loaded_model = build_model_from_spec(spec_with_checkpoint)

        # Verify weights were loaded
        assert torch.allclose(loaded_model.linear.weight, torch.full_like(loaded_model.linear.weight, 42.0))
        assert torch.allclose(loaded_model.linear.bias, torch.full_like(loaded_model.linear.bias, 7.0))
    finally:
        # Clean up temp file
        os.unlink(checkpoint_path)


def test_checkpoint_loading_nested_model():
    """Test that checkpoint loading works for nested sub-models."""
    # Create encoder and save checkpoint
    encoder_spec = ModelSpec(
        constructor_name="encoder_test_model",
        params={"input_channels": 64, "embedding_dim": 128}
    )
    encoder_model = build_model_from_spec(encoder_spec)

    # Modify encoder weights
    with torch.no_grad():
        encoder_model.conv.weight.fill_(3.14)

    # Save encoder checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        encoder_checkpoint_path = f.name
        torch.save(encoder_model.state_dict(), encoder_checkpoint_path)

    try:
        # Create parent spec with encoder checkpoint
        encoder_spec_with_checkpoint = ModelSpec(
            constructor_name="encoder_test_model",
            params={"input_channels": 64, "embedding_dim": 128},
            checkpoint_path=encoder_checkpoint_path
        )

        parent_spec = ModelSpec(
            constructor_name="parent_test_model",
            params={"num_classes": 10},
            sub_models={"encoder_model": encoder_spec_with_checkpoint}
        )

        # Build parent model (should load encoder checkpoint)
        parent_model = build_model_from_spec(parent_spec)

        # Verify encoder weights were loaded
        assert torch.allclose(
            parent_model.encoder.conv.weight,
            torch.full_like(parent_model.encoder.conv.weight, 3.14)
        )
    finally:
        os.unlink(encoder_checkpoint_path)


def test_checkpoint_path_formatting():
    """Test that checkpoint paths are formatted with lag and fold."""
    # Create temp directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        lag = 200
        fold = 3
        checkpoint_path = os.path.join(tmpdir, "lag_{lag}", "best_model_fold{fold}.pt")

        # Create the actual directory and file
        actual_dir = os.path.join(tmpdir, f"lag_{lag}")
        os.makedirs(actual_dir, exist_ok=True)
        actual_path = os.path.join(actual_dir, f"best_model_fold{fold}.pt")

        # Create and save a model
        spec = ModelSpec(
            constructor_name="simple_test_model",
            params={"input_dim": 10, "output_dim": 5}
        )
        model = build_model_from_spec(spec)

        # Set distinctive weights
        with torch.no_grad():
            model.linear.weight.fill_(99.0)

        torch.save(model.state_dict(), actual_path)

        # Now create spec with formatted checkpoint path
        spec_with_formatted_path = ModelSpec(
            constructor_name="simple_test_model",
            params={"input_dim": 10, "output_dim": 5},
            checkpoint_path=checkpoint_path
        )

        # Build model with lag and fold parameters
        loaded_model = build_model_from_spec(spec_with_formatted_path, lag=lag, fold=fold)

        # Verify weights were loaded
        assert torch.allclose(loaded_model.linear.weight, torch.full_like(loaded_model.linear.weight, 99.0))


def test_checkpoint_path_without_formatting():
    """Test that checkpoint paths without placeholders work correctly."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        checkpoint_path = f.name

        # Create and save a model
        spec = ModelSpec(
            constructor_name="simple_test_model",
            params={"input_dim": 10, "output_dim": 5}
        )
        model = build_model_from_spec(spec)

        with torch.no_grad():
            model.linear.weight.fill_(123.0)

        torch.save(model.state_dict(), checkpoint_path)

    try:
        spec_with_checkpoint = ModelSpec(
            constructor_name="simple_test_model",
            params={"input_dim": 10, "output_dim": 5},
            checkpoint_path=checkpoint_path
        )

        # Build with lag/fold but path doesn't use them
        loaded_model = build_model_from_spec(spec_with_checkpoint, lag=100, fold=2)

        assert torch.allclose(loaded_model.linear.weight, torch.full_like(loaded_model.linear.weight, 123.0))
    finally:
        os.unlink(checkpoint_path)
