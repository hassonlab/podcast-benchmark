"""Tests for ModelSpec and model building utilities."""

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
