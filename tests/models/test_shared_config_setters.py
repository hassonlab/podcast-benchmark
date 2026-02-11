"""Tests for shared_config_setters functions."""

import pytest
from core.config import ModelSpec
from models.shared_config_setters import set_model_spec_fields


def test_set_input_channels_override_when_constructor_names_is_none():
    """Test that it correctly overrides the parameter value when model_spec_constructor_names is None."""
    # Create a simple ModelSpec
    model_spec = ModelSpec(
        constructor_name="test_model", params={"input_channels": 10, "output_dim": 5}
    )

    # Call the function with model_spec_constructor_names=None
    num_electrodes = 64
    value_set = set_model_spec_fields(
        model_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names=None,
    )

    # Assert that the value was set
    assert value_set is True
    # Assert that the input_channels parameter was overridden
    assert model_spec.params["input_channels"] == 64


def test_set_input_channels_sets_correct_modelspec_when_constructor_names_specified():
    """Test that it correctly sets the right ModelSpec when model_spec_constructor_names is not None."""
    # Create a nested ModelSpec structure
    encoder_spec = ModelSpec(
        constructor_name="encoder_model",
        params={"input_channels": 10, "embedding_dim": 128},
    )

    decoder_spec = ModelSpec(
        constructor_name="decoder_model",
        params={"input_channels": 10, "output_dim": 50},
    )

    parent_spec = ModelSpec(
        constructor_name="parent_model",
        params={"num_classes": 10},
        sub_models={"encoder": encoder_spec, "decoder": decoder_spec},
    )

    # Set input_channels only for the encoder_model
    num_electrodes = 64
    value_set = set_model_spec_fields(
        parent_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names=["encoder_model"],
    )

    # Assert that the value was set
    assert value_set is True
    # Assert that only the encoder's input_channels was updated
    assert encoder_spec.params["input_channels"] == 64
    # Assert that the decoder's input_channels was NOT updated
    assert decoder_spec.params["input_channels"] == 10
    # Assert that the parent's params were not updated
    assert "input_channels" not in parent_spec.params


def test_set_input_channels_sets_multiple_matching_constructors():
    """Test that it sets input_channels for all matching constructor names."""
    # Create a nested ModelSpec structure with multiple models with the same constructor name
    encoder1_spec = ModelSpec(
        constructor_name="encoder_model",
        params={"input_channels": 10, "embedding_dim": 128},
    )

    encoder2_spec = ModelSpec(
        constructor_name="encoder_model",
        params={"input_channels": 20, "embedding_dim": 256},
    )

    parent_spec = ModelSpec(
        constructor_name="parent_model",
        params={"num_classes": 10},
        sub_models={"encoder1": encoder1_spec, "encoder2": encoder2_spec},
    )

    # Set input_channels for all encoder_model instances
    num_electrodes = 64
    value_set = set_model_spec_fields(
        parent_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names=["encoder_model"],
    )

    # Assert that the value was set
    assert value_set is True
    # Assert that both encoders were updated
    assert encoder1_spec.params["input_channels"] == 64
    assert encoder2_spec.params["input_channels"] == 64


def test_set_input_channels_searches_nested_submodels():
    """Test that it recursively searches nested sub-models for matching constructor names."""
    # Create a deeply nested ModelSpec structure
    innermost_spec = ModelSpec(
        constructor_name="inner_encoder",
        params={"input_channels": 10, "embedding_dim": 64},
    )

    middle_spec = ModelSpec(
        constructor_name="middle_model",
        params={"hidden_dim": 128},
        sub_models={"encoder": innermost_spec},
    )

    parent_spec = ModelSpec(
        constructor_name="parent_model",
        params={"num_classes": 10},
        sub_models={"middle": middle_spec},
    )

    # Set input_channels for the deeply nested inner_encoder
    num_electrodes = 64
    value_set = set_model_spec_fields(
        parent_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names=["inner_encoder"],
    )

    # Assert that the value was set
    assert value_set is True
    # Assert that the innermost encoder was updated
    assert innermost_spec.params["input_channels"] == 64


def test_set_input_channels_raises_error_when_constructor_not_found():
    """Test that it raises an error when the specified constructor name is not present in the ModelSpec."""
    # Create a simple ModelSpec
    encoder_spec = ModelSpec(
        constructor_name="encoder_model",
        params={"input_channels": 10, "embedding_dim": 128},
    )

    parent_spec = ModelSpec(
        constructor_name="parent_model",
        params={"num_classes": 10},
        sub_models={"encoder": encoder_spec},
    )

    # Try to set input_channels for a non-existent constructor name
    num_electrodes = 64
    value_set = set_model_spec_fields(
        parent_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names=["non_existent_model"],
    )

    # Assert that the value was not set (returns False)
    assert value_set is False


def test_set_input_channels_handles_multiple_constructor_names():
    """Test that it correctly handles multiple constructor names in the list."""
    # Create a nested ModelSpec structure
    encoder_spec = ModelSpec(
        constructor_name="encoder_model",
        params={"input_channels": 10, "embedding_dim": 128},
    )

    decoder_spec = ModelSpec(
        constructor_name="decoder_model",
        params={"input_channels": 10, "output_dim": 50},
    )

    classifier_spec = ModelSpec(
        constructor_name="classifier_model",
        params={"input_dim": 128, "num_classes": 10},
    )

    parent_spec = ModelSpec(
        constructor_name="parent_model",
        params={"dropout": 0.5},
        sub_models={
            "encoder": encoder_spec,
            "decoder": decoder_spec,
            "classifier": classifier_spec,
        },
    )

    # Set input_channels for both encoder and decoder
    num_electrodes = 64
    value_set = set_model_spec_fields(
        parent_spec,
        {"input_channels": num_electrodes},
        model_spec_constructor_names=["encoder_model", "decoder_model"],
    )

    # Assert that the value was set
    assert value_set is True
    # Assert that both encoder and decoder were updated
    assert encoder_spec.params["input_channels"] == 64
    assert decoder_spec.params["input_channels"] == 64
    # Assert that classifier was not updated
    assert "input_channels" not in classifier_spec.params
