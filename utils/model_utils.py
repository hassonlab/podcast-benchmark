"""Utilities for building models from ModelSpec configurations."""

from core.config import ModelSpec
from core.registry import model_constructor_registry


def build_model_from_spec(model_spec: ModelSpec):
    """Recursively build a model from a ModelSpec.

    This function first builds all sub-models specified in model_spec.sub_models,
    then constructs the parent model by passing both the params and the built
    sub-models as keyword arguments to the registered constructor function.

    Args:
        model_spec: ModelSpec instance describing the model to build

    Returns:
        The constructed model instance

    Raises:
        KeyError: If the constructor_name is not found in model_constructor_registry

    Example:
        >>> encoder_spec = ModelSpec(
        ...     constructor_name="pitom_model",
        ...     params={"input_channels": 64, "output_dim": 768},
        ...     sub_models={}
        ... )
        >>> gpt2_spec = ModelSpec(
        ...     constructor_name="gpt2_brain",
        ...     params={"freeze_lm": True},
        ...     sub_models={"encoder_model": encoder_spec}
        ... )
        >>> model = build_model_from_spec(gpt2_spec)
        # This will:
        # 1. Build encoder: pitom_model(input_channels=64, output_dim=768)
        # 2. Build parent: gpt2_brain(freeze_lm=True, encoder_model=<built_encoder>)
    """
    if model_spec.constructor_name not in model_constructor_registry:
        raise KeyError(
            f"Model constructor '{model_spec.constructor_name}' not found in registry. "
            f"Available constructors: {list(model_constructor_registry.keys())}"
        )

    # Recursively build all sub-models
    built_sub_models = {}
    for param_name, sub_spec in model_spec.sub_models.items():
        built_sub_models[param_name] = build_model_from_spec(sub_spec)

    # Get the constructor function
    constructor_fn = model_constructor_registry[model_spec.constructor_name]

    # Combine params and built sub-models into a single kwargs dict
    all_kwargs = {**model_spec.params, **built_sub_models}

    # Build and return the model
    return constructor_fn(all_kwargs)
