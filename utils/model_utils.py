"""Utilities for building models from ModelSpec configurations."""

import os
from typing import Optional

import torch

from core.config import ModelSpec
from core.registry import model_constructor_registry


def build_model_from_spec(
    model_spec: ModelSpec, lag: Optional[int] = None, fold: Optional[int] = None
):
    """Recursively build a model from a ModelSpec, handling nested sub-models and checkpoints.

    This function first builds all sub-models specified in model_spec.sub_models,
    then constructs the parent model by passing both the params and the built
    sub-models as keyword arguments to the registered constructor function.

    Supports loading checkpoints with dynamic path formatting for lag and fold values.

    Args:
        model_spec: ModelSpec instance describing the model to build
        lag: Current lag value in ms (for checkpoint path formatting)
        fold: Current fold number (for checkpoint path formatting)

    Returns:
        The constructed model instance with checkpoints loaded if specified

    Raises:
        KeyError: If the constructor_name is not found in model_constructor_registry
        FileNotFoundError: If a specified checkpoint path does not exist

    Example:
        >>> encoder_spec = ModelSpec(
        ...     constructor_name="pitom_model",
        ...     params={"input_channels": 64, "output_dim": 768},
        ...     sub_models={},
        ...     checkpoint_path="checkpoints/encoder/lag_{lag}/fold_{fold}/best_model.pt"
        ... )
        >>> gpt2_spec = ModelSpec(
        ...     constructor_name="gpt2_brain",
        ...     params={"freeze_lm": True},
        ...     sub_models={"encoder_model": encoder_spec}
        ... )
        >>> model = build_model_from_spec(gpt2_spec, lag=200, fold=3)
        # This will:
        # 1. Build encoder: pitom_model(input_channels=64, output_dim=768)
        # 2. Load encoder checkpoint from checkpoints/encoder/lag_200/fold_3/best_model.pt
        # 3. Build parent: gpt2_brain(freeze_lm=True, encoder_model=<built_encoder>)
    """
    if model_spec.constructor_name not in model_constructor_registry:
        raise KeyError(
            f"Model constructor '{model_spec.constructor_name}' not found in registry. "
            f"Available constructors: {list(model_constructor_registry.keys())}"
        )

    # Recursively build all sub-models (they will handle their own checkpoint loading)
    built_sub_models = {}
    for param_name, sub_spec in model_spec.sub_models.items():
        built_sub_models[param_name] = build_model_from_spec(sub_spec, lag, fold)

    # Get the constructor function
    constructor_fn = model_constructor_registry[model_spec.constructor_name]

    # Combine params and built sub-models into a single kwargs dict
    all_kwargs = {**model_spec.params, **built_sub_models}

    # Build the model
    model = constructor_fn(all_kwargs)

    # Load checkpoint if specified for this model
    if model_spec.checkpoint_path:
        checkpoint_path = model_spec.checkpoint_path

        # Support dynamic path formatting with lag and fold
        if lag is not None and fold is not None:
            if "{lag}" in checkpoint_path or "{fold}" in checkpoint_path:
                checkpoint_path = checkpoint_path.format(lag=lag, fold=fold)

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please ensure the checkpoint exists or run pre-training first."
            )

    return model
