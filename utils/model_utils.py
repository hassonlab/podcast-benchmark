"""Utilities for building models from ModelSpec configurations."""

import os
from typing import Optional

import torch

from core.config import ModelSpec
from core.registry import model_constructor_registry


def _randomly_initialize_model(model: torch.nn.Module) -> None:
    """Randomize every parameter, falling back when native hooks leave one unchanged.

    Parameters are checked one module at a time to avoid copying the full model.
    """
    with torch.no_grad():
        # Leaf-first order matches torch.nn.Module.apply and avoids retaining a
        # checkpoint-sized copy of all parameters for the unchanged-value check.
        for module_name, module in reversed(list(model.named_modules())):
            direct_parameters = list(module.named_parameters(recurse=False))
            previous_parameters = {
                name: parameter.detach().clone()
                for name, parameter in direct_parameters
            }

            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

            for name, parameter in direct_parameters:
                if not torch.equal(parameter, previous_parameters[name]):
                    continue
                if not (parameter.is_floating_point() or parameter.is_complex()):
                    qualified_name = f"{module_name}.{name}".lstrip(".")
                    raise TypeError(
                        f"Cannot randomly initialize non-floating parameter "
                        f"'{qualified_name}' "
                        f"with dtype {parameter.dtype}."
                    )
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)


def build_model_from_spec(
    model_spec: ModelSpec, lag: Optional[int] = None, fold: Optional[int] = None
):
    """Recursively build a model from a ModelSpec, handling nested sub-models and checkpoints.

    This function first builds all sub-models specified in model_spec.sub_models,
    then constructs the parent model by passing both the params and the built
    sub-models as keyword arguments to the registered constructor function.

    Supports loading checkpoints with dynamic path formatting for lag and fold values.
    When ``random_init`` is enabled, the model is reset after checkpoint loading so
    the checkpoint supplies the architecture but not the learned weights.

    Args:
        model_spec: ModelSpec instance describing the model to build
        lag: Current lag value in ms (for checkpoint path formatting)
        fold: Current fold number (for checkpoint path formatting)

    Returns:
        The constructed model instance, with checkpoints loaded and optional random
        reinitialization applied

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

    # Get the constructor function from registry (registry stores {"constructor": fn, "required_data_getter": str|None})
    model_info = model_constructor_registry[model_spec.constructor_name]
    constructor_fn = model_info["constructor"]

    # Combine params and built sub-models into a single kwargs dict.
    # Forward top-level flags into model params for convenience/compatibility.
    all_kwargs = {**model_spec.params, **built_sub_models}
    if "feature_cache" not in all_kwargs:
        all_kwargs["feature_cache"] = model_spec.feature_cache

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

    if model_spec.random_init:
        _randomly_initialize_model(model)

    return model
