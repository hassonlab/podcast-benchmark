"""
Configuration System for Example Foundation Model

This module provides a simple configuration system for the transformer foundation model.
It demonstrates how to structure configs that can be saved/loaded from YAML files.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import yaml
import os


@dataclass
class TransformerConfig:
    """
    Configuration for the SimpleTransformer foundation model.

    This config includes all the architectural parameters needed to
    reconstruct the model from a saved checkpoint.
    """

    # Model architecture parameters
    input_channels: int = 64  # Number of input channels (e.g., electrodes)
    model_dim: int = 256  # Dimension of the transformer model
    num_layers: int = 4  # Number of transformer encoder layers
    num_heads: int = 8  # Number of attention heads
    dim_feedforward: int = 1024  # Dimension of feedforward network
    dropout: float = 0.1  # Dropout probability
    max_seq_len: int = 1000  # Maximum sequence length

    # Data parameters (for preprocessing)
    window_width: float = 0.625  # Window width in seconds for neural data
    sample_rate: int = 512  # Expected sample rate of input data

    def save(self, path: str):
        """
        Save config to a YAML file.

        Args:
            path: Path to save the config file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TransformerConfig":
        """
        Create a TransformerConfig from a dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            TransformerConfig instance
        """
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


def load_config(path: str) -> TransformerConfig:
    """
    Load a TransformerConfig from a YAML file.

    This is the standard function used to load model configs from a model directory.

    Args:
        path: Path to the config YAML file

    Returns:
        TransformerConfig instance

    Example:
        >>> config = load_config("pretrained_model/config.yaml")
        >>> print(config.model_dim)
        256
    """
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TransformerConfig.from_dict(config_dict)


def save_config(config: TransformerConfig, path: str):
    """
    Save a TransformerConfig to a YAML file.

    Args:
        config: TransformerConfig instance to save
        path: Path to save the config file

    Example:
        >>> config = TransformerConfig(model_dim=512, num_layers=6)
        >>> save_config(config, "my_model/config.yaml")
    """
    config.save(path)


def create_default_config() -> TransformerConfig:
    """
    Create a default configuration for the transformer model.

    This is used when creating a new model from scratch.

    Returns:
        TransformerConfig with default values
    """
    return TransformerConfig()


def create_config_from_params(
    input_channels: int,
    model_dim: int = 256,
    num_layers: int = 4,
    **kwargs
) -> TransformerConfig:
    """
    Create a config with specific parameters.

    This is a convenience function for programmatically creating configs.

    Args:
        input_channels: Number of input channels
        model_dim: Model dimension
        num_layers: Number of transformer layers
        **kwargs: Additional config parameters

    Returns:
        TransformerConfig instance

    Example:
        >>> config = create_config_from_params(
        ...     input_channels=128,
        ...     model_dim=512,
        ...     num_layers=6,
        ...     dropout=0.2
        ... )
    """
    return TransformerConfig(
        input_channels=input_channels,
        model_dim=model_dim,
        num_layers=num_layers,
        **kwargs
    )
