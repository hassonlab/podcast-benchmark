"""
Integration code for the Example Foundation Model

This module demonstrates TWO ways to use a foundation model in the benchmark:

1. **Feature Extraction (Frozen)**: Load a pretrained model, freeze it, and use it to
   extract embeddings during preprocessing. Then train a simple decoder on top.

2. **Finetuning**: Include the foundation model as part of your decoder architecture,
   and continue training it (fully or partially) on your downstream task.

Both patterns are registered with the framework's registry system.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from core import registry
from .simple_transformer import load_pretrained_model, SimpleTransformer


# =============================================================================
# PATTERN 1: FEATURE EXTRACTION (FROZEN MODEL)
# =============================================================================

@registry.register_data_preprocessor("example_foundation_feature_extraction")
def extract_foundation_features(data, preprocessor_params):
    """
    Extract frozen features from the foundation model during preprocessing.

    This pattern:
    1. Loads the pretrained foundation model
    2. Freezes all parameters
    3. Runs the data through the model to extract embeddings
    4. Returns embeddings which are then used to train a simple decoder

    Args:
        data: Neural data of shape [num_samples, num_channels, num_timepoints]
        preprocessor_params: Dictionary with:
            - model_dir: Path to pretrained model directory
            - batch_size: Batch size for processing (default: 32)

    Returns:
        embeddings: Numpy array of shape [num_samples, model_dim]
    """
    model_dir = preprocessor_params["model_dir"]
    batch_size = preprocessor_params.get("batch_size", 32)

    # Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(model_dir, device=device)
    model.eval()
    model.freeze()

    print(f"Loaded foundation model from {model_dir}")
    print(f"Model has {model.get_num_params():,} parameters (all frozen)")

    # Extract embeddings in batches
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Extracting features"):
            batch = torch.tensor(
                data[i : i + batch_size],
                dtype=torch.float32,
                device=device
            )
            batch_embeddings = model(batch, return_sequence=False)
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Extracted embeddings of shape: {embeddings.shape}")

    return embeddings


# Simple MLP decoder to use on top of frozen features
class MLPDecoder(nn.Module):
    """
    Simple MLP decoder for use with frozen foundation features.

    This is used with PATTERN 1 (feature extraction).
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # Build layers
        prev_dim = input_dim
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_dim, size))
            if use_layer_norm and size != layer_sizes[-1]:
                self.layer_norms.append(nn.LayerNorm(size))
            prev_dim = size

        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Don't apply activation/dropout to final layer
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        return x


@registry.register_model_constructor("example_foundation_mlp")
def create_mlp_decoder(model_params):
    """
    Create MLP decoder for frozen foundation features.

    This is used with PATTERN 1 (feature extraction).

    Expected model_params:
        - input_dim: Dimension of foundation model embeddings
        - layer_sizes: List of layer sizes for MLP
        - dropout: Dropout probability (optional)
        - use_layer_norm: Whether to use layer normalization (optional)
    """
    return MLPDecoder(
        input_dim=model_params["input_dim"],
        layer_sizes=model_params["layer_sizes"],
        dropout=model_params.get("dropout", 0.0),
        use_layer_norm=model_params.get("use_layer_norm", False),
    )


# =============================================================================
# PATTERN 2: FINETUNING (TRAINABLE MODEL)
# =============================================================================

class FoundationModelDecoder(nn.Module):
    """
    Decoder that includes the foundation model as a trainable submodule.

    This pattern:
    1. Loads the foundation model with pretrained weights
    2. Optionally freezes some layers (partial finetuning)
    3. Includes the foundation model in the decoder architecture
    4. During training, gradients flow through unfrozen parts

    This is used with PATTERN 2 (finetuning).
    """

    def __init__(
        self,
        model_dir: str,
        output_dim: int,
        mlp_layer_sizes: list[int],
        freeze_foundation: bool = False,
        num_frozen_layers: int = 0,
        dropout: float = 0.0,
    ):
        """
        Args:
            model_dir: Path to pretrained foundation model directory
            output_dim: Output dimension for final predictions
            mlp_layer_sizes: Layer sizes for decoder head MLP
            freeze_foundation: If True, freeze entire foundation model
            num_frozen_layers: Number of foundation layers to freeze (0 = none)
            dropout: Dropout probability
        """
        super().__init__()

        # Load pretrained foundation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.foundation_model = load_pretrained_model(model_dir, device=device)

        # Handle freezing
        if freeze_foundation:
            self.foundation_model.freeze()
            print("Foundation model completely frozen")
        elif num_frozen_layers > 0:
            self.foundation_model.freeze_layers(num_frozen_layers)
            print(f"Froze first {num_frozen_layers} layers of foundation model")
        else:
            print("Foundation model fully trainable")

        # Decoder head (MLP on top of foundation features)
        foundation_dim = self.foundation_model.model_dim
        self.decoder_head = MLPDecoder(
            input_dim=foundation_dim,
            layer_sizes=mlp_layer_sizes + [output_dim],
            dropout=dropout,
            use_layer_norm=True,
        )

    def forward(self, x):
        """
        Forward pass through foundation model and decoder head.

        Args:
            x: Input tensor [batch_size, num_channels, seq_len]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Get features from foundation model
        features = self.foundation_model(x, return_sequence=False)

        # Pass through decoder head
        output = self.decoder_head(features)

        return output


@registry.register_model_constructor("example_foundation_finetune")
def create_finetuning_decoder(model_params):
    """
    Create decoder with foundation model for finetuning.

    This is used with PATTERN 2 (finetuning).

    Expected model_params:
        - model_dir: Path to pretrained model directory
        - output_dim: Output dimension
        - mlp_layer_sizes: Layer sizes for decoder head
        - freeze_foundation: Whether to freeze entire foundation (optional)
        - num_frozen_layers: Number of layers to freeze (optional)
        - dropout: Dropout probability (optional)
    """
    return FoundationModelDecoder(
        model_dir=model_params["model_dir"],
        output_dim=model_params["output_dim"],
        mlp_layer_sizes=model_params.get("mlp_layer_sizes", [128]),
        freeze_foundation=model_params.get("freeze_foundation", False),
        num_frozen_layers=model_params.get("num_frozen_layers", 0),
        dropout=model_params.get("dropout", 0.1),
    )


# =============================================================================
# CONFIG SETTERS
# =============================================================================

@registry.register_config_setter("example_foundation_feature_extraction")
def set_feature_extraction_config(experiment_config, raws, _df_word):
    """
    Config setter for feature extraction pattern.

    Sets the input_dim for the MLP based on the foundation model's dimension.
    """
    from .config import load_config

    model_dir = experiment_config.model_params["model_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    foundation_config = load_config(config_path)

    # Set input_dim to match foundation model output
    experiment_config.model_params["input_dim"] = foundation_config.model_dim

    # Set preprocessor params
    if not experiment_config.data_params.preprocessor_params:
        experiment_config.data_params.preprocessor_params = {}
    experiment_config.data_params.preprocessor_params["model_dir"] = model_dir

    return experiment_config


@registry.register_config_setter("example_foundation_finetune")
def set_finetuning_config(experiment_config, raws, _df_word):
    """
    Config setter for finetuning pattern.

    Sets the output_dim and loads foundation model config.
    """
    from .config import load_config

    model_dir = experiment_config.model_params["model_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    foundation_config = load_config(config_path)

    # Set window width based on foundation model
    experiment_config.data_params.window_width = foundation_config.window_width

    return experiment_config
