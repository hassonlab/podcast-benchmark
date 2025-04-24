import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from foundation_model.model_code.config import VideoMAEExperimentConfig
from foundation_model.model_code.models_mae import MaskedAutoencoderViT
from foundation_model.model_code.utils import create_model
import registry


class MLP(nn.Module):
    def __init__(
        self, layer_sizes, activation=F.relu, dropout_rate=0.2, use_layer_norm=True
    ):
        """
        Initialize a Multi-Layer Perceptron with configurable architecture and LayerNorm.

        Args:
            layer_sizes (list): List of integers specifying the size of each layer.
                               First element is input size, last element is output size.
            activation (function): Activation function to use between layers (default: ReLU).
            dropout_rate (float): Dropout probability for regularization (default: 0.2).
            use_layer_norm (bool): Whether to use LayerNorm after each hidden layer (default: True).
        """
        super(MLP, self).__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.use_layer_norm = use_layer_norm

        # Create linear layers and layer norms based on specified sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add layer norm for all but the output layer
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(layer_sizes[i + 1]))

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, layer_sizes[0]]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, layer_sizes[-1]]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Final layer will be normed.
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            # Apply activation, and dropout to all but the final layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)

        return x


class FoundationModelFinetuneMLP(nn.Module):
    def __init__(
        self,
        foundation_model_config: VideoMAEExperimentConfig,
        mlp_layer_sizes,
        model_dir=None,
        mlp_activation=F.relu,
        dropout_rate=0.2,
        use_layer_norm=True,
    ):
        super(FoundationModelFinetuneMLP, self).__init__()
        self.foundation_model = create_model(foundation_model_config)
        if model_dir:
            checkpoint = torch.load(
                os.path.join(model_dir, "model_weight_test.pth"),
                weights_only=True,
            )
            self.foundation_model.load_state_dict(checkpoint)

        self.mlp = MLP(mlp_layer_sizes, mlp_activation, dropout_rate, use_layer_norm)

    def forward(self, x):
        x = self.foundation_model(x, forward_features=True)
        return self.mlp(x)


@registry.register_model_constructor()
def foundation_mlp(model_params):
    return MLP(model_params["layer_sizes"])


@registry.register_model_constructor()
def foundation_model_finetune_mlp(model_params):
    return FoundationModelFinetuneMLP(
        model_params["foundation_model_config"],
        model_params["mlp_layer_sizes"],
    )
