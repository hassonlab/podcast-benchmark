import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from foundation_model.model_code.config import VideoMAEExperimentConfig
from foundation_model.model_code.models_mae import MaskedAutoencoderViT
from foundation_model.foundation_decoder_utils import create_foundation_model
import registry


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=False,
    ):
        """
        Initialize a Multi-Layer Perceptron with configurable architecture and LayerNorm.

        Args:
            layer_sizes (list): List of integers specifying the size of each layer.
                               First element is input size, last element is output size.
            activation (function): Activation function to use between layers (default: ReLU).
            dropout_rate (float): Dropout probability for regularization (default: 0.).
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

            # Apply activation, and dropout to all but the final layer
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)

        return x


class FoundationModelMLP(nn.Module):
    def __init__(
        self,
        mlp_layer_sizes,
        model_dir=None,
        mlp_activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=True,
        finetune=False,
        foundation_model_config: Optional[VideoMAEExperimentConfig] = None,
    ):
        super(FoundationModelMLP, self).__init__()
        self.finetune = finetune
        if finetune:
            self.foundation_model = create_foundation_model(
                foundation_model_config, model_dir=model_dir
            )

        self.mlp = MLP(mlp_layer_sizes, mlp_activation, dropout_rate, use_layer_norm)

    def forward(self, x):
        if self.finetune:
            x = self.foundation_model(x, forward_features=True)
        return self.mlp(x)


class FoundationModelAttentionPoolingDecoder(nn.Module):
    def __init__(
        self,
        mlp_layer_sizes,
        model_dir=None,
        mlp_activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=True,
        finetune=False,
        foundation_model_config: Optional[VideoMAEExperimentConfig] = None,
    ):
        super().__init__()
        self.finetune = finetune
        if finetune:
            self.foundation_model = create_foundation_model(
                foundation_model_config, model_dir
            )

        self.query = nn.Parameter(torch.randn(1, mlp_layer_sizes[0]))  # 1 x C
        self.mlp = MLP(mlp_layer_sizes, mlp_activation, dropout_rate, use_layer_norm)

    def forward(self, x, return_weights=False):
        """
        Args:
            x: Tensor of shape (B, N, C), transformer outputs
        Returns:
            output: Tensor of shape (B, out_dim)
        """
        if self.finetune:
            x = self.foundation_model(x, forward_features=True, global_pool=False)

        B, N, C = x.shape

        # Expand query to batch size
        query = self.query.expand(B, -1, -1)  # (B, 1, C)

        # Compute attention scores
        attn_scores = torch.matmul(query, x.transpose(1, 2))  # (B, 1, N)
        attn_scores = attn_scores / (C**0.5)  # scale by sqrt(d)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, N)

        # Weighted sum
        pooled = torch.matmul(attn_weights, x)  # (B, 1, C)
        pooled = pooled.squeeze(1)  # (B, C)

        # Final projection
        output = self.mlp(pooled)  # (B, out_dim)
        if return_weights:
            return output, attn_weights
        return output


@registry.register_model_constructor()
def foundation_mlp(model_params):
    return FoundationModelMLP(model_params["layer_sizes"], finetune=False)


@registry.register_model_constructor()
def foundation_model_finetune_mlp(model_params):
    return FoundationModelMLP(
        model_params["mlp_layer_sizes"],
        model_dir=model_params.get("model_dir"),
        foundation_model_config=model_params["foundation_model_config"],
        finetune=True,
    )


@registry.register_model_constructor()
def foundation_model_finetune_attention(model_params):
    return FoundationModelAttentionPoolingDecoder(
        model_params["mlp_layer_sizes"],
        model_dir=model_params.get("model_dir"),
        foundation_model_config=model_params["foundation_model_config"],
        finetune=True,
    )
