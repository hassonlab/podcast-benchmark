import torch
import torch.nn as nn
from torch.nn import functional as F

import registry


class PitomModel(nn.Module):
    def __init__(
        self,
        input_channels,
        output_dim,
        conv_filters=128,
        reg=0.35,
        reg_head=0,
        dropout=0.2,
        output_activation: str = "tanh",
    ):
        """
        PyTorch implementation of the PITOM decoding model.

        Args:
            input_channels: Numbr of electrodes in data (int)
            output_dim: Dimension of output vector (int)
            conv_filters: Number of convolutional filters (default: 128)
            reg: L2 regularization factor for convolutional layers (default: 0.35)
            reg_head: L2 regularization factor for dense head (default: 0)
            dropout: Dropout rate (default: 0.2)
        """
        super(PitomModel, self).__init__()

        self.conv_filters = conv_filters
        self.reg = reg
        self.reg_head = reg_head
        self.dropout = dropout
        self.output_dim = output_dim
        self.output_activation = output_activation
        # Define the CNN architecture
        self.desc = [(conv_filters, 3), ("max", 2), (conv_filters, 2)]

        # Build the layers
        self.layers = nn.ModuleList()

        for i, (filters, kernel_size) in enumerate(self.desc):
            if filters == "max":
                self.layers.append(
                    nn.MaxPool1d(
                        kernel_size=kernel_size,
                        stride=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
            else:
                # Conv block
                conv = nn.Conv1d(
                    in_channels=input_channels if i == 0 else conv_filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=0,  # 'valid' in Keras
                    bias=False,
                )

                # Apply weight decay equivalent to L2 regularization
                self.layers.append(conv)
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(filters))
                self.layers.append(nn.Dropout(dropout))

                input_channels = filters

        # Final locally connected layer (using Conv1d with groups as approximation)
        # Note: True locally connected layers aren't standard in PyTorch
        # This is an approximation that would need to be customized further for exact equivalence
        self.final_conv = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=2,
            stride=1,
            padding=0,  # 'valid' in Keras
            bias=True,
        )

        self.final_bn = nn.BatchNorm1d(conv_filters)
        self.final_act = nn.ReLU()

        # Output layer
        self.dense = nn.Linear(conv_filters, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Apply layers
        for layer in self.layers:
            x = layer(x)

        # Apply final conv block
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_act(x)

        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        # Apply output layer if needed
        x = self.dense(x)
        x = self.layer_norm(x)
        # Apply configurable output activation.
        if self.output_activation == "tanh":
            x = self.tanh(x)
        elif self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        # else: identity / raw logits

        # Squeeze the output to match the label shape [batch_size] instead of [batch_size, 1]
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


class EnsemblePitomModel(nn.Module):
    def __init__(
        self,
        num_models: int,
        input_channels,
        output_dim: int,
        conv_filters=128,
        reg=0.35,
        reg_head=0,
        dropout=0.2,
        output_activation: str = "tanh",
    ):
        """
        PyTorch implementation of the PITOM decoding model.

        Args:
            num_models: The number of models to include in the ensemble. The outputs will be averaged at the end.
            input_channels: Numbr of electrodes in data (int)
            output_dim: Dimensionality of output (int)
            conv_filters: Number of convolutional filters (default: 128)
            reg: L2 regularization factor for convolutional layers (default: 0.35)
            reg_head: L2 regularization factor for dense head (default: 0)
            dropout: Dropout rate (default: 0.2)
        """
        super(EnsemblePitomModel, self).__init__()

        self.models = nn.ModuleList()
        for _ in range(num_models):
            self.models.append(
                PitomModel(
                    input_channels,
                    output_dim,
                    conv_filters=conv_filters,
                    reg=reg,
                    reg_head=reg_head,
                    dropout=dropout,
                    output_activation=output_activation,
                )
            )

    def forward(self, x, preserve_ensemble=False):
        # Run all models and average together all embeddings.
        embeddings = torch.stack([model(x) for model in self.models], dim=1)
        if not preserve_ensemble:
            embeddings = embeddings.mean(1)
        return embeddings


# Constructors
@registry.register_model_constructor()
def pitom_model(model_params):
    return PitomModel(
        input_channels=model_params["input_channels"],
        output_dim=model_params["embedding_dim"],
        conv_filters=model_params["conv_filters"],
        reg=model_params["reg"],
        reg_head=model_params["reg_head"],
        dropout=model_params["dropout"],
        output_activation=model_params.get("output_activation", "tanh"),
    )


@registry.register_model_constructor()
def ensemble_pitom_model(model_params):
    return EnsemblePitomModel(
        num_models=model_params["num_models"],
        input_channels=model_params["input_channels"],
        output_dim=model_params["embedding_dim"],
        conv_filters=model_params["conv_filters"],
        reg=model_params["reg"],
        reg_head=model_params["reg_head"],
        dropout=model_params["dropout"],
        output_activation=model_params.get("output_activation", "tanh"),
    )
