import torch.nn as nn
import torch.nn.functional as F

from core import registry


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
        output_activation: str = "linear",
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
        self.output_activation = output_activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Don't apply activation/dropout to final layer
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)

        # Apply output activation if specified
        if self.output_activation == "sigmoid":
            import torch
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            import torch
            x = torch.tanh(x)
        elif self.output_activation == "softmax":
            x = F.softmax(x, dim=-1)
        # "linear" means no activation (default)

        # Squeeze the output to match the label shape [batch_size] instead of [batch_size, 1]
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x


@registry.register_model_constructor("linear_probe")
def create_linear_probe(model_params):
    """
    Model-agnostic MLP decoder for frozen foundation features.

    Expected model_params:
        - input_dim: Dimension of input embeddings
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
