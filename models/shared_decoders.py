import torch
import torch.nn as nn
from torch.nn import functional as F

from core import registry


class MLPProbeDecoder(nn.Module):
    """Small MLP probe for cached frozen foundation features."""

    def __init__(
        self,
        input_dim: int | None,
        layer_sizes: list[int],
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        use_layer_norm: bool = False,
        output_activation: str = "linear",
        output_dim: int | None = None,
    ):
        super().__init__()
        if not layer_sizes:
            if output_dim is None:
                raise ValueError("MLPProbeDecoder requires layer_sizes or output_dim.")
            layer_sizes = [output_dim]

        self.output_dim = output_dim or layer_sizes[-1]
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(input_dropout)
        self.use_layer_norm = use_layer_norm
        self.output_activation = output_activation

        prev_dim = input_dim
        for i, size in enumerate(layer_sizes):
            if i == 0 and prev_dim is None:
                self.layers.append(nn.LazyLinear(size))
            else:
                self.layers.append(nn.Linear(prev_dim, size))
            if use_layer_norm and size != layer_sizes[-1]:
                self.layer_norms.append(nn.LayerNorm(size))
            prev_dim = size

    def forward(self, x, **kwargs):
        x = self.input_dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)

        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)
        elif self.output_activation == "softmax":
            x = F.softmax(x, dim=-1)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x

    def forward_from_features(self, features, **kwargs):
        return self(features, **kwargs)


@registry.register_model_constructor("mlp_probe_decoder")
@registry.register_model_constructor("foundation_mlp_probe")
def create_mlp_probe_decoder(model_params):
    output_dim = model_params.get("output_dim")
    layer_sizes = model_params.get("layer_sizes")
    if layer_sizes is None:
        layer_sizes = list(model_params.get("mlp_layer_sizes", []))
        if output_dim is not None:
            layer_sizes.append(output_dim)

    return MLPProbeDecoder(
        input_dim=model_params.get("input_dim"),
        layer_sizes=layer_sizes,
        dropout=model_params.get("dropout", 0.0),
        input_dropout=model_params.get("input_dropout", 0.0),
        use_layer_norm=model_params.get("use_layer_norm", False),
        output_activation=model_params.get("output_activation", "linear"),
        output_dim=output_dim,
    )
