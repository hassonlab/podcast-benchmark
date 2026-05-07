import torch
import torch.nn as nn

from core import registry
from models.shared_model_helpers import apply_activation


class TorchLinearModel(nn.Module):
    """Simple linear readout implemented in PyTorch.

    Behavior:
      - Expects input shaped (B, C, T).
      - Pools across time with adaptive average pooling to produce (B, C).
      - Applies a single Linear layer mapping channels -> output_dim.

    Regularization:
      - L2 regularization is handled by the optimizer's weight_decay.
        Set `training_params.weight_decay` to the desired ridge lambda.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        bias: bool = True,
        activation: str = "linear",
    ):
        super(TorchRidgeModel, self).__init__()
        self.input_channels = int(input_channels)
        self.output_dim = int(output_dim)
        self.activation = activation

        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.input_channels, self.output_dim, bias=bias)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        # Support both time-series input (B, C, T), preprocessed RMS input (B, C),
        # and the common swapped layout (B, T, C).
        if x.ndim == 3:
            # If channels are in the last axis (B, T, C) and we know expected
            # input channels, permute to (B, C, T).
            n, a, b = x.shape
            if a != self.input_channels and b == self.input_channels:
                x = x.permute(0, 2, 1).contiguous()

            # x: (B, C, T) -> pool -> (B, C)
            x = self.time_pool(x).squeeze(-1)
        elif x.ndim == 2:
            # x already reduced to per-channel features (B, C)
            pass
        else:
            raise ValueError(
                f"TorchRidgeModel expects 2D or 3D input (B,C) or (B,C,T), got {x.shape}"
            )

        out = self.linear(x)

        out = apply_activation(out, self.activation)
        # Squeeze final dim for scalar outputs
        if out.shape[1] == 1:
            return out.squeeze(1)
        return out


@registry.register_model_constructor()
def torch_linear_model(model_params):
    return TorchLinearModel(
        input_channels=model_params["input_channels"],
        output_dim=model_params.get("embedding_dim", 1),
        bias=model_params.get("bias", True),
        activation=model_params.get("activation", "linear"),
    )
