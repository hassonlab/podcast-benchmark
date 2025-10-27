import torch
import torch.nn as nn
import torch.nn.functional as F

import registry


class SimpleTemporalModel(nn.Module):
    """A small, robust model that pools across time then applies an MLP.

    Expects input shaped (batch, channels, timesteps). This model is safer for
    quick experiments because it does not rely on complex conv architectures
    or locally-connected approximations.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        output_activation: str = "tanh",
    ):
        super(SimpleTemporalModel, self).__init__()
        self.input_channels = int(input_channels)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.output_activation = output_activation

        # Pool across the time axis to produce one descriptor per channel.
        # Input: (B, C, T) -> AdaptiveAvgPool1d(1) -> (B, C, 1) -> squeeze -> (B, C)
        self.time_pool = nn.AdaptiveAvgPool1d(1)

        # Simple two-layer MLP operating on channel features
        self.mlp = nn.Sequential(
            nn.Linear(self.input_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Expect x shape: (B, C, T). Accept common swapped layout (B, T, C)
        if x.ndim != 3:
            raise ValueError(f"SimpleTemporalModel expects 3D input (B,C,T), got {x.shape}")

        # If model_params.input_channels is known, detect and fix axes swap where
        # the caller passed (B, T, C) instead of (B, C, T).
        try:
            n, a, b = x.shape
            if a != self.input_channels and b == self.input_channels:
                # Detected swapped axes: permute to [B, C, T]
                x = x.permute(0, 2, 1).contiguous()
            elif a != self.input_channels:
                raise ValueError(
                    f"SimpleTemporalModel expects {self.input_channels} channels but input has shape {tuple(x.shape)}"
                )
        except Exception:
            raise

        # Pool temporal dimension
        x = self.time_pool(x).squeeze(-1)  # -> (B, C)

        out = self.mlp(x)

        if self.output_activation == "tanh":
            out = self.tanh(out)
        elif self.output_activation == "sigmoid":
            out = torch.sigmoid(out)

        # If scalar output, return shape (B,) to match labels
        if out.shape[1] == 1:
            return out.squeeze(1)
        return out


@registry.register_model_constructor()
def simple_temporal_model(model_params):
    return SimpleTemporalModel(
        input_channels=model_params["input_channels"],
        output_dim=model_params.get("embedding_dim", 1),
        hidden_dim=model_params.get("hidden_dim", 128),
        dropout=model_params.get("dropout", 0.2),
        output_activation=model_params.get("output_activation", "tanh"),
    )
