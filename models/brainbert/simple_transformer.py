"""
Simple Transformer Foundation Model

This is a basic transformer encoder implementation designed to demonstrate how to
integrate a foundation model with the podcast benchmark framework.

This model is intentionally simple and well-documented to serve as an educational
example for onboarding new users.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to input embeddings using sine and cosine functions.

    This allows the transformer to understand the order of the input sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    A simple transformer encoder model for processing neural data.

    This model can be used in two ways:
    1. Feature extraction (frozen): Extract embeddings from pretrained weights
    2. Finetuning: Continue training the model on downstream tasks

    Architecture:
    - Input projection layer (maps input channels to model dimension)
    - Positional encoding
    - Stack of transformer encoder layers
    - Output aggregation (mean pooling over sequence)
    """

    def __init__(
        self,
        input_channels: int,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        """
        Initialize the transformer model.

        Args:
            input_channels: Number of input channels (e.g., electrodes)
            model_dim: Dimension of the transformer model
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.input_channels = input_channels
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Project input to model dimension
        # Input shape: [batch_size, num_channels, seq_len]
        # We want: [seq_len, batch_size, model_dim]
        self.input_projection = nn.Linear(input_channels, model_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # We use [seq_len, batch_size, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape [batch_size, num_channels, seq_len]
            return_sequence: If True, return full sequence output [batch_size, seq_len, model_dim]
                           If False, return aggregated output [batch_size, model_dim]

        Returns:
            Transformed features, either aggregated or full sequence
        """
        # x shape: [batch_size, num_channels, seq_len]
        batch_size, num_channels, seq_len = x.shape

        # Transpose to [batch_size, seq_len, num_channels]
        x = x.transpose(1, 2)

        # Project to model dimension: [batch_size, seq_len, model_dim]
        x = self.input_projection(x)

        # Transpose to [seq_len, batch_size, model_dim] for transformer
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Transpose back to [batch_size, seq_len, model_dim]
        x = x.transpose(0, 1)

        if return_sequence:
            # Return full sequence
            return x
        else:
            # Aggregate over sequence dimension (mean pooling)
            # Output shape: [batch_size, model_dim]
            return x.mean(dim=1)

    def get_num_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def freeze(self):
        """Freeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_layers(self, num_layers_to_freeze: int):
        """
        Freeze the first N transformer layers, keeping the rest trainable.

        This is useful for partial finetuning where you want to keep early
        layers frozen but adapt later layers to your task.

        Args:
            num_layers_to_freeze: Number of transformer layers to freeze (starting from input)
        """
        # Freeze input projection and positional encoding
        for param in self.input_projection.parameters():
            param.requires_grad = False
        for param in self.pos_encoder.parameters():
            param.requires_grad = False

        # Freeze specified transformer layers
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True


def create_model_from_config(config) -> SimpleTransformer:
    """
    Create a SimpleTransformer model from a config object.

    Args:
        config: Configuration object with model parameters

    Returns:
        Initialized SimpleTransformer model
    """
    return SimpleTransformer(
        input_channels=config.input_channels,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )


def load_pretrained_model(model_dir: str, device: str = "cpu") -> SimpleTransformer:
    """
    Load a pretrained model with key remapping AND shape correction logic.
    """
    import os
    import torch
    from .config import load_config
    
    # 1. Load config
    config_path = os.path.join(model_dir, "config.yaml") 
    config = load_config(config_path)

    # 2. Create model
    model = create_model_from_config(config)

    # 3. Load checkpoint
    ckpt_name = "stft_large_pretrained.pth"
    checkpoint_path = os.path.join(model_dir, ckpt_name)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_dir, "checkpoint.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    print(f">>> Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 4. Extract State Dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            src_state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            src_state_dict = checkpoint["model_state_dict"]
        else:
            src_state_dict = checkpoint
    else:
        src_state_dict = checkpoint

    # 5. Key Remapping & Shape Correction
    new_state_dict = {}
    
    for k, v in src_state_dict.items():
        # (1) 불필요한 Head 제거
        if "spec_prediction_head" in k:
            continue
            
        new_k = k
        new_v = v # 값을 수정할 수 있도록 변수 할당
        
        # (2) Transformer Encoder 이름 변환
        if k.startswith("transformer."):
            new_k = k.replace("transformer.", "transformer_encoder.")
            
        # (3) Input Projection 변환
        elif "input_encoding.in_proj" in k:
            new_k = k.replace("input_encoding.in_proj", "input_projection")
            
        # (4) Positional Encoding 변환 및 SHAPE 수정 (핵심!)
        elif "positional_encoding.pe" in k:
            new_k = "pos_encoder.pe"
            # 에러 원인 해결: [1, 5000, 768] -> [5000, 1, 768]로 변환
            if new_v.shape == (1, 5000, 768): 
                print(f">>> Transposing {k} shape from {new_v.shape} to (5000, 1, 768)")
                new_v = new_v.transpose(0, 1)
        
        # (5) Layer Norm 변환
        elif "input_encoding.layer_norm" in k:
            new_k = k.replace("input_encoding.layer_norm", "layer_norm")

        new_state_dict[new_k] = new_v

    # 6. Load Weights
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(">>> Model weights loaded successfully (Strict match).")
    except RuntimeError as e:
        print(f">>> Warning: Strict loading failed. Retrying with strict=False.\n>>> Error details: {e}")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if any("transformer_encoder" in k for k in missing):
            print(">>> CRITICAL ERROR: Main transformer weights are missing!")
        else:
            print(f">>> Loaded with warnings. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.to(device)
    return model