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
from typing import Optional, Tuple


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


class MultiSubjBrainPositionalEncoding(nn.Module):
    """
    Multi-subject brain positional encoding using LIP coordinates.
    
    This class implements the same positional encoding mechanism as the original
    PopT neuroprobe implementation, where LIP coordinates are used as indices
    to lookup positional encodings from a sinusoidal PE table.
    
    Architecture:
    - Each axis (L, I, P, seq_id) gets d_model/4 dimensions
    - LIP coordinates are used as indices to lookup PE from pre-computed table
    - Final PE is concatenation of LIP PE (3*d_model/4) + seq_id PE (d_model/4)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the model embeddings (must be divisible by 4)
            dropout: Dropout probability
            max_len: Maximum coordinate value (for PE table size)
        """
        super().__init__()
        
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        pe_dim = int(d_model / 4)  # Each axis (L, I, P, seq_id) gets d/4 dimensions
        
        # Create sinusoidal positional encoding table
        pe = torch.zeros(max_len, pe_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, pe_dim]
        self.register_buffer('pe', pe)
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, seq: torch.Tensor, positions: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass matching original PopT neuroprobe implementation.
        
        Args:
            seq: Input embeddings [batch_size, num_channels, d_model] (channels as sequence)
            positions: Tuple of (coords, seq_id)
                - coords: [batch_size, num_channels, 3] LongTensor (LIP coordinates)
                - seq_id: [batch_size, num_channels] LongTensor (sequence IDs, typically all 0)
        
        Returns:
            Tuple of (output, input_embeddings)
                - output: [batch_size, num_channels+1, d_model] - seq + positional encoding (with CLS token)
                - input_embeddings: [batch_size, num_channels+1, d_model] - positional encoding only
        
        Note: In original PopT, channels are treated as sequence. This matches that behavior.
        """
        coords, seq_id = positions
        
        # coords: [batch_size, num_channels, 3]
        # seq_id: [batch_size, num_channels]
        # seq: [batch_size, num_channels, d_model] (channels as sequence)
        
        batch_size, num_channels, d_model = seq.shape
        
        # Ensure coords and seq_id are within bounds
        coords = torch.clamp(coords, 0, self.max_len - 1)
        seq_id = torch.clamp(seq_id, 0, self.max_len - 1)
        
        # 1. LIP coordinates as indices to lookup PE
        p_embed = self.pe[0, coords]  # [batch_size, num_channels, 3, pe_dim]
        
        # 2. Reshape: flatten the 3 axes
        n_batch, n_channels, n_axes, d_p_embed = p_embed.shape
        p_embed = p_embed.reshape(n_batch, n_channels, n_axes * d_p_embed)
        # [batch_size, num_channels, 3*pe_dim] = [batch_size, num_channels, 3*d_model/4]
        
        # 3. Sequence ID PE lookup
        seq_id_pe = self.pe[0, seq_id]  # [batch_size, num_channels, pe_dim] = [batch_size, num_channels, d_model/4]
        
        # 4. Concatenate LIP PE and seq_id PE for each channel
        channel_pe = torch.cat([p_embed, seq_id_pe], dim=-1)
        # [batch_size, num_channels, d_model]
        
        # 5. CLS token PE (use index 0, repeat 4 times to get d_model dimensions)
        # Original PopT uses index 0 PE repeated 4 times for CLS token
        cls_pe = self.pe[0, 0].repeat(batch_size, 4).unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 6. Concatenate CLS PE with channel PEs
        input_embeddings = torch.cat([cls_pe, channel_pe], dim=1)
        # [batch_size, num_channels+1, d_model]
        
        # 7. Add positional encoding to input embeddings
        # seq already contains CLS token (added in SimpleTransformer.forward())
        out = seq + input_embeddings
        return self.dropout(out), input_embeddings


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
        use_lip_coords: bool = False,
        max_coord_value: int = 5000,
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
            use_lip_coords: If True, use LIP coordinates for positional encoding
            max_coord_value: Maximum coordinate value for PE table (when using LIP coords)
        """
        super().__init__()

        self.input_channels = input_channels
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_lip_coords = use_lip_coords

        # Project input to model dimension
        # Input shape: [batch_size, num_channels, seq_len]
        # We want: [seq_len, batch_size, model_dim]
        self.input_projection = nn.Linear(input_channels, model_dim)

        # Positional encoding
        if use_lip_coords:
            # Use LIP coordinate-based positional encoding
            self.pos_encoder = MultiSubjBrainPositionalEncoding(
                model_dim, dropout, max_coord_value
            )
        else:
            # Use standard sequence-based positional encoding
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
        lip_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Matches original PopT neuroprobe implementation:
        - Time dimension is aggregated (mean) first (if raw signal)
        - Channels are treated as sequence
        - CLS token is added before positional encoding

        Args:
            x: Input tensor of shape:
               - BrainBERT output: [batch_size, num_channels, model_dim] (e.g., [batch, channels, 768])
               - Raw signal: [batch_size, num_channels, seq_len] (time dimension present)
            return_sequence: If True, return full sequence output [batch_size, num_channels+1, model_dim]
                           If False, return aggregated output [batch_size, model_dim] (CLS token)
            lip_coords: LIP coordinates [batch_size, num_channels, 3] LongTensor (required if use_lip_coords=True)

        Returns:
            Transformed features, either aggregated or full sequence
        """
        batch_size, num_channels, *rest = x.shape
        
        # Determine if input is BrainBERT output (already aggregated) or raw signal
        if len(rest) == 0:
            # BrainBERT output: [batch, channels, model_dim] - already processed
            # This shouldn't happen as input_projection expects input_channels, not model_dim
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected [batch, channels, input_channels] or [batch, channels, time]")
        elif len(rest) == 1:
            # Check if this is BrainBERT output (input_channels dimension) or raw signal (time dimension)
            third_dim = rest[0]
            if third_dim == self.input_channels:
                # BrainBERT output: [batch, channels, input_channels] (e.g., [batch, channels, 768])
                # Already aggregated, just need to project to model_dim
                x_emb = self.input_projection(x)  # [batch, channels, model_dim]
            else:
                # Raw signal: [batch, channels, time]
                seq_len = third_dim
                # Process time dimension: aggregate and project
                x_aggregated = x.mean(dim=2, keepdim=True)  # [batch, channels, 1]
                x_emb = self.input_projection(x_aggregated)  # [batch, channels, model_dim]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Validate LIP coordinates if needed
        if self.use_lip_coords:
            if lip_coords is None:
                raise ValueError("lip_coords is required when use_lip_coords=True")
            if lip_coords.shape != (batch_size, num_channels, 3):
                raise ValueError(f"lip_coords shape mismatch: expected {(batch_size, num_channels, 3)}, "
                               f"got {lip_coords.shape}")
            # Ensure LongTensor
            lip_coords = lip_coords.long()
        
        # x_emb is now [batch_size, num_channels, model_dim] - ready for CLS token and PE
        x = x_emb

        # Step 5: Add CLS token (matching original PopT.forward())
        # Original PopT adds CLS token before positional encoding
        cls_token = torch.ones(batch_size, 1, self.model_dim, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [batch_size, num_channels+1, model_dim]

        # Step 6: Add positional encoding
        if self.use_lip_coords:
            # For LIP-based PE, prepare positions tuple
            # lip_coords: [batch_size, num_channels, 3] (excluding CLS)
            # seq_id: [batch_size, num_channels] (all zeros, excluding CLS)
            seq_id = torch.zeros(batch_size, num_channels, dtype=torch.long, device=x.device)
            positions = (lip_coords, seq_id)
            
            # MultiSubjBrainPositionalEncoding expects [batch, num_channels+1, dim] (including CLS)
            # and returns (output, pe) tuple
            x, _ = self.pos_encoder(x, positions)
            # x: [batch_size, num_channels+1, model_dim]
        else:
            # Standard sequence-based PE (channels as sequence)
            # CLS token already added above
            
            # Transpose to [seq_len, batch_size, model_dim] for transformer
            x = x.transpose(0, 1)  # [num_channels+1, batch_size, model_dim]
            x = self.pos_encoder(x)
            # Transpose back to [batch_size, seq_len, model_dim]
            x = x.transpose(0, 1)  # [batch_size, num_channels+1, model_dim]

        # Step 6: Pass through transformer encoder
        # Transformer expects [seq_len, batch_size, model_dim]
        x = x.transpose(0, 1)  # [num_channels+1, batch_size, model_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch_size, num_channels+1, model_dim]

        # Step 7: Apply layer normalization
        x = self.layer_norm(x)

        if return_sequence:
            # Return full sequence (including CLS token)
            return x  # [batch_size, num_channels+1, model_dim]
        else:
            # Return CLS token only (matching original PopT behavior)
            return x[:, 0, :]  # [batch_size, model_dim]

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
        use_lip_coords=getattr(config, 'use_lip_coords', False),
        max_coord_value=getattr(config, 'max_coord_value', 5000),
    )

def load_pretrained_model(model_dir: str, device: str = "cpu") -> SimpleTransformer:
    """
    Load a pretrained PopT model with a clean, table-based adaptation logic.
    """
    import os
    from .config import load_config

    # --- 1. Configuration & Model Initialization ---
    config_path = os.path.join(model_dir, "config.yaml")
    config = load_config(config_path)
    model = create_model_from_config(config)

    # --- 2. Checkpoint Discovery (Fail Fast) ---
    ckpt_name = getattr(config, 'checkpoint_file', 'checkpoint.pth')
    checkpoint_path = os.path.join(model_dir, ckpt_name)
    candidates = [f for f in os.listdir(model_dir) if f.startswith("pretrained_popt") and f.endswith(".pth")]
    if candidates:
        checkpoint_path = os.path.join(model_dir, candidates[0])
    else:
        raise FileNotFoundError(f"FATAL: No checkpoint found in {model_dir}")

    print(f"Loading weights from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to load checkpoint. {e}")

    # --- 3. Unwrap State Dict ---
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    # --- 4. Clean Mapping & Adaptation (The Clean Way) ---
    new_state_dict = {}
    
    # Mapping table: old key names -> new key names
    # Keys not in this table are kept as-is
    prefix_map = {
        "input_encoding.in_proj": "input_projection",
        "input_encoding.positional_encoding": "pos_encoder",
        "input_encoding.layer_norm": "layer_norm",
        "encoder.": "transformer_encoder.",       # Candidate 1
        "transformer.": "transformer_encoder.",   # Candidate 2 (Likely this one!)
    }

    # Iterate through all checkpoint keys and apply mapping
    for k, v in state_dict.items():
        # (A) Skip PE (Positional Encoding) to avoid dimension conflicts and regenerate mathematically
        if "pos_encoder.pe" in k or "positional_encoding.pe" in k:
            continue

        # (B) Apply name mapping
        new_k = k
        for old_prefix, new_prefix in prefix_map.items():
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, new_prefix)
                break
        
        # (C) Automatic shape transpose
        # If dimension order is flipped ([1, S, D] vs [S, 1, D]), automatically fix it
        # Only check if the key exists in the model
        model_param = dict(model.named_parameters()).get(new_k)
        if model_param is not None and v.ndim == 3:
             if v.shape != model_param.shape:
                if v.shape[0] == 1 and v.shape[1] == model_param.shape[0]:
                    v = v.transpose(0, 1)

        new_state_dict[new_k] = v

    # --- 5. Load & Verify ---
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    # Verify that critical components (Transformer Encoder, etc.) are loaded
    critical_components = ["transformer_encoder", "input_projection"]
    critical_missing = [k for k in missing if any(c in k for c in critical_components)]

    if critical_missing:
        # Diagnostic mode: show what keys were actually in the checkpoint when error occurs
        print(f"\n{'!'*80}")
        print("CRITICAL FAILURE: Mapping failed. The checkpoint keys do not match expected names.")
        print(f"Missing Critical Keys: {critical_missing[:3]} ...")
        print("-" * 40)
        print("Available Top-Level Keys in Checkpoint (First 10):")
        # Show only top-level key prefixes (deduplicated) to understand structure
        top_keys = sorted(list(set([k.split('.')[0] for k in state_dict.keys()])))
        print(top_keys[:10])
        print(f"{'!'*80}\n")
        
        raise RuntimeError("FATAL: Critical weights missing. See log above for available keys.")

    print("✅ Successfully loaded PopT model.")
    model.to(device)
    model.eval()
    return model