import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# Lazy imports for ecog_foundation_model - only imported when needed
# from ecog_foundation_model.config import VideoMAEExperimentConfig
# from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
from foundation_model.foundation_decoder_utils import create_foundation_model
import registry


def freeze_encoder_blocks(model, num_unfrozen_blocks: int = 1):
    """
    Freeze all encoder blocks except the last `num_unfrozen_blocks`.

    Args:
        model: MaskedAutoencoderViT instance
        num_unfrozen_blocks: Number of final encoder blocks to keep trainable
    """
    # Lazy import to avoid dependency issues
    from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
    if not isinstance(model, MaskedAutoencoderViT):
        raise TypeError(f"Expected MaskedAutoencoderViT, got {type(model)}")
    num_blocks = len(model.blocks)
    assert (
        0 <= num_unfrozen_blocks <= num_blocks
    ), f"num_unfrozen_blocks should be between 0 and {num_blocks}"

    for i, block in enumerate(model.blocks):
        requires_grad = i >= (num_blocks - num_unfrozen_blocks)
        for param in block.parameters():
            param.requires_grad = requires_grad

    # Optionally freeze patch_embed and position embeddings
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    if hasattr(model, "cls_token"):
        model.cls_token.requires_grad = False
    if model.sep_pos_embed:
        for pos_embed in [model.pos_embed_spatial, model.pos_embed_temporal]:
            pos_embed.requires_grad = False
        if hasattr(model, "pos_embed_class"):
            model.pos_embed_class.requires_grad = False
    else:
        model.pos_embed.requires_grad = False


def create_and_freeze_foundation_model(
    foundation_model_config, model_dir, freeze_foundation_model, num_unfrozen_blocks
):
    # Lazy import to avoid dependency issues
    try:
    foundation_model = create_foundation_model(foundation_model_config, model_dir)
    if freeze_foundation_model:
        freeze_encoder_blocks(foundation_model, num_unfrozen_blocks)
    return foundation_model
    except ImportError as e:
        if "ecog_foundation_model" in str(e):
            raise ImportError("ecog_foundation_model is required for foundation model finetuning. Please install it or use a non-finetuning model.") from e
        raise


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=False,
        norm_embedding=False,
    ):
        """
        Initialize a Multi-Layer Perceptron with configurable architecture and LayerNorm.

        Args:
            layer_sizes (list): List of integers specifying the size of each layer.
                               First element is input size, last element is output size.
            activation (function): Activation function to use between layers (default: ReLU).
            dropout_rate (float): Dropout probability for regularization (default: 0.).
            use_layer_norm (bool): Whether to use LayerNorm after each hidden layer (default: True).
            norm_embedding (bool): Whether to normalize the output embedding.
        """
        super(MLP, self).__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.use_layer_norm = use_layer_norm
        self.norm_embedding = norm_embedding

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

        if self.norm_embedding:
            x = F.normalize(x, dim=1)
        return x


class FoundationModelFullAttentionDecoder(nn.Module):
    """
    Attention decoder that collapses N latents into a single latent via a learnable query vector per head.
    """

    def __init__(
        self,
        mlp_layer_sizes,
        mlp_activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=True,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        norm_embedding=False,
    ):
        super().__init__()
        dim = mlp_layer_sizes[0]
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Learnable query vector per head (no projection from x)
        self.query = nn.Parameter(torch.randn(num_heads, head_dim))

        # Only key and value projections
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # MLP expects input of shape [batch, embed_dim]
        self.mlp = MLP(
            mlp_layer_sizes,
            mlp_activation,
            dropout_rate,
            use_layer_norm,
            norm_embedding,
        )

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # Project keys and values: [B, heads, N, head_dim]
        k = self.k(x).view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # Expand query: [B, heads, 1, head_dim]
        q = self.query.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)

        # Compute attention scores and weights: [B, heads, 1, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum over values: [B, heads, 1, head_dim]
        out = attn @ v

        # Collapse to single embedding per example: [B, C]
        out = out.transpose(1, 2).reshape(B, C)

        # Pass through MLP
        return self.mlp(out)


class FoundationModelMLP(nn.Module):
    def __init__(
        self,
        mlp_layer_sizes,
        model_dir=None,
        mlp_activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=True,
        finetune=False,
        foundation_model_config=None,  # Optional[VideoMAEExperimentConfig] = None,
        freeze_foundation_model: bool = False,
        num_unfrozen_blocks: int = 0,
        norm_embedding: bool = False,
    ):
        super(FoundationModelMLP, self).__init__()
        self.finetune = finetune
        if finetune:
            self.foundation_model = create_and_freeze_foundation_model(
                foundation_model_config,
                model_dir,
                freeze_foundation_model,
                num_unfrozen_blocks,
            )

        self.mlp = MLP(
            mlp_layer_sizes,
            mlp_activation,
            dropout_rate,
            use_layer_norm,
            norm_embedding,
        )

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
        foundation_model_config=None,  # Optional[VideoMAEExperimentConfig] = None,
        freeze_foundation_model: bool = False,
        num_unfrozen_blocks: int = 0,
        norm_embedding: bool = False,
    ):
        super().__init__()
        self.finetune = finetune
        if finetune:
            self.foundation_model = create_and_freeze_foundation_model(
                foundation_model_config,
                model_dir,
                freeze_foundation_model,
                num_unfrozen_blocks,
            )

        self.query = nn.Parameter(torch.randn(1, mlp_layer_sizes[0]))  # 1 x C
        self.mlp = MLP(
            mlp_layer_sizes,
            mlp_activation,
            dropout_rate,
            use_layer_norm,
            norm_embedding,
        )

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
    return FoundationModelMLP(
        model_params["layer_sizes"],
        finetune=False,
        norm_embedding=model_params.get("norm_embedding", False),
    )


@registry.register_model_constructor()
def foundation_model_finetune_mlp(model_params):
    return FoundationModelMLP(
        model_params["mlp_layer_sizes"],
        model_dir=model_params.get("model_dir"),
        foundation_model_config=model_params["foundation_model_config"],
        finetune=True,
        freeze_foundation_model=model_params.get("freeze_foundation_model", False),
        num_unfrozen_blocks=model_params.get("num_unfrozen_blocks", 0),
        norm_embedding=model_params.get("norm_embedding", False),
    )


@registry.register_model_constructor()
def foundation_model_finetune_attention(model_params):
    return FoundationModelAttentionPoolingDecoder(
        model_params["mlp_layer_sizes"],
        model_dir=model_params.get("model_dir"),
        foundation_model_config=model_params["foundation_model_config"],
        finetune=True,
        freeze_foundation_model=model_params.get("freeze_foundation_model", False),
        num_unfrozen_blocks=model_params.get("num_unfrozen_blocks", 0),
        norm_embedding=model_params.get("norm_embedding", False),
    )


@registry.register_model_constructor()
def foundation_model_attention(model_params):
    return FoundationModelAttentionPoolingDecoder(
        model_params["layer_sizes"],
        finetune=False,
        norm_embedding=model_params.get("norm_embedding", False),
    )


@registry.register_model_constructor()
def foundation_model_full_attention(model_params):
    return FoundationModelFullAttentionDecoder(
        model_params["layer_sizes"],
        num_heads=model_params["num_heads"],
        norm_embedding=model_params.get("norm_embedding", False),
    )
