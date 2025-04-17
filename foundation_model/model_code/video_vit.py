# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=32,
        t_patch_size=4,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(
            f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.t_grid_size = frames // t_patch_size

        kernel_size = [t_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        # attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.0,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# TODO: Test this.
class MaskedBatchNorm(nn.Module):
    def __init__(self, n_channels, scale_factor=1e6):
        """
        Initialize norm class.

        n_channels: The number of channels (bands) in our data.
        scale_factor: How much to scale the input data by before normalizing. Set value so that
            the variance is roughly >0.1 and <~ 100. Needed with our data to maintain numerical stability.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(n_channels, affine=False)
        self.scale_factor = scale_factor

    def forward(self, x, mask):
        """
        Normalize input along the channel axis while ignoring masked out values. Running statistics are tracked.

        x: batch tensor of shape [batch, channels, frames, height, width].
        mask: tensor of shape [height, width] denoting which electrodes are masked. Those with False will be removed.
        """
        # Code adapted from: https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T * H * W)
        x = x * self.scale_factor

        if mask is None:
            return self.bn(x).view(B, C, T, H, W)

        # We don't want to include padded values in the normalization process so we take advantage of the fact
        # that batch norm does not differentiate between batch and time axis. If we just used the mask on the time
        # axis than we would have a jagged tensor but here we can flatten it out and remove all masked values.
        # Swap time and channel axes to set the batch and time axis next to each other.
        flattened_x = x.permute(0, 2, 1).reshape(B * T * H * W, C, 1)
        # Make mask align with flattened out x array.
        expanded_mask = mask.unsqueeze(0).repeat((B * T, 1, 1)).view(B, 1, T * H * W)
        flattened_mask = expanded_mask.reshape(-1, 1, 1) > 0
        # Apply mask.
        masked_values = torch.masked_select(flattened_x, flattened_mask).reshape(
            -1, C, 1
        )
        normed = self.bn(masked_values)
        # Scatter in normalized values and fix shape back to original batch.
        scattered = flattened_x.masked_scatter(flattened_mask, normed)
        backshaped = (
            scattered.reshape(B, T * H * W, C).permute(0, 2, 1).view(B, C, T, H, W)
        )
        return backshaped
