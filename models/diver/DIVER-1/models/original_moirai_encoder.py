import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
import abc
from collections.abc import Callable
import math
from functools import cached_property, partial
from typing import Any, Optional
from einops import einsum, rearrange, repeat


class AttentionBias(nn.Module, abc.ABC):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]: ...


class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))

        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = rearrange( 
            ~ind * weight[:1] + ind * weight[1:],
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        return bias

class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mean_dim = tuple(range(-len(normalized_shape), 0))

        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(
        self, x: Float[torch.Tensor, "*batch normalized_shape"]
    ) -> Float[torch.Tensor, "*batch normalized_shape"]:
        output = x * torch.rsqrt(
            x.pow(2).mean(dim=self.mean_dim, keepdim=True) + self.eps
        )
        if self.weight is not None:
            return output * self.weight
        return output

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"weight={self.weight is not None}"
        )

class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, num_groups: int, **kwargs: Any):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

    @abc.abstractmethod
    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]: ...

class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        proj_layer: type[Projection],
        kwargs: Optional[dict[str, Any]] = None,
        key_proj_layer: Optional[type[Projection]] = None,
        key_kwargs: Optional[dict[str, Any]] = None,
        partial_factor: Optional[tuple[float, float]] = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            num_groups=num_groups,
            **(kwargs or {}),
        )
        if key_proj_layer is None:
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(
                proj_width=self.proj_width,
                num_heads=num_heads,
                num_groups=num_groups,
                **(key_kwargs or {}),
            )

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self) -> tuple[int, int, int]:
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ) -> tuple[
        Float[torch.Tensor, "*batch group hpg seq dim"],
        Float[torch.Tensor, "*batch group hpg seq dim"],
    ]:
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key



class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)

class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self,
        x: Float[torch.Tensor, "... in_dim"],
        centroid: Optional[Float[torch.Tensor, "expert in_dim"]] = None,
    ) -> Float[torch.Tensor, "... out_dim"]:
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self, x: Float[torch.Tensor, "... in_dim"]
    ) -> Float[torch.Tensor, "... out_dim"]:
        return self.activation(self.fc1(x))


class GatedLinearUnitFeedForward(FeedForward):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim):
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(
        self, x: Float[torch.Tensor, "... in_dim"]
    ) -> Float[torch.Tensor, "... out_dim"]:
        return self.activation(self.fc_gate(x)) * self.fc1(x)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        bias: bool = True,
        norm_layer: Optional[type[nn.Module] | partial[nn.Module]] = nn.LayerNorm,
        softmax_scale: Optional[float] = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        time_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        var_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
        time_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
        flash_attention: bool = True, 
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups
        self.var_attn_bias = var_attn_bias() if var_attn_bias is not None else None
        self.time_attn_bias = time_attn_bias() if time_attn_bias is not None else None
        self.var_qk_proj = var_qk_proj() if var_qk_proj is not None else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj is not None else None

        self.softmax_scale = softmax_scale or 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.q_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.attn_dropout_p = attn_dropout_p
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.flash_attention = flash_attention

    def _get_var_id(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_var_id: Optional[Int[torch.Tensor, "*batch q_len"]],
        kv_var_id: Optional[Int[torch.Tensor, "*batch kv_len"]],
    ) -> tuple[
        Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ]:
        if self.var_attn_bias is not None or self.var_qk_proj is not None:
            if query_var_id is None:
                query_var_id = repeat(
                    torch.zeros((), device=query.device, dtype=torch.long),
                    f" -> {' '.join(map(str, query.shape[:-4]))} 1 1 {query.shape[-2]}",
                )
            else:
                query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")

            if kv_var_id is None:
                kv_var_id = repeat(
                    torch.zeros((), device=key.device, dtype=torch.long),
                    f" -> {' '.join(map(str, key.shape[:-4]))} 1 1 {key.shape[-2]}",
                )
            else:
                kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")

        return query_var_id, kv_var_id

    def _get_time_id(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_time_id: Optional[Int[torch.Tensor, "*batch q_len"]],
        kv_time_id: Optional[Int[torch.Tensor, "*batch kv_len"]],
    ) -> tuple[
        Optional[Int[torch.Tensor, "*batch 1 1 q_len"]],
        Optional[Int[torch.Tensor, "*batch 1 1 kv_len"]],
    ]:
        if self.time_attn_bias is not None or self.time_qk_proj is not None:
            if query_time_id is None:
                query_time_id = repeat(
                    torch.arange(
                        query.shape[-2], device=query.device, dtype=torch.long
                    ),
                    f"q_len -> {' '.join(map(str, query.shape[:-4]))} 1 1 q_len",
                )
            else:
                query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")

            if kv_time_id is None:
                kv_time_id = repeat(
                    torch.arange(key.shape[-2], device=key.device, dtype=torch.long),
                    f"kv_len -> {' '.join(map(str, key.shape[:-4]))} 1 1 kv_len",
                )
            else:
                kv_time_id = rearrange(kv_time_id, "... kv_len-> ... 1 1 kv_len")

        return query_time_id, kv_time_id

    def _update_attn_mask(
        self,
        attn_mask: Optional[Bool[torch.Tensor, "*batch q_len kv_len"]],
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_var_id: Optional[Int[torch.Tensor, "*batch 1 1 q_len"]] = None,
        kv_var_id: Optional[Int[torch.Tensor, "*batch 1 1 kv_len"]] = None,
        query_time_id: Optional[Int[torch.Tensor, "*batch 1 1 q_len"]] = None,
        kv_time_id: Optional[Int[torch.Tensor, "*batch 1 1 kv_len"]] = None,
    ) -> Optional[
        Bool[torch.Tensor, "*batch #group #hpg q_len kv_len"]
        | Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]
    ]:
        if attn_mask is not None:
            attn_mask = rearrange(
                attn_mask,
                "... q_len kv_len -> ... 1 1 q_len kv_len",
            )

        attn_bias = 0
        if self.var_attn_bias is not None:
            attn_bias = attn_bias + self.var_attn_bias(
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
            )

        if self.time_attn_bias is not None:
            attn_bias = attn_bias + self.time_attn_bias(
                query,
                key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )

        attn_mask = (
            attn_mask
            if isinstance(attn_bias, int)
            else (
                attn_bias
                if attn_mask is None
                else attn_bias.masked_fill(attn_mask.logical_not(), -1e6)
            )
        )
        return attn_mask

    def _qk_proj(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_var_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_var_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
        query_time_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_time_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ) -> tuple[
        Float[torch.Tensor, "*batch group hpg q_len dim"],
        Float[torch.Tensor, "*batch group hpg kv_len dim"],
    ]:
        if self.var_qk_proj is not None:
            query, key = self.var_qk_proj(
                query, key, query_id=query_var_id, kv_id=kv_var_id
            )

        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key, query_id=query_time_id, kv_id=kv_time_id
            )

        return query, key

    def forward(
        self,
        query: Float[torch.Tensor, "*batch q_len dim"],
        key: Float[torch.Tensor, "*batch kv_len dim"],
        value: Float[torch.Tensor, "*batch kv_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch q_len kv_len"]] = None,
        query_var_id: Optional[Int[torch.Tensor, "*batch q_len"]] = None,
        kv_var_id: Optional[Int[torch.Tensor, "*batch kv_len"]] = None,
        query_time_id: Optional[Int[torch.Tensor, "*batch q_len"]] = None,
        kv_time_id: Optional[Int[torch.Tensor, "*batch kv_len"]] = None,
    ) -> Float[torch.Tensor, "*batch q_len dim"]:
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self.q_norm(
            rearrange(
                query,
                "... q_len (group hpg dim) -> ... group hpg q_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        key = self.k_norm(
            repeat(
                key,
                "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        value = repeat(
            value,
            "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )

        query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )

        attn_mask = self._update_attn_mask(
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        query, key = self._qk_proj(
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        if self.flash_attention:
            query_flat = rearrange(query, "... group hpg q_len dim -> ... (group hpg) q_len dim")
            key_flat = rearrange(key,   "... group hpg kv_len dim -> ... (group hpg) kv_len dim")
            value_flat = rearrange(value, "... group hpg kv_len dim -> ... (group hpg) kv_len dim")
            attn_mask_flat = rearrange(attn_mask, "... group hpg q_len kv_len -> ... (group hpg) q_len kv_len")            
            out = F.scaled_dot_product_attention(
                    query_flat,
                    key_flat,
                    value_flat,
                    attn_mask=attn_mask_flat,
                    dropout_p=self.attn_dropout_p,
                    scale=self.softmax_scale,
            )  
            out = rearrange(out, "... (group hpg) q_len dim -> ... group hpg q_len dim", group=self.num_groups, hpg=self.heads_per_group)
        else:
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout_p,
                scale=self.softmax_scale,
            )
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
        return self.out_proj(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: Optional[nn.Module],
        norm2: Optional[nn.Module],
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        centroid: Optional[Float[torch.Tensor, "expert dim"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        if self.pre_norm:
            x = x + self._sa_block(
                self.norm1(x), attn_mask, var_id=var_id, time_id=time_id
            )
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            x = self.norm1(
                x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id)
            )
            x = self.norm2(x + self.ffn(x, centroid=centroid))
        return x

    def _sa_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]],
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: Optional[int] = None,
        num_groups: Optional[int] = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.LayerNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_moe: bool = False,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        time_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        var_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        time_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: Optional[int] = None,
        use_bias: bool = False, 
        flash_attention: bool = False,
        use_mup: bool = False, 
    ):
        super().__init__()
        self.use_mup = use_mup 
        if not flash_attention:
            print('flash attention is not True')
        self.use_moe = use_moe
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads 

        var_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            var_attn_bias_layer,
            shared_var_attn_bias,
        )
        time_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            time_attn_bias_layer,
            shared_time_attn_bias,
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=use_bias,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None if not use_mup else 1/(d_model // num_heads), 
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
            flash_attention=flash_attention, 
        )
        if not use_moe:
            get_ffn = partial(
                GatedLinearUnitFeedForward if use_glu else FeedForward,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=use_bias,
                ffn_dropout_p=dropout_p,
            )
        get_encoder_layer_norm = partial(norm_layer, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(d_model)

    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable,
        shared_layer: bool,
    ) -> Optional[Callable[[], nn.Module]]:
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        if self.use_moe:
            for idx, layer in enumerate(self.layers):
                x = layer(
                    x,
                    attn_mask,
                    var_id=var_id,
                    time_id=time_id,
                    centroid=self.centroid[idx],
                )
        else:
            for layer in self.layers:
                x = layer(x, attn_mask, var_id=var_id, time_id=time_id)
        return self.norm(x)