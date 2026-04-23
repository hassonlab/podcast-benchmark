from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core import registry


def _slice_batch(value: Any, indices: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, np.ndarray):
        return value[indices.cpu().numpy()]
    if isinstance(value, list):
        index_list = indices.cpu().tolist()
        return [value[i] for i in index_list]
    if isinstance(value, tuple):
        index_list = indices.cpu().tolist()
        return tuple(value[i] for i in index_list)
    return value


def _normalize_cache_keys(cache_key: Any, batch_size: int) -> list[Any]:
    if cache_key is None:
        return []
    if torch.is_tensor(cache_key):
        keys = cache_key.detach().cpu().tolist()
    elif isinstance(cache_key, np.ndarray):
        keys = cache_key.tolist()
    elif isinstance(cache_key, Sequence) and not isinstance(cache_key, (str, bytes)):
        keys = list(cache_key)
    else:
        keys = [cache_key]

    if len(keys) != batch_size:
        raise ValueError(
            f"cache_key length must match batch size. Got {len(keys)} keys for batch size {batch_size}."
        )
    return keys


class CachingModel(nn.Module):
    def __init__(self, inner_model: nn.Module, cache_store: dict[Any, torch.Tensor] | None = None):
        super().__init__()
        self.inner_model = inner_model
        self._cache = cache_store if cache_store is not None else {}
        self._hits = 0
        self._misses = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def forward(self, x, cache_key=None, **kwargs):
        if cache_key is None:
            return self.inner_model(x, **kwargs)

        batch_size = x.shape[0]
        cache_keys = _normalize_cache_keys(cache_key, batch_size)
        device = x.device if torch.is_tensor(x) else None

        outputs: list[torch.Tensor | None] = [None] * batch_size
        missing_positions: list[int] = []
        missing_keys: list[Any] = []

        for idx, key in enumerate(cache_keys):
            cached = self._cache.get(key)
            if cached is None:
                missing_positions.append(idx)
                missing_keys.append(key)
                continue

            self._hits += 1
            outputs[idx] = cached.to(device=device) if device is not None else cached

        if missing_positions:
            miss_idx = torch.tensor(missing_positions, dtype=torch.long)
            sliced_x = _slice_batch(x, miss_idx)
            sliced_kwargs = {
                name: _slice_batch(value, miss_idx)
                for name, value in kwargs.items()
            }
            computed = self.inner_model(sliced_x, **sliced_kwargs)
            if not torch.is_tensor(computed):
                raise TypeError(
                    "CachingModel only supports tensor outputs from the wrapped model."
                )

            for out_idx, key, value in zip(missing_positions, missing_keys, computed):
                detached = value.detach().cpu()
                self._cache[key] = detached
                outputs[out_idx] = value
                self._misses += 1

        return torch.stack(outputs, dim=0)


@registry.register_model_constructor("caching_model")
def caching_model_constructor(model_params):
    inner_model = model_params["inner_model"]
    cache_store = model_params.get("_cache_store")
    return CachingModel(inner_model=inner_model, cache_store=cache_store)
