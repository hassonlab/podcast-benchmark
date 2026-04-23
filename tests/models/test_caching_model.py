import torch
import torch.nn as nn

from models.caching import CachingModel


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, **kwargs):
        self.calls += x.shape[0]
        return x.mean(dim=-1)


def test_caching_model_hits_and_detaches():
    encoder = RecordingEncoder()
    model = CachingModel(encoder)

    x = torch.randn(2, 3, 4)
    keys = torch.tensor([10, 11])

    first = model(x, cache_key=keys)
    second = model(x, cache_key=keys)

    assert encoder.calls == 2
    assert torch.allclose(first, second)
    assert model.cache_stats() == {"hits": 2, "misses": 2, "size": 2}
    for cached in model._cache.values():
        assert cached.device.type == "cpu"
        assert cached.requires_grad is False


def test_caching_model_clear_cache():
    encoder = RecordingEncoder()
    model = CachingModel(encoder)
    x = torch.randn(1, 3, 4)

    model(x, cache_key=torch.tensor([1]))
    model.clear_cache()

    assert model.cache_stats() == {"hits": 0, "misses": 0, "size": 0}


def test_shared_cache_store_across_wrapper_instances():
    shared_store = {}
    first_encoder = RecordingEncoder()
    second_encoder = RecordingEncoder()

    first_model = CachingModel(first_encoder, cache_store=shared_store)
    second_model = CachingModel(second_encoder, cache_store=shared_store)

    x = torch.randn(2, 3, 4)
    keys = torch.tensor([0, 1])

    first_model(x, cache_key=keys)
    second_model(x, cache_key=keys)

    assert first_encoder.calls == 2
    assert second_encoder.calls == 0
    assert len(shared_store) == 2
