"""Tests for generic feature-cache helpers."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import NeuralDictDataset
from utils.decoding_utils import CachedFeatureModel, extract_features_for_caching


class CacheableToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)
        self.encode_calls = 0
        self.forward_from_features_calls = 0
        self.forward_calls = 0

    def encode_features(self, x, **kwargs):
        self.encode_calls += 1
        return self.encoder(x)

    def forward_from_features(self, features, **kwargs):
        self.forward_from_features_calls += 1
        return self.head(features)

    def forward(self, x, **kwargs):
        self.forward_calls += 1
        return self.forward_from_features(self.encode_features(x, **kwargs), **kwargs)


def test_extract_features_for_caching_uses_encode_features_not_forward():
    model = CacheableToyModel()
    x = torch.randn(5, 4)
    target = torch.arange(5)
    ds = NeuralDictDataset(x, {"extra": torch.arange(5)}, target)
    loader = DataLoader(ds, batch_size=2)

    features, input_dicts, y = extract_features_for_caching(
        model, loader, torch.device("cpu")
    )

    assert features.shape == (5, 3)
    assert torch.equal(y, target)
    assert len(input_dicts) == 3
    assert model.encode_calls == 3
    assert model.forward_calls == 0


def test_cached_feature_model_uses_forward_from_features():
    model = CacheableToyModel()
    wrapper = CachedFeatureModel(model)
    features = torch.randn(4, 3)

    out = wrapper(features)

    assert out.shape == (4, 2)
    assert model.forward_from_features_calls == 1
    assert model.forward_calls == 0
