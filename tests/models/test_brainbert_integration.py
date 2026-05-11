import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import DataParams, ExperimentConfig, ModelSpec, TaskConfig
from models.brainbert.integration import (
    ReferenceBrainBERTDecoder,
    _adaptive_avg_pool_temporal_patches,
    set_finetuning_config,
)


def test_adaptive_avg_pool_temporal_patches_exact_length_noop():
    features = torch.arange(2 * 10 * 3, dtype=torch.float32).view(2, 10, 3)

    pooled = _adaptive_avg_pool_temporal_patches(features, 10)

    torch.testing.assert_close(pooled, features)


def test_adaptive_avg_pool_temporal_patches_downsamples_with_means():
    features = torch.arange(1 * 8 * 1, dtype=torch.float32).view(1, 8, 1)

    pooled = _adaptive_avg_pool_temporal_patches(features, 4)

    expected = torch.tensor([[[0.5], [2.5], [4.5], [6.5]]])
    torch.testing.assert_close(pooled, expected)


def test_adaptive_avg_pool_temporal_patches_uses_all_tokens():
    features = torch.arange(1 * 13 * 2, dtype=torch.float32).view(1, 13, 2)

    pooled = _adaptive_avg_pool_temporal_patches(features, 10)

    expected = F.adaptive_avg_pool1d(features.transpose(1, 2), 10).transpose(1, 2)
    torch.testing.assert_close(pooled, expected)


def test_adaptive_avg_pool_temporal_patches_can_expand_to_requested_length():
    features = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)

    pooled = _adaptive_avg_pool_temporal_patches(features, 5)

    assert pooled.shape == (2, 5, 4)


class StubBrainBERTUpstream(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, inputs, pad_mask, intermediate_rep=False):
        assert intermediate_rep is True
        assert pad_mask is None
        return torch.arange(
            inputs.shape[0] * self.seq_len * self.hidden_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        ).view(inputs.shape[0], self.seq_len, self.hidden_dim)


class StubFinetuneModel(nn.Module):
    def __init__(self, upstream):
        super().__init__()
        self.upstream = upstream
        self.frozen_upstream = False


def test_brainbert_decoder_pools_temporal_tokens_flattened():
    batch_size = 2
    num_electrodes = 3
    temporal_patches_to_keep = 4
    hidden_dim = 5
    decoder = ReferenceBrainBERTDecoder(
        StubFinetuneModel(StubBrainBERTUpstream(seq_len=8, hidden_dim=hidden_dim)),
        num_electrodes=num_electrodes,
        hidden_dim=hidden_dim,
        temporal_patches_to_keep=temporal_patches_to_keep,
    )
    x = torch.zeros(batch_size, num_electrodes, 6, 7)

    features = decoder(x, return_feature_emb_instead_of_projection=True)

    assert features.shape == (
        batch_size,
        num_electrodes * temporal_patches_to_keep * hidden_dim,
    )


class StubRaw:
    ch_names = ["E1", "E2", "E3"]
    info = {"sfreq": 2048}


def test_brainbert_config_setter_uses_explicit_nested_stft_params():
    config = ExperimentConfig(
        model_spec=ModelSpec(
            constructor_name="brainbert_finetune",
            params={"output_dim": 5},
        ),
        task_config=TaskConfig(
            data_params=DataParams(
                target_sr=1000,
                preprocessing_fn_name=["disk_cache_preprocessor"],
                preprocessor_params=[
                    {
                        "base_preprocessing_fn_name": [
                            "stft_preprocessing",
                            "foundation_feature_cache",
                        ],
                        "base_preprocessor_params": [
                            {"freq_channel_cutoff": 32},
                            {"mode": "normal"},
                        ],
                    }
                ],
            ),
        ),
    )

    result = set_finetuning_config(config, [StubRaw()], None)
    wrapper_params = result.task_config.data_params.preprocessor_params[0]
    stft_params = wrapper_params["base_preprocessor_params"][0]

    assert wrapper_params["base_preprocessing_fn_name"] == [
        "stft_preprocessing",
        "foundation_feature_cache",
    ]
    assert stft_params["freq_channel_cutoff"] == 32
    assert stft_params["fs"] == 1000
    assert stft_params["nperseg"] == 400
    assert result.model_spec.params["input_channels"] == 32


def test_brainbert_config_setter_requires_explicit_stft_preprocessor():
    config = ExperimentConfig(
        model_spec=ModelSpec(
            constructor_name="brainbert_finetune",
            params={"output_dim": 5},
        ),
        task_config=TaskConfig(data_params=DataParams()),
    )

    with pytest.raises(ValueError, match="requires an explicit 'stft_preprocessing'"):
        set_finetuning_config(config, [StubRaw()], None)
