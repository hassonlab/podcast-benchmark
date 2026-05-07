"""Tests for DIVER integration module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.diver.integration import DIVERDecoder, create_data_info_list


class MockDiverBackbone(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.projection = nn.Linear(1, feature_dim)

    def forward(self, x, **kwargs):
        return self.projection(x.mean(dim=-1, keepdim=True))


class MockDiverModel(nn.Module):
    """Mock DIVER model with explicit feature/readout modules."""

    def __init__(self, feature_dim=7, output_dim=5):
        super().__init__()
        self.output_dim = output_dim
        self.last_data_info_list = None
        self.backbone = MockDiverBackbone(feature_dim)
        self.ft_core_model = nn.Linear(feature_dim, feature_dim)
        self.ft_model_output_adapter = nn.Linear(feature_dim, output_dim)

    def feature_extraction_func(self, x, data_info_list=None):
        self.last_data_info_list = data_info_list
        return x

    def ft_model_input_adapter(self, x, data_info_list=None):
        self.last_data_info_list = data_info_list
        return x.mean(dim=1)

    def forward(self, x, data_info_list=None):
        features = self.ft_model_input_adapter(
            self.feature_extraction_func(
                self.backbone(
                    x,
                    data_info_list=data_info_list,
                    use_mask=False,
                    return_encoder_output=True,
                ),
                data_info_list=data_info_list,
            ),
            data_info_list=data_info_list,
        )
        return self.ft_model_output_adapter(self.ft_core_model(features))


class MockSplitDiverModel(MockDiverModel):
    """Named alias for tests that assert the feature/readout split directly."""


class TestDIVERDecoderForwardWithXyzId:
    """Test DIVERDecoder.forward correctly handles xyz_id kwarg."""

    def test_forward_builds_data_info_list_from_xyz_id(self):
        """Test that xyz_id is correctly converted to data_info_list."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="linear")

        batch_size = 4
        num_channels = 10
        seq_len = 100

        # Create input tensor
        x = torch.randn(batch_size, num_channels, seq_len)

        # Create xyz_id: [batch_size, num_channels, 3]
        xyz_id = np.random.randint(-50, 50, size=(batch_size, num_channels, 3))

        # Call forward with xyz_id
        _ = decoder(x, xyz_id=xyz_id)

        # Verify data_info_list was built correctly
        assert mock_model.last_data_info_list is not None
        assert len(mock_model.last_data_info_list) == batch_size

        for i, info in enumerate(mock_model.last_data_info_list):
            assert info["num_channels"] == num_channels
            assert info["modality"] == "iEEG"
            assert "xyz_id" in info
            np.testing.assert_array_equal(info["xyz_id"], xyz_id[i])

    def test_forward_handles_xyz_id_as_tensor(self):
        """Test that xyz_id works when passed as a torch tensor."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="linear")

        batch_size = 2
        num_channels = 8
        seq_len = 50

        x = torch.randn(batch_size, num_channels, seq_len)
        xyz_id = torch.randint(-50, 50, size=(batch_size, num_channels, 3))

        _ = decoder(x, xyz_id=xyz_id)

        assert mock_model.last_data_info_list is not None
        assert len(mock_model.last_data_info_list) == batch_size

        for i, info in enumerate(mock_model.last_data_info_list):
            # xyz_id should be converted to numpy
            assert isinstance(info["xyz_id"], np.ndarray)
            np.testing.assert_array_equal(info["xyz_id"], xyz_id[i].numpy())

    def test_forward_without_xyz_id_uses_default(self):
        """Test that forward works without xyz_id (uses default data_info_list)."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="linear")

        batch_size = 3
        num_channels = 12
        seq_len = 80

        x = torch.randn(batch_size, num_channels, seq_len)

        _ = decoder(x)

        # Should have created default data_info_list
        assert mock_model.last_data_info_list is not None
        assert len(mock_model.last_data_info_list) == batch_size

        for info in mock_model.last_data_info_list:
            assert info["num_channels"] == num_channels
            assert info["modality"] == "iEEG"
            # Default doesn't include xyz_id
            assert "xyz_id" not in info

    def test_forward_legacy_data_info_list_still_works(self):
        """Test backward compatibility with legacy data_info_list kwarg."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="linear")

        batch_size = 2
        num_channels = 6
        seq_len = 40

        x = torch.randn(batch_size, num_channels, seq_len)

        # Create legacy data_info_list directly
        legacy_data_info_list = [
            {"num_channels": num_channels, "modality": "iEEG", "custom_field": i}
            for i in range(batch_size)
        ]

        _ = decoder(x, data_info_list=legacy_data_info_list)

        # Should pass through unchanged
        assert mock_model.last_data_info_list is legacy_data_info_list

    def test_forward_xyz_id_takes_precedence_over_data_info_list(self):
        """Test that xyz_id is used when both xyz_id and data_info_list are provided."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="linear")

        batch_size = 2
        num_channels = 6
        seq_len = 40

        x = torch.randn(batch_size, num_channels, seq_len)

        # Create both xyz_id and legacy data_info_list
        xyz_id = np.random.randint(-50, 50, size=(batch_size, num_channels, 3))
        legacy_data_info_list = [
            {"num_channels": 999, "modality": "EEG"}  # Different values
            for _ in range(batch_size)
        ]

        _ = decoder(x, xyz_id=xyz_id, data_info_list=legacy_data_info_list)

        # xyz_id should take precedence - new data_info_list should be built from xyz_id
        assert mock_model.last_data_info_list is not legacy_data_info_list
        assert len(mock_model.last_data_info_list) == batch_size

        for i, info in enumerate(mock_model.last_data_info_list):
            assert info["num_channels"] == num_channels  # From x.shape, not legacy
            assert info["modality"] == "iEEG"
            np.testing.assert_array_equal(info["xyz_id"], xyz_id[i])


class TestDIVERDecoderOutputActivation:
    """Test DIVERDecoder output activation handling."""

    def test_sigmoid_activation(self):
        """Test sigmoid output activation."""
        mock_model = MockDiverModel(output_dim=1)
        decoder = DIVERDecoder(mock_model, output_activation="sigmoid")

        x = torch.randn(4, 10, 100)
        output = decoder(x)

        # Sigmoid output should be in [0, 1]
        assert output.min() >= 0
        assert output.max() <= 1
        # Output should be squeezed for output_dim=1
        assert output.shape == (4,)

    def test_softmax_activation(self):
        """Test softmax output activation."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="softmax")

        x = torch.randn(4, 10, 100)
        output = decoder(x)

        # Softmax output should sum to 1 along last dim
        assert output.shape == (4, 5)
        torch.testing.assert_close(output.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)

    def test_linear_activation(self):
        """Test linear (no) output activation."""
        mock_model = MockDiverModel(output_dim=5)
        decoder = DIVERDecoder(mock_model, output_activation="linear")

        x = torch.randn(4, 10, 100)
        output = decoder(x)

        # Linear output can be any value
        assert output.shape == (4, 5)


class TestDIVERDecoderFeatureCache:
    """Test DIVERDecoder cacheable feature contract."""

    def test_forward_matches_encode_then_forward_from_features(self):
        mock_model = MockSplitDiverModel(feature_dim=7, output_dim=3)
        decoder = DIVERDecoder(mock_model, output_activation="linear", output_dim=3)
        decoder.eval()

        x = torch.randn(4, 10, 100)
        with torch.no_grad():
            direct = decoder(x)
            split = decoder.forward_from_features(decoder.encode_features(x))

        torch.testing.assert_close(direct, split)


class TestCreateDataInfoList:
    """Test the create_data_info_list helper function."""

    def test_basic_creation(self):
        """Test basic data_info_list creation."""
        batch_size = 4
        num_channels = 10

        data_info_list = create_data_info_list(batch_size, num_channels)

        assert len(data_info_list) == batch_size
        for info in data_info_list:
            assert info["num_channels"] == num_channels
            assert info["modality"] == "iEEG"

    def test_with_channel_names(self):
        """Test creation with channel names."""
        batch_size = 2
        num_channels = 3
        channel_names = ["ch1", "ch2", "ch3"]

        data_info_list = create_data_info_list(
            batch_size, num_channels, channel_names=channel_names
        )

        for info in data_info_list:
            assert info["channel_names"] == channel_names

    def test_with_xyz_id(self):
        """Test creation with xyz_id coordinates."""
        batch_size = 2
        num_channels = 5
        xyz_id = np.random.randint(-50, 50, size=(num_channels, 3))

        data_info_list = create_data_info_list(
            batch_size, num_channels, xyz_id=xyz_id
        )

        for info in data_info_list:
            np.testing.assert_array_equal(info["xyz_id"], xyz_id)
