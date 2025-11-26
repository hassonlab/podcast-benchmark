"""
Tests for utils/dataset.py.

Tests the DictDataset class for handling dictionary inputs with PyTorch DataLoader.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from utils.dataset import NeuralDictDataset


@pytest.fixture
def basic_dataset():
    """Create a basic dataset with two features."""
    neural_data = torch.randn(10, 15)
    input_dict = {
        "feature1": torch.randn(10, 5),
        "feature2": torch.randn(10, 3),
    }
    target = torch.randn(10, 2)
    return NeuralDictDataset(neural_data, input_dict, target)


@pytest.fixture
def scalar_target_dataset():
    """Create a dataset with scalar targets."""
    neural_data = torch.randn(10, 15)
    input_dict = {
        "x": torch.randn(5, 10),
        "y": torch.randn(5, 20),
    }
    target = torch.randn(5)
    return NeuralDictDataset(neural_data, input_dict, target)


def test_dataset_length(basic_dataset):
    """Test that dataset returns correct length."""
    assert len(basic_dataset) == 10


def test_getitem_returns_tuple(basic_dataset):
    """Test that __getitem__ returns a tuple of (neural_data, dict, target)."""
    item = basic_dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 3
    assert isinstance(item[0], torch.Tensor)  # neural_data
    assert isinstance(item[1], dict)  # input_dict
    assert isinstance(item[2], torch.Tensor)  # target


def test_getitem_dict_keys(basic_dataset):
    """Test that returned dict contains all expected keys."""
    _, item_dict, _ = basic_dataset[0]
    assert "feature1" in item_dict
    assert "feature2" in item_dict
    assert len(item_dict) == 2


def test_getitem_tensor_shapes(basic_dataset):
    """Test that indexed tensors have correct shapes."""
    neural_data, item_dict, label = basic_dataset[0]
    assert neural_data.shape == (15,)
    assert item_dict["feature1"].shape == (5,)
    assert item_dict["feature2"].shape == (3,)
    assert label.shape == (2,)


def test_getitem_different_indices(basic_dataset):
    """Test that different indices return different values."""
    neural_data_1, item1_dict, label1 = basic_dataset[0]
    neural_data_2, item2_dict, label2 = basic_dataset[5]

    # Values should be different (with high probability for random data)
    assert not torch.allclose(neural_data_1, neural_data_2)
    assert not torch.allclose(item1_dict["feature1"], item2_dict["feature1"])
    assert not torch.allclose(label1, label2)


def test_scalar_target(scalar_target_dataset):
    """Test dataset with scalar targets."""
    _, item_dict, label = scalar_target_dataset[2]
    assert label.shape == ()  # Scalar tensor
    assert item_dict["x"].shape == (10,)
    assert item_dict["y"].shape == (20,)


def test_mismatched_input_lengths_raises_error():
    """Test that mismatched input tensor lengths raise ValueError."""
    neural_data = torch.randn(10, 15)
    bad_input = {
        "feature1": torch.randn(10, 5),
        "feature2": torch.randn(8, 3),  # Wrong length
    }
    target = torch.randn(10)

    with pytest.raises(ValueError, match="same length"):
        NeuralDictDataset(neural_data, bad_input, target)


def test_mismatched_target_length_raises_error():
    """Test that mismatched target length raises ValueError."""
    neural_data = torch.randn(10, 15)
    input_dict = {
        "feature1": torch.randn(10, 5),
    }
    target = torch.randn(8)  # Wrong length

    with pytest.raises(ValueError, match="same length"):
        NeuralDictDataset(neural_data, input_dict, target)


def test_dataloader_compatibility(basic_dataset):
    """Test that dataset works with PyTorch DataLoader."""
    dataloader = DataLoader(basic_dataset, batch_size=3, shuffle=True)

    batch_neural_data, batch_dict, batch_target = next(iter(dataloader))

    assert isinstance(batch_dict, dict)
    assert batch_neural_data.shape[0] <= 3  # Batch size
    assert batch_neural_data.shape == (batch_neural_data.shape[0], 15)
    assert batch_dict["feature1"].shape[0] <= 3  # Batch size
    assert batch_dict["feature1"].shape == (batch_dict["feature1"].shape[0], 5)
    assert batch_dict["feature2"].shape == (batch_dict["feature2"].shape[0], 3)
    assert batch_target.shape == (batch_dict["feature1"].shape[0], 2)


def test_dataloader_multiple_batches(basic_dataset):
    """Test that DataLoader iterates over all samples."""
    dataloader = DataLoader(basic_dataset, batch_size=3, shuffle=False)

    total_samples = 0
    for batch_neural_data, batch_dict, batch_target in dataloader:
        total_samples += batch_dict["feature1"].shape[0]
        # Verify each batch has correct structure
        assert batch_neural_data.shape[0] == batch_dict["feature1"].shape[0]
        assert isinstance(batch_dict, dict)
        assert "feature1" in batch_dict
        assert "feature2" in batch_dict

    assert total_samples == 10


def test_dataloader_last_batch(basic_dataset):
    """Test that DataLoader handles the last incomplete batch correctly."""
    dataloader = DataLoader(basic_dataset, batch_size=3, shuffle=False)

    batches = list(dataloader)
    assert len(batches) == 4  # 10 samples / 3 batch_size = 4 batches (3+3+3+1)

    # Last batch should have 1 sample
    last_batch_neural_data, last_batch_dict, last_batch_target = batches[-1]
    assert last_batch_neural_data.shape[0] == 1
    assert last_batch_dict["feature1"].shape[0] == 1
    assert last_batch_target.shape[0] == 1


def test_empty_input_dict():
    """Test behavior with empty input dictionary."""
    neural_data = torch.randn(7, 15)
    input_dict = {}
    target = torch.randn(5)

    dataset = NeuralDictDataset(neural_data, input_dict, target)
    assert len(dataset) == 5

    ret_neural_data, item_dict, label = dataset[0]
    assert ret_neural_data.shape == (15,)
    assert isinstance(item_dict, dict)
    assert len(item_dict) == 0
    assert label.shape == ()


def test_single_feature():
    """Test dataset with only one feature."""
    neural_data = torch.randn(7, 15)
    input_dict = {
        "single_feature": torch.randn(7, 4),
    }
    target = torch.randn(7, 1)

    dataset = NeuralDictDataset(neural_data, input_dict, target)
    ret_neural_data, item_dict, label = dataset[3]

    assert len(item_dict) == 1
    assert ret_neural_data.shape == (15,)
    assert item_dict["single_feature"].shape == (4,)
    assert label.shape == (1,)
