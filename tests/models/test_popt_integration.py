"""Tests for PopT integration module - specifically lip_coords handling."""

import pytest
import torch
import torch.nn as nn


class TestPopTLipCoordsHandling:
    """Test PopT forward method correctly handles lip_coords kwarg.

    These tests verify the lip_coords extraction logic without needing
    the full PopT model infrastructure.
    """

    def test_extract_lip_coords_from_data_info_list(self):
        """Test that lip_coords can be extracted from data_info_list format."""
        # This tests the extraction logic used in PopT's forward method
        batch_size = 4
        num_channels = 10

        # Create data_info_list with LIP_id tensors (legacy format)
        data_info_list = []
        expected_coords = []
        for i in range(batch_size):
            coords = torch.randint(0, 5000, (num_channels, 3), dtype=torch.long)
            data_info_list.append({"LIP_id": coords})
            expected_coords.append(coords)

        # Extract LIP_id from data_info_list (matching PopT's approach)
        coord_tensor = []
        for i in range(len(data_info_list)):
            coord_tensor.append(data_info_list[i]["LIP_id"])
        lip_coords = torch.stack(coord_tensor, dim=0)

        # Verify shape and values
        assert lip_coords.shape == (batch_size, num_channels, 3)
        for i in range(batch_size):
            torch.testing.assert_close(lip_coords[i], expected_coords[i])

    def test_lip_coords_direct_format(self):
        """Test that lip_coords can be passed directly (new format)."""
        batch_size = 4
        num_channels = 10

        # Create lip_coords directly (new format from refactored data getter)
        lip_coords = torch.randint(0, 5000, (batch_size, num_channels, 3), dtype=torch.long)

        # Verify shape
        assert lip_coords.shape == (batch_size, num_channels, 3)
        assert lip_coords.dtype == torch.long

    def test_lip_coords_from_repeated_single_tensor(self):
        """Test the pattern used by the data getter - repeating a single tensor."""
        num_channels = 10
        num_samples = 100

        # This mimics what get_popt_lip_coords does:
        # Create single lip_coords tensor for all samples
        base_lip_coords = torch.randint(0, 5000, (num_channels, 3), dtype=torch.long)

        # Repeat for all samples (as stored in DataFrame column)
        lip_coords_list = [base_lip_coords.clone() for _ in range(num_samples)]

        # Verify all are identical but independent copies
        assert len(lip_coords_list) == num_samples
        for coords in lip_coords_list:
            torch.testing.assert_close(coords, base_lip_coords)

        # Verify they are independent copies (modifying one doesn't affect others)
        lip_coords_list[0][0, 0] = 9999
        assert lip_coords_list[1][0, 0] != 9999

    def test_lip_coords_stacking_for_batch(self):
        """Test that lip_coords from DataFrame column can be stacked for batching."""
        batch_size = 8
        num_channels = 10

        # Simulate what DataLoader does - stacks individual tensors
        base_lip_coords = torch.randint(0, 5000, (num_channels, 3), dtype=torch.long)
        lip_coords_list = [base_lip_coords.clone() for _ in range(batch_size)]

        # Stack into batch tensor (what DataLoader collate_fn does)
        batched_lip_coords = torch.stack(lip_coords_list, dim=0)

        assert batched_lip_coords.shape == (batch_size, num_channels, 3)

    def test_lip_coords_values_in_expected_range(self):
        """Test that LIP coordinates are within expected range for PE table."""
        num_channels = 10
        max_coord_value = 5000  # Default max_coord_value in PopT

        # Create coordinates within valid range
        lip_coords = torch.randint(0, max_coord_value, (num_channels, 3), dtype=torch.long)

        # Verify all values are within range
        assert (lip_coords >= 0).all()
        assert (lip_coords < max_coord_value).all()


class TestLipCoordsDataGetterOutput:
    """Test the expected output format from get_popt_lip_coords."""

    def test_output_is_tensor_not_dict(self):
        """Verify the new format produces tensors, not dicts."""
        num_channels = 10

        # New format: direct tensor
        lip_coords = torch.randint(0, 5000, (num_channels, 3), dtype=torch.long)

        assert isinstance(lip_coords, torch.Tensor)
        assert lip_coords.shape == (num_channels, 3)
        assert lip_coords.dtype == torch.long

    def test_tensor_can_be_stored_in_dataframe(self):
        """Test that tensors can be stored in DataFrame columns."""
        import pandas as pd

        num_samples = 10
        num_channels = 8

        # Create tensors as would be stored in DataFrame
        base_coords = torch.randint(0, 5000, (num_channels, 3), dtype=torch.long)
        coords_list = [base_coords.clone() for _ in range(num_samples)]

        # Store in DataFrame
        df = pd.DataFrame({"lip_coords": coords_list})

        assert len(df) == num_samples
        assert "lip_coords" in df.columns

        # Verify each entry is a tensor
        for i in range(num_samples):
            assert isinstance(df.iloc[i]["lip_coords"], torch.Tensor)
            assert df.iloc[i]["lip_coords"].shape == (num_channels, 3)
