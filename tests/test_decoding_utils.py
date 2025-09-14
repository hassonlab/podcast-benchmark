"""
Tests for decoding_utils.py.

Tests the compute_cosine_distances function for computing cosine distances
between predictions and word embeddings, with support for ensemble predictions.
"""

import pytest
import torch
import numpy as np
from scipy.spatial.distance import cosine
from decoding_utils import compute_cosine_distances


class TestComputeCosineDistances:
    """Test compute_cosine_distances function for various input configurations."""

    @pytest.fixture
    def sample_word_embeddings(self):
        """Create sample word embeddings for testing."""
        # 4 words, each with 6-dimensional embeddings
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # word 0
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # word 1
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # word 2
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # word 3 (mix of word 0 and 1)
        ], dtype=torch.float32)

    @pytest.fixture
    def sample_predictions_2d(self):
        """Create sample 2D predictions (single prediction per sample)."""
        # 3 samples, each with 6-dimensional prediction
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # should be closest to word 0
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # should be closest to word 1
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # should be closest to word 2
        ], dtype=torch.float32)

    @pytest.fixture
    def sample_predictions_3d(self):
        """Create sample 3D predictions (ensemble predictions)."""
        # 2 samples, 3 ensemble predictions each, 6-dimensional
        return torch.tensor([
            # Sample 0: ensemble predictions all close to word 0
            [
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
            ],
            # Sample 1: ensemble predictions all close to word 1
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
            ],
        ], dtype=torch.float32)

    def test_basic_2d_predictions(self, sample_predictions_2d, sample_word_embeddings):
        """Test basic functionality with 2D predictions (single predictions)."""
        distances = compute_cosine_distances(sample_predictions_2d, sample_word_embeddings)

        # Check output shape
        assert distances.shape == (3, 4)  # 3 samples, 4 words

        # Check that predictions are closest to expected words
        # Sample 0 should be closest to word 0
        assert torch.argmin(distances[0]).item() == 0

        # Sample 1 should be closest to word 1
        assert torch.argmin(distances[1]).item() == 1

        # Sample 2 should be closest to word 2
        assert torch.argmin(distances[2]).item() == 2

    def test_basic_3d_predictions(self, sample_predictions_3d, sample_word_embeddings):
        """Test basic functionality with 3D predictions (ensemble predictions)."""
        distances = compute_cosine_distances(sample_predictions_3d, sample_word_embeddings)

        # Check output shape
        assert distances.shape == (2, 4)  # 2 samples, 4 words

        # Check that ensemble predictions are closest to expected words
        # Sample 0 ensemble should be closest to word 0
        assert torch.argmin(distances[0]).item() == 0

        # Sample 1 ensemble should be closest to word 1
        assert torch.argmin(distances[1]).item() == 1

    def test_perfect_matches_give_zero_distance(self, sample_word_embeddings):
        """Test that perfect matches result in zero cosine distance."""
        # Use the word embeddings themselves as predictions
        predictions = sample_word_embeddings.clone()

        distances = compute_cosine_distances(predictions, sample_word_embeddings)

        # Diagonal should be zeros (or very close due to floating point)
        for i in range(len(sample_word_embeddings)):
            assert distances[i, i].item() < 1e-6

    def test_orthogonal_vectors_give_unit_distance(self):
        """Test that orthogonal vectors give cosine distance of 1."""
        # Create orthogonal vectors
        word_embeddings = torch.tensor([
            [1.0, 0.0],  # word 0
            [0.0, 1.0],  # word 1 (orthogonal to word 0)
        ], dtype=torch.float32)

        predictions = torch.tensor([
            [1.0, 0.0],  # prediction identical to word 0
        ], dtype=torch.float32)

        distances = compute_cosine_distances(predictions, word_embeddings)

        # Distance to word 0 should be 0 (identical)
        assert distances[0, 0].item() < 1e-6

        # Distance to word 1 should be 1 (orthogonal)
        assert abs(distances[0, 1].item() - 1.0) < 1e-6

    def test_ensemble_averaging(self):
        """Test that ensemble predictions are properly averaged."""
        # Create word embeddings
        word_embeddings = torch.tensor([
            [1.0, 0.0, 0.0],  # word 0
            [0.0, 1.0, 0.0],  # word 1
        ], dtype=torch.float32)

        # Create ensemble with predictions pointing to different words
        ensemble_predictions = torch.tensor([
            [
                [1.0, 0.0, 0.0],  # prediction 1: distance 0 to word 0, distance 1 to word 1
                [0.0, 1.0, 0.0],  # prediction 2: distance 1 to word 0, distance 0 to word 1
            ]
        ], dtype=torch.float32)

        distances = compute_cosine_distances(ensemble_predictions, word_embeddings)

        # The function averages the distances from each ensemble member:
        # Distance to word 0: (0 + 1) / 2 = 0.5
        # Distance to word 1: (1 + 0) / 2 = 0.5
        # So both distances should be 0.5
        expected_distance = 0.5

        assert abs(distances[0, 0].item() - expected_distance) < 1e-6
        assert abs(distances[0, 1].item() - expected_distance) < 1e-6

    def test_consistency_with_scipy_cosine(self, sample_predictions_2d, sample_word_embeddings):
        """Test that results are consistent with scipy's cosine distance."""
        distances = compute_cosine_distances(sample_predictions_2d, sample_word_embeddings)

        # Compare with scipy calculations
        for i, pred in enumerate(sample_predictions_2d):
            for j, word_emb in enumerate(sample_word_embeddings):
                # Normalize vectors as the function does
                pred_norm = pred / torch.norm(pred)
                word_norm = word_emb / torch.norm(word_emb)

                # Calculate using scipy
                scipy_distance = cosine(pred_norm.numpy(), word_norm.numpy())

                # Compare with our function's result
                our_distance = distances[i, j].item()

                assert abs(our_distance - scipy_distance) < 1e-6

    def test_input_validation_wrong_dimensions(self, sample_word_embeddings):
        """Test that function raises error for wrong input dimensions."""
        # Test 1D input (invalid)
        with pytest.raises(ValueError, match="Predictions must be 2D or 3D tensor"):
            bad_predictions = torch.tensor([1.0, 0.0, 0.0])
            compute_cosine_distances(bad_predictions, sample_word_embeddings)

        # Test 4D input (invalid)
        with pytest.raises(ValueError, match="Predictions must be 2D or 3D tensor"):
            bad_predictions = torch.zeros(2, 2, 2, 6)
            compute_cosine_distances(bad_predictions, sample_word_embeddings)

    def test_different_batch_sizes(self, sample_word_embeddings):
        """Test function with different batch sizes."""
        embedding_dim = sample_word_embeddings.shape[1]

        # Test with batch size 1
        pred_1 = torch.randn(1, embedding_dim)
        distances_1 = compute_cosine_distances(pred_1, sample_word_embeddings)
        assert distances_1.shape == (1, len(sample_word_embeddings))

        # Test with batch size 10
        pred_10 = torch.randn(10, embedding_dim)
        distances_10 = compute_cosine_distances(pred_10, sample_word_embeddings)
        assert distances_10.shape == (10, len(sample_word_embeddings))

    def test_different_ensemble_sizes(self, sample_word_embeddings):
        """Test function with different ensemble sizes."""
        num_samples = 3
        embedding_dim = sample_word_embeddings.shape[1]

        # Test with ensemble size 2
        pred_ens_2 = torch.randn(num_samples, 2, embedding_dim)
        distances_2 = compute_cosine_distances(pred_ens_2, sample_word_embeddings)
        assert distances_2.shape == (num_samples, len(sample_word_embeddings))

        # Test with ensemble size 5
        pred_ens_5 = torch.randn(num_samples, 5, embedding_dim)
        distances_5 = compute_cosine_distances(pred_ens_5, sample_word_embeddings)
        assert distances_5.shape == (num_samples, len(sample_word_embeddings))

    def test_different_embedding_dimensions(self):
        """Test function with different embedding dimensions."""
        num_words = 3
        num_samples = 2

        # Test with 10-dimensional embeddings
        word_emb_10d = torch.randn(num_words, 10)
        pred_10d = torch.randn(num_samples, 10)
        distances = compute_cosine_distances(pred_10d, word_emb_10d)
        assert distances.shape == (num_samples, num_words)

        # Test with 300-dimensional embeddings (like typical word vectors)
        word_emb_300d = torch.randn(num_words, 300)
        pred_300d = torch.randn(num_samples, 300)
        distances = compute_cosine_distances(pred_300d, word_emb_300d)
        assert distances.shape == (num_samples, num_words)

    def test_gradient_flow(self, sample_predictions_2d, sample_word_embeddings):
        """Test that gradients flow through the computation properly."""
        # Make predictions require gradients
        predictions = sample_predictions_2d.clone().requires_grad_(True)

        distances = compute_cosine_distances(predictions, sample_word_embeddings)

        # Compute a loss and backpropagate
        loss = distances.sum()
        loss.backward()

        # Check that gradients were computed
        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape
        assert not torch.allclose(predictions.grad, torch.zeros_like(predictions.grad))

    def test_numerical_stability_with_zero_vectors(self, sample_word_embeddings):
        """Test numerical stability when dealing with zero vectors."""
        # Create predictions with a zero vector
        predictions = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zero vector
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # normal vector
        ], dtype=torch.float32)

        # Function should not crash and should return finite values
        distances = compute_cosine_distances(predictions, sample_word_embeddings)

        assert torch.isfinite(distances).all()
        assert distances.shape == (2, len(sample_word_embeddings))

    def test_output_range(self, sample_predictions_2d, sample_word_embeddings):
        """Test that output distances are in valid range [0, 2]."""
        distances = compute_cosine_distances(sample_predictions_2d, sample_word_embeddings)

        # Cosine distance should be between 0 and 2
        # (0 for identical vectors, 2 for opposite vectors)
        assert (distances >= 0).all()
        assert (distances <= 2).all()

    def test_deterministic_output(self, sample_predictions_2d, sample_word_embeddings):
        """Test that function produces deterministic output."""
        distances_1 = compute_cosine_distances(sample_predictions_2d, sample_word_embeddings)
        distances_2 = compute_cosine_distances(sample_predictions_2d, sample_word_embeddings)

        assert torch.allclose(distances_1, distances_2, atol=1e-7)

    def test_single_sample_single_ensemble(self, sample_word_embeddings):
        """Test edge case with single sample and single ensemble member."""
        # Single sample, single ensemble member (equivalent to 2D case)
        predictions = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # 1 sample, 1 ensemble, 6 dims
        ], dtype=torch.float32)

        distances = compute_cosine_distances(predictions, sample_word_embeddings)

        assert distances.shape == (1, len(sample_word_embeddings))
        assert torch.argmin(distances[0]).item() == 0  # Should be closest to word 0

    def test_large_ensemble_averaging(self):
        """Test that large ensembles are properly averaged."""
        # Create many ensemble members that all point to the same direction
        num_ensemble = 100
        target_direction = torch.tensor([1.0, 0.0, 0.0])

        # Add small noise to each ensemble member
        noise_scale = 0.01
        ensemble = target_direction.unsqueeze(0).repeat(num_ensemble, 1)
        ensemble += torch.randn_like(ensemble) * noise_scale

        predictions = ensemble.unsqueeze(0)  # Add batch dimension

        word_embeddings = torch.tensor([
            [1.0, 0.0, 0.0],  # target word
            [0.0, 1.0, 0.0],  # different word
        ], dtype=torch.float32)

        distances = compute_cosine_distances(predictions, word_embeddings)

        # Despite noise, ensemble average should be closest to target word
        assert torch.argmin(distances[0]).item() == 0

        # Distance to target should be small due to averaging effect
        assert distances[0, 0].item() < 0.1