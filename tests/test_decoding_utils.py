"""
Tests for decoding_utils.py.

Tests the compute_cosine_distances function for computing cosine distances
between predictions and word embeddings, with support for ensemble predictions.
"""

import pytest
import torch
import numpy as np
from scipy.spatial.distance import cosine
from decoding_utils import compute_cosine_distances, compute_class_scores


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


class TestComputeClassScores:
    """Test compute_class_scores function for converting cosine distances to class probabilities."""

    @pytest.fixture
    def sample_distances_and_labels(self):
        """Create sample cosine distances and word labels for testing."""
        # 3 samples, 6 word embeddings
        cosine_distances = torch.tensor([
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],  # sample 0
            [0.8, 0.2, 0.7, 0.3, 0.6, 0.4],  # sample 1
            [0.4, 0.6, 0.5, 0.5, 0.1, 0.9],  # sample 2
        ], dtype=torch.float32)

        # 6 word embeddings belong to 3 classes (2 embeddings per class)
        word_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        return cosine_distances, word_labels

    def test_basic_functionality(self, sample_distances_and_labels):
        """Test basic functionality of compute_class_scores."""
        cosine_distances, word_labels = sample_distances_and_labels

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Check output shapes
        assert probabilities.shape == (3, 3)  # 3 samples, 3 classes
        assert logits.shape == (3, 3)

        # Check that probabilities sum to 1 for each sample
        for i in range(3):
            assert abs(probabilities[i].sum().item() - 1.0) < 1e-6

        # Check that probabilities are non-negative
        assert (probabilities >= 0).all()

    def test_class_averaging_logic(self):
        """Test that distances are properly averaged within each class."""
        # Create simple test case where we can manually calculate expected values
        cosine_distances = torch.tensor([
            [0.1, 0.3, 0.2, 0.8],  # 1 sample, 4 word embeddings
        ], dtype=torch.float32)

        # Class 0: embeddings 0,1 (distances 0.1, 0.3) -> average = 0.2
        # Class 1: embeddings 2,3 (distances 0.2, 0.8) -> average = 0.5
        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Expected class distances: [0.2, 0.5]
        # Expected logits: [1-0.2, 1-0.5] = [0.8, 0.5]
        expected_logits = torch.tensor([[0.8, 0.5]], dtype=torch.float32)

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_single_embedding_per_class(self):
        """Test case where each class has only one embedding."""
        cosine_distances = torch.tensor([
            [0.2, 0.5, 0.8],  # 1 sample, 3 word embeddings
        ], dtype=torch.float32)

        # Each embedding belongs to a different class
        word_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Logits should be 1 - distance for each embedding
        expected_logits = torch.tensor([[0.8, 0.5, 0.2]], dtype=torch.float32)

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_different_class_sizes(self):
        """Test with classes having different numbers of word embeddings."""
        cosine_distances = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 1 sample, 6 word embeddings
        ], dtype=torch.float32)

        # Class 0: 1 embedding, Class 1: 2 embeddings, Class 2: 3 embeddings
        word_labels = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Expected class averages:
        # Class 0: 0.1 (1 embedding)
        # Class 1: (0.2 + 0.3) / 2 = 0.25 (2 embeddings)
        # Class 2: (0.4 + 0.5 + 0.6) / 3 = 0.5 (3 embeddings)
        expected_class_distances = torch.tensor([0.1, 0.25, 0.5])
        expected_logits = torch.tensor([[1 - 0.1, 1 - 0.25, 1 - 0.5]])

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_softmax_probabilities(self):
        """Test that softmax transformation is applied correctly."""
        # Simple case with known logits
        cosine_distances = torch.tensor([
            [0.0, 1.0, 0.5],  # distances
        ], dtype=torch.float32)

        word_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Expected logits: [1.0, 0.0, 0.5]
        expected_logits = torch.tensor([[1.0, 0.0, 0.5]])
        assert torch.allclose(logits, expected_logits, atol=1e-6)

        # Check that softmax was applied correctly
        expected_probabilities = torch.softmax(expected_logits, dim=1)
        assert torch.allclose(probabilities, expected_probabilities, atol=1e-6)

    def test_multiple_samples(self):
        """Test with multiple samples."""
        cosine_distances = torch.tensor([
            [0.1, 0.9],  # sample 0
            [0.8, 0.2],  # sample 1
        ], dtype=torch.float32)

        word_labels = torch.tensor([0, 1], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Check shapes
        assert probabilities.shape == (2, 2)  # 2 samples, 2 classes
        assert logits.shape == (2, 2)

        # Expected logits for each sample
        expected_logits = torch.tensor([
            [0.9, 0.1],  # sample 0: [1-0.1, 1-0.9]
            [0.2, 0.8],  # sample 1: [1-0.8, 1-0.2]
        ])

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_consistent_class_ordering(self):
        """Test that class ordering is consistent (classes are always sorted internally)."""
        cosine_distances = torch.tensor([
            [0.1, 0.2, 0.3],
        ], dtype=torch.float32)

        # Test with different label orderings but same mapping
        word_labels_1 = torch.tensor([0, 1, 2], dtype=torch.long)
        word_labels_2 = torch.tensor([0, 1, 2], dtype=torch.long)  # Same labels

        _, logits_1 = compute_class_scores(cosine_distances, word_labels_1)
        _, logits_2 = compute_class_scores(cosine_distances, word_labels_2)

        # Results should be identical
        assert torch.allclose(logits_1, logits_2, atol=1e-6)

        # Test that classes are in sorted order internally
        # Labels 2, 0, 1 should result in classes ordered as 0, 1, 2
        labels_unsorted = torch.tensor([2, 0, 1], dtype=torch.long)
        distances_for_unsorted = torch.tensor([[0.3, 0.1, 0.2]], dtype=torch.float32)  # corresponding to labels 2,0,1

        _, logits_unsorted = compute_class_scores(distances_for_unsorted, labels_unsorted)

        # Should have same shape as original
        assert logits_unsorted.shape == logits_1.shape

    def test_empty_class_handling(self):
        """Test handling of edge cases with class gaps."""
        cosine_distances = torch.tensor([
            [0.1, 0.2, 0.3],
        ], dtype=torch.float32)

        # Classes 0, 2, 5 (gaps in numbering)
        word_labels = torch.tensor([0, 2, 5], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Should have 3 classes (corresponding to the unique labels)
        assert probabilities.shape == (1, 3)
        assert logits.shape == (1, 3)

        # Probabilities should still sum to 1
        assert abs(probabilities[0].sum().item() - 1.0) < 1e-6

    def test_gradient_flow(self):
        """Test that gradients flow through the computation."""
        cosine_distances = torch.tensor([
            [0.1, 0.9, 0.2, 0.8],
        ], dtype=torch.float32, requires_grad=True)

        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        probabilities, logits = compute_class_scores(cosine_distances, word_labels)

        # Use a meaningful loss (not sum of probabilities which is always 1)
        # Take probability of first class as loss
        loss = probabilities[0, 0]
        loss.backward()

        # Check that gradients were computed
        assert cosine_distances.grad is not None
        assert cosine_distances.grad.shape == cosine_distances.shape
        assert not torch.allclose(cosine_distances.grad, torch.zeros_like(cosine_distances.grad))

    def test_device_consistency(self):
        """Test that function works with different devices."""
        cosine_distances = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
        ], dtype=torch.float32)

        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Test on CPU
        probabilities_cpu, logits_cpu = compute_class_scores(cosine_distances, word_labels)

        # Move to GPU if available
        if torch.cuda.is_available():
            cosine_distances_gpu = cosine_distances.cuda()
            word_labels_gpu = word_labels.cuda()

            probabilities_gpu, logits_gpu = compute_class_scores(cosine_distances_gpu, word_labels_gpu)

            # Results should be the same
            assert torch.allclose(probabilities_cpu, probabilities_gpu.cpu(), atol=1e-6)
            assert torch.allclose(logits_cpu, logits_gpu.cpu(), atol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small distances (high similarity)
        cosine_distances_small = torch.tensor([
            [1e-7, 1e-8, 1e-6],
        ], dtype=torch.float32)

        word_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        probabilities_small, logits_small = compute_class_scores(cosine_distances_small, word_labels)

        # Should not have NaN or Inf values
        assert torch.isfinite(probabilities_small).all()
        assert torch.isfinite(logits_small).all()

        # Test with distances close to 2 (very dissimilar)
        cosine_distances_large = torch.tensor([
            [1.999, 1.998, 1.997],
        ], dtype=torch.float32)

        probabilities_large, logits_large = compute_class_scores(cosine_distances_large, word_labels)

        # Should not have NaN or Inf values
        assert torch.isfinite(probabilities_large).all()
        assert torch.isfinite(logits_large).all()

    def test_integration_with_compute_cosine_distances(self):
        """Test integration with compute_cosine_distances function."""
        # Create test data
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],  # should be closest to word 0
            [0.0, 1.0, 0.0],  # should be closest to word 1
        ], dtype=torch.float32)

        word_embeddings = torch.tensor([
            [1.0, 0.0, 0.0],  # class 0
            [0.8, 0.2, 0.0],  # class 0 (similar to above)
            [0.0, 1.0, 0.0],  # class 1
            [0.0, 0.8, 0.2],  # class 1 (similar to above)
        ], dtype=torch.float32)

        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Compute distances then class scores
        distances = compute_cosine_distances(predictions, word_embeddings)
        probabilities, logits = compute_class_scores(distances, word_labels)

        # Check shapes
        assert probabilities.shape == (2, 2)  # 2 samples, 2 classes

        # First prediction should be more likely to be class 0
        assert probabilities[0, 0] > probabilities[0, 1]

        # Second prediction should be more likely to be class 1
        assert probabilities[1, 1] > probabilities[1, 0]