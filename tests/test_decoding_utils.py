"""
Tests for decoding_utils.py.

Tests the compute_cosine_distances function for computing cosine distances
between predictions and word embeddings, with support for ensemble predictions.
"""

import pytest
import torch
import numpy as np
from scipy.spatial.distance import cosine
from decoding_utils import (
    compute_cosine_distances,
    compute_class_scores,
    build_vocabulary,
    compute_word_embedding_task_metrics,
)


class TestComputeCosineDistances:
    """Test compute_cosine_distances function for various input configurations."""

    @pytest.fixture
    def sample_word_embeddings(self):
        """Create sample word embeddings for testing."""
        # 4 words, each with 6-dimensional embeddings
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # word 0
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # word 1
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # word 2
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # word 3 (mix of word 0 and 1)
            ],
            dtype=torch.float32,
        )

    @pytest.fixture
    def sample_predictions_2d(self):
        """Create sample 2D predictions (single prediction per sample)."""
        # 3 samples, each with 6-dimensional prediction
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # should be closest to word 0
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # should be closest to word 1
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # should be closest to word 2
            ],
            dtype=torch.float32,
        )

    @pytest.fixture
    def sample_predictions_3d(self):
        """Create sample 3D predictions (ensemble predictions)."""
        # 2 samples, 3 ensemble predictions each, 6-dimensional
        return torch.tensor(
            [
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
            ],
            dtype=torch.float32,
        )

    def test_basic_2d_predictions(self, sample_predictions_2d, sample_word_embeddings):
        """Test basic functionality with 2D predictions (single predictions)."""
        distances = compute_cosine_distances(
            sample_predictions_2d, sample_word_embeddings
        )

        # Check output shape
        assert distances.shape == (3, 4)  # 3 samples, 4 words

        # Check that predictions are closest to expected words
        assert torch.argmin(distances[0]).item() == 0
        assert torch.argmin(distances[1]).item() == 1
        assert torch.argmin(distances[2]).item() == 2

    def test_basic_3d_predictions(self, sample_predictions_3d, sample_word_embeddings):
        """Test basic functionality with 3D predictions (ensemble predictions)."""
        distances = compute_cosine_distances(
            sample_predictions_3d, sample_word_embeddings
        )

        # Check output shape
        assert distances.shape == (2, 4)  # 2 samples, 4 words

        # Check that ensemble predictions are closest to expected words
        assert torch.argmin(distances[0]).item() == 0
        assert torch.argmin(distances[1]).item() == 1

    def test_perfect_matches_give_zero_distance(self, sample_word_embeddings):
        """Test that perfect matches result in zero cosine distance."""
        # Use the word embeddings themselves as predictions
        predictions = sample_word_embeddings.clone()

        distances = compute_cosine_distances(predictions, sample_word_embeddings)

        for i in range(len(sample_word_embeddings)):
            assert distances[i, i].item() < 1e-6

    def test_orthogonal_vectors_give_unit_distance(self):
        """Test that orthogonal vectors give cosine distance of 1."""
        # Create orthogonal vectors
        word_embeddings = torch.tensor(
            [
                [1.0, 0.0],  # word 0
                [0.0, 1.0],  # word 1 (orthogonal to word 0)
            ],
            dtype=torch.float32,
        )

        predictions = torch.tensor(
            [
                [1.0, 0.0],  # prediction identical to word 0
            ],
            dtype=torch.float32,
        )

        distances = compute_cosine_distances(predictions, word_embeddings)

        # Distance to word 0 should be 0 (identical)
        assert distances[0, 0].item() < 1e-6

        # Distance to word 1 should be 1 (orthogonal)
        assert abs(distances[0, 1].item() - 1.0) < 1e-6

    def test_ensemble_averaging(self):
        """Test that ensemble predictions are properly averaged."""
        # Create word embeddings
        word_embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # word 0
                [0.0, 1.0, 0.0],  # word 1
            ],
            dtype=torch.float32,
        )

        # Create ensemble with predictions pointing to different words
        ensemble_predictions = torch.tensor(
            [
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],  # prediction 1: distance 0 to word 0, distance 1 to word 1
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],  # prediction 2: distance 1 to word 0, distance 0 to word 1
                ]
            ],
            dtype=torch.float32,
        )

        distances = compute_cosine_distances(ensemble_predictions, word_embeddings)

        # The function averages the distances from each ensemble member:
        # Distance to word 0: (0 + 1) / 2 = 0.5
        # Distance to word 1: (1 + 0) / 2 = 0.5
        # So both distances should be 0.5
        expected_distance = 0.5

        assert abs(distances[0, 0].item() - expected_distance) < 1e-6
        assert abs(distances[0, 1].item() - expected_distance) < 1e-6

    def test_consistency_with_scipy_cosine(
        self, sample_predictions_2d, sample_word_embeddings
    ):
        """Test that results are consistent with scipy's cosine distance."""
        distances = compute_cosine_distances(
            sample_predictions_2d, sample_word_embeddings
        )

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
        predictions = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zero vector
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # normal vector
            ],
            dtype=torch.float32,
        )

        # Function should not crash and should return finite values
        distances = compute_cosine_distances(predictions, sample_word_embeddings)

        assert torch.isfinite(distances).all()
        assert distances.shape == (2, len(sample_word_embeddings))

    def test_output_range(self, sample_predictions_2d, sample_word_embeddings):
        """Test that output distances are in valid range [0, 2]."""
        distances = compute_cosine_distances(
            sample_predictions_2d, sample_word_embeddings
        )

        # Cosine distance should be between 0 and 2
        # (0 for identical vectors, 2 for opposite vectors)
        assert (distances >= 0).all()
        assert (distances <= 2).all()

    def test_deterministic_output(self, sample_predictions_2d, sample_word_embeddings):
        """Test that function produces deterministic output."""
        distances_1 = compute_cosine_distances(
            sample_predictions_2d, sample_word_embeddings
        )
        distances_2 = compute_cosine_distances(
            sample_predictions_2d, sample_word_embeddings
        )

        assert torch.allclose(distances_1, distances_2, atol=1e-7)

    def test_single_sample_single_ensemble(self, sample_word_embeddings):
        """Test edge case with single sample and single ensemble member."""
        # Single sample, single ensemble member (equivalent to 2D case)
        predictions = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],  # 1 sample, 1 ensemble, 6 dims
            dtype=torch.float32,
        )

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

        word_embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # target word
                [0.0, 1.0, 0.0],  # different word
            ],
            dtype=torch.float32,
        )

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
        cosine_distances = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],  # sample 0
                [0.8, 0.2, 0.7, 0.3, 0.6, 0.4],  # sample 1
                [0.4, 0.6, 0.5, 0.5, 0.1, 0.9],  # sample 2
            ],
            dtype=torch.float32,
        )

        # 6 word embeddings belong to 3 classes (2 embeddings per class)
        word_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        return cosine_distances, word_labels

    def test_basic_functionality(self, sample_distances_and_labels):
        """Test basic functionality of compute_class_scores."""
        cosine_distances, word_labels = sample_distances_and_labels

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

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
        cosine_distances = torch.tensor(
            [
                [0.1, 0.3, 0.2, 0.8],  # 1 sample, 4 word embeddings
            ],
            dtype=torch.float32,
        )

        # Class 0: embeddings 0,1 (distances 0.1, 0.3) -> average = 0.2
        # Class 1: embeddings 2,3 (distances 0.2, 0.8) -> average = 0.5
        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

        # Expected class distances: [0.2, 0.5]
        # Expected logits: [1-0.2, 1-0.5] = [0.8, 0.5]
        expected_logits = torch.tensor([[0.8, 0.5]], dtype=torch.float32)

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_single_embedding_per_class(self):
        """Test case where each class has only one embedding."""
        cosine_distances = torch.tensor(
            [
                [0.2, 0.5, 0.8],  # 1 sample, 3 word embeddings
            ],
            dtype=torch.float32,
        )

        # Each embedding belongs to a different class
        word_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

        # Logits should be 1 - distance for each embedding
        expected_logits = torch.tensor([[0.8, 0.5, 0.2]], dtype=torch.float32)

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_different_class_sizes(self):
        """Test with classes having different numbers of word embeddings."""
        cosine_distances = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 1 sample, 6 word embeddings
            ],
            dtype=torch.float32,
        )

        # Class 0: 1 embedding, Class 1: 2 embeddings, Class 2: 3 embeddings
        word_labels = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

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
        cosine_distances = torch.tensor(
            [
                [0.0, 1.0, 0.5],  # distances
            ],
            dtype=torch.float32,
        )

        word_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

        # Expected logits: [1.0, 0.0, 0.5]
        expected_logits = torch.tensor([[1.0, 0.0, 0.5]])
        assert torch.allclose(logits, expected_logits, atol=1e-6)

        # Check that softmax was applied correctly
        expected_probabilities = torch.softmax(expected_logits, dim=1)
        assert torch.allclose(probabilities, expected_probabilities, atol=1e-6)

    def test_multiple_samples(self):
        """Test with multiple samples."""
        cosine_distances = torch.tensor(
            [
                [0.1, 0.9],  # sample 0
                [0.8, 0.2],  # sample 1
            ],
            dtype=torch.float32,
        )

        word_labels = torch.tensor([0, 1], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

        # Check shapes
        assert probabilities.shape == (2, 2)  # 2 samples, 2 classes
        assert logits.shape == (2, 2)

        # Expected logits for each sample
        expected_logits = torch.tensor(
            [
                [0.9, 0.1],  # sample 0: [1-0.1, 1-0.9]
                [0.2, 0.8],  # sample 1: [1-0.8, 1-0.2]
            ]
        )

        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_consistent_class_ordering(self):
        """Test that class ordering is consistent (classes are always sorted internally)."""
        cosine_distances = torch.tensor(
            [
                [0.1, 0.2, 0.3],
            ],
            dtype=torch.float32,
        )

        # Test with different label orderings but same mapping
        word_labels_1 = torch.tensor([0, 1, 2], dtype=torch.long)
        word_labels_2 = torch.tensor([0, 1, 2], dtype=torch.long)  # Same labels

        _, logits_1, _ = compute_class_scores(cosine_distances, word_labels_1)
        _, logits_2, _ = compute_class_scores(cosine_distances, word_labels_2)

        # Results should be identical
        assert torch.allclose(logits_1, logits_2, atol=1e-6)

        # Test that classes are in sorted order internally
        # Labels 2, 0, 1 should result in classes ordered as 0, 1, 2
        labels_unsorted = torch.tensor([2, 0, 1], dtype=torch.long)
        distances_for_unsorted = torch.tensor(
            [[0.3, 0.1, 0.2]], dtype=torch.float32
        )  # corresponding to labels 2,0,1

        _, logits_unsorted, _ = compute_class_scores(
            distances_for_unsorted, labels_unsorted
        )

        # Should have same shape as original
        assert logits_unsorted.shape == logits_1.shape

    def test_empty_class_handling(self):
        """Test handling of edge cases with class gaps."""
        cosine_distances = torch.tensor(
            [
                [0.1, 0.2, 0.3],
            ],
            dtype=torch.float32,
        )

        # Classes 0, 2, 5 (gaps in numbering)
        word_labels = torch.tensor([0, 2, 5], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

        # Should have 3 classes (corresponding to the unique labels)
        assert probabilities.shape == (1, 3)
        assert logits.shape == (1, 3)

        # Probabilities should still sum to 1
        assert abs(probabilities[0].sum().item() - 1.0) < 1e-6

    def test_gradient_flow(self):
        """Test that gradients flow through the computation."""
        cosine_distances = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.8],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances, word_labels)

        # Use a meaningful loss (not sum of probabilities which is always 1)
        # Take probability of first class as loss
        loss = probabilities[0, 0]
        loss.backward()

        # Check that gradients were computed
        assert cosine_distances.grad is not None
        assert cosine_distances.grad.shape == cosine_distances.shape
        assert not torch.allclose(
            cosine_distances.grad, torch.zeros_like(cosine_distances.grad)
        )

    def test_device_consistency(self):
        """Test that function works with different devices."""
        cosine_distances = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float32,
        )

        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Test on CPU
        probabilities_cpu, logits_cpu, unique_classes_cpu = compute_class_scores(
            cosine_distances, word_labels
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            cosine_distances_gpu = cosine_distances.cuda()
            word_labels_gpu = word_labels.cuda()

            probabilities_gpu, logits_gpu, unique_classes_gpu = compute_class_scores(
                cosine_distances_gpu, word_labels_gpu
            )

            # Results should be the same
            assert torch.allclose(probabilities_cpu, probabilities_gpu.cpu(), atol=1e-6)
            assert torch.allclose(logits_cpu, logits_gpu.cpu(), atol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small distances (high similarity)
        cosine_distances_small = torch.tensor(
            [
                [1e-7, 1e-8, 1e-6],
            ],
            dtype=torch.float32,
        )

        word_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        probabilities_small, logits_small, unique_classes_small = compute_class_scores(
            cosine_distances_small, word_labels
        )

        # Should not have NaN or Inf values
        assert torch.isfinite(probabilities_small).all()
        assert torch.isfinite(logits_small).all()

        # Test with distances close to 2 (very dissimilar)
        cosine_distances_large = torch.tensor(
            [
                [1.999, 1.998, 1.997],
            ],
            dtype=torch.float32,
        )

        probabilities_large, logits_large, unique_classes_large = compute_class_scores(
            cosine_distances_large, word_labels
        )

        # Should not have NaN or Inf values
        assert torch.isfinite(probabilities_large).all()
        assert torch.isfinite(logits_large).all()

    def test_integration_with_compute_cosine_distances(self):
        """Test integration with compute_cosine_distances function."""
        # Create test data
        predictions = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # should be closest to word 0
                [0.0, 1.0, 0.0],  # should be closest to word 1
            ],
            dtype=torch.float32,
        )

        word_embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # class 0
                [0.8, 0.2, 0.0],  # class 0 (similar to above)
                [0.0, 1.0, 0.0],  # class 1
                [0.0, 0.8, 0.2],  # class 1 (similar to above)
            ],
            dtype=torch.float32,
        )

        word_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Compute distances then class scores
        distances = compute_cosine_distances(predictions, word_embeddings)
        probabilities, logits, unique_classes = compute_class_scores(distances, word_labels)

        # Check shapes
        assert probabilities.shape == (2, 2)  # 2 samples, 2 classes

        # First prediction should be more likely to be class 0
        assert probabilities[0, 0] > probabilities[0, 1]

        # Second prediction should be more likely to be class 1
        assert probabilities[1, 1] > probabilities[1, 0]

    def test_can_handle_no_provided_word_labels(self, sample_distances_and_labels):
        """Test basic functionality of compute_class_scores."""
        cosine_distances, _ = sample_distances_and_labels

        probabilities, logits, unique_classes = compute_class_scores(cosine_distances)

        # Check output shapes
        assert probabilities.shape == (3, 6)  # 3 samples, 3 classes
        assert logits.shape == (3, 6)

        # Check that probabilities sum to 1 for each sample
        for i in range(3):
            assert abs(probabilities[i].sum().item() - 1.0) < 1e-6

        # Check that probabilities are non-negative
        assert (probabilities >= 0).all()


class TestBuildVocabulary:
    """Test build_vocabulary function for creating word-to-ID mappings."""

    def test_basic_functionality(self):
        """Test basic functionality with unique words."""
        words = ["apple", "banana", "cherry"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Check word_to_id mapping
        assert word_to_id == {"apple": 0, "banana": 1, "cherry": 2}

        # Check id_to_word mapping
        assert id_to_word == {0: "apple", 1: "banana", 2: "cherry"}

        # Check position_to_id mapping
        assert position_to_id == [0, 1, 2]

    def test_with_repeated_words(self):
        """Test functionality with repeated words."""
        words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Check word_to_id mapping - should assign IDs in order of first appearance
        assert word_to_id == {"apple": 0, "banana": 1, "cherry": 2}

        # Check id_to_word mapping
        assert id_to_word == {0: "apple", 1: "banana", 2: "cherry"}

        # Check position_to_id mapping - should map each position to correct word ID
        expected_position_to_id = [
            0,
            1,
            0,
            2,
            1,
            0,
        ]  # apple, banana, apple, cherry, banana, apple
        assert position_to_id == expected_position_to_id

    def test_empty_list(self):
        """Test with empty word list."""
        words = []
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {}
        assert id_to_word == {}
        assert position_to_id == []

    def test_single_word(self):
        """Test with single word."""
        words = ["hello"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {"hello": 0}
        assert id_to_word == {0: "hello"}
        assert position_to_id == [0]

    def test_single_word_repeated(self):
        """Test with single word repeated multiple times."""
        words = ["hello", "hello", "hello"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {"hello": 0}
        assert id_to_word == {0: "hello"}
        assert position_to_id == [0, 0, 0]

    def test_order_preservation(self):
        """Test that word IDs are assigned in order of first appearance."""
        words = ["zebra", "apple", "banana", "zebra", "apple"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # IDs should be assigned based on first appearance order
        assert word_to_id == {"zebra": 0, "apple": 1, "banana": 2}
        assert id_to_word == {0: "zebra", 1: "apple", 2: "banana"}
        assert position_to_id == [0, 1, 2, 0, 1]  # zebra, apple, banana, zebra, apple

    def test_case_sensitivity(self):
        """Test that function treats different cases as different words."""
        words = ["Apple", "apple", "APPLE", "Apple"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Should treat different cases as different words
        assert word_to_id == {"Apple": 0, "apple": 1, "APPLE": 2}
        assert id_to_word == {0: "Apple", 1: "apple", 2: "APPLE"}
        assert position_to_id == [0, 1, 2, 0]  # Apple, apple, APPLE, Apple

    def test_with_special_characters(self):
        """Test with words containing special characters."""
        words = ["hello-world", "test_word", "word@symbol", "hello-world"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {"hello-world": 0, "test_word": 1, "word@symbol": 2}
        assert id_to_word == {0: "hello-world", 1: "test_word", 2: "word@symbol"}
        assert position_to_id == [0, 1, 2, 0]

    def test_with_numbers_as_strings(self):
        """Test with numeric strings."""
        words = ["1", "2", "10", "1", "3", "2"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {"1": 0, "2": 1, "10": 2, "3": 3}
        assert id_to_word == {0: "1", 1: "2", 2: "10", 3: "3"}
        assert position_to_id == [0, 1, 2, 0, 3, 1]

    def test_with_empty_strings(self):
        """Test with empty strings in the word list."""
        words = ["hello", "", "world", "", "hello"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Empty string should be treated as a valid word
        assert word_to_id == {"hello": 0, "": 1, "world": 2}
        assert id_to_word == {0: "hello", 1: "", 2: "world"}
        assert position_to_id == [0, 1, 2, 1, 0]

    def test_consistency_across_calls(self):
        """Test that multiple calls with same input produce consistent results."""
        words = ["apple", "banana", "apple", "cherry"]

        word_to_id1, id_to_word1, position_to_id1 = build_vocabulary(words)
        word_to_id2, id_to_word2, position_to_id2 = build_vocabulary(words)

        assert word_to_id1 == word_to_id2
        assert id_to_word1 == id_to_word2
        assert position_to_id1 == position_to_id2

    def test_large_vocabulary(self):
        """Test with a larger vocabulary to ensure scalability."""
        # Create a list with 1000 unique words plus repetitions
        unique_words = [f"word_{i}" for i in range(1000)]
        words = unique_words + unique_words[:100]  # Add some repetitions

        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Check that all unique words got IDs
        assert len(word_to_id) == 1000
        assert len(id_to_word) == 1000

        # Check that position_to_id has correct length
        assert len(position_to_id) == 1100  # 1000 unique + 100 repetitions

        # Check that all IDs are in valid range
        assert all(0 <= word_id < 1000 for word_id in word_to_id.values())
        assert all(0 <= word_id < 1000 for word_id in id_to_word.keys())
        assert all(0 <= pos_id < 1000 for pos_id in position_to_id)

    def test_return_types(self):
        """Test that function returns correct types."""
        words = ["apple", "banana", "apple"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Check types
        assert isinstance(word_to_id, dict)
        assert isinstance(id_to_word, dict)
        assert isinstance(position_to_id, list)

        # Check that all dict keys are strings and values are ints
        for word, word_id in word_to_id.items():
            assert isinstance(word, str)
            assert isinstance(word_id, int)

        # Check that all id_to_word keys are ints and values are strings
        for word_id, word in id_to_word.items():
            assert isinstance(word_id, int)
            assert isinstance(word, str)

        # Check that all list elements are ints
        for pos_id in position_to_id:
            assert isinstance(pos_id, int)

    def test_word_to_id_completeness(self):
        """Test that word_to_id contains all unique words from input."""
        words = ["cat", "dog", "cat", "bird", "dog", "fish", "cat"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        unique_words = set(words)
        assert set(word_to_id.keys()) == unique_words
        assert set(id_to_word.values()) == unique_words
        assert len(word_to_id) == len(unique_words)
        assert len(id_to_word) == len(unique_words)

    def test_position_to_id_correctness(self):
        """Test that position_to_id correctly maps each position to the right word ID."""
        words = ["red", "green", "blue", "red", "yellow", "blue", "green"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        # Verify each position maps to correct word
        for i, word in enumerate(words):
            expected_id = word_to_id[word]
            actual_id = position_to_id[i]
            assert (
                actual_id == expected_id
            ), f"Position {i}: expected {expected_id}, got {actual_id}"

            # Also verify reverse mapping consistency
            assert (
                id_to_word[actual_id] == word
            ), f"Reverse mapping failed for ID {actual_id}"

    def test_unicode_words(self):
        """Test with unicode/non-ASCII words."""
        words = ["café", "naïve", "résumé", "café", "piñata"]
        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {"café": 0, "naïve": 1, "résumé": 2, "piñata": 3}
        assert id_to_word == {0: "café", 1: "naïve", 2: "résumé", 3: "piñata"}
        assert position_to_id == [0, 1, 2, 0, 3]

    def test_very_long_words(self):
        """Test with very long word strings."""
        long_word = "a" * 1000
        short_word = "b"
        words = [long_word, short_word, long_word, short_word]

        word_to_id, id_to_word, position_to_id = build_vocabulary(words)

        assert word_to_id == {long_word: 0, short_word: 1}
        assert id_to_word == {0: long_word, 1: short_word}
        assert position_to_id == [0, 1, 0, 1]


class TestComputeWordEmbeddingTaskMetrics:
    """Test compute_word_embedding_task_metrics function for brain data decoding metrics."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns predictable outputs."""

        class MockModel:
            def __init__(self):
                self.eval_called = False

            def eval(self):
                self.eval_called = True
                return self

            def __call__(self, x):
                # Return a simple linear transformation for predictable results
                # Each sample gets transformed to embeddings close to specific words
                batch_size = x.shape[0]
                embedding_dim = 4

                # Create embeddings that will be close to word 0, 1, 0, 1... pattern
                predictions = torch.zeros(batch_size, embedding_dim)
                for i in range(batch_size):
                    if i % 2 == 0:
                        predictions[i] = torch.tensor(
                            [1.0, 0.0, 0.0, 0.0]
                        )  # Close to word 0
                    else:
                        predictions[i] = torch.tensor(
                            [0.0, 1.0, 0.0, 0.0]
                        )  # Close to word 1

                return predictions

        return MockModel()

    @pytest.fixture
    def sample_data(self):
        """Create sample brain data and word embeddings for testing."""
        # 4 test samples, 10 brain features
        X_test = torch.randn(4, 10)

        # 4 word embeddings, 4 dimensions each
        Y_test = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # word 0 - class 0
                [0.9, 0.1, 0.0, 0.0],  # word 0 variant - class 0
                [0.0, 1.0, 0.0, 0.0],  # word 1 - class 1
                [0.1, 0.9, 0.0, 0.0],  # word 1 variant - class 1
            ],
            dtype=torch.float32,
        )

        # Selected words for vocabulary (enough for both test and train indices)
        selected_words = [
            "dog",
            "cat",
            "dog",
            "cat",
            "dog",
            "cat",
            "bird",
            "fish",
            "dog",
            "cat",
        ]

        # Indices for test and train (disjoint to prevent data leakage)
        test_index = np.array([0, 1, 2, 3])  # test uses first 4 positions
        train_index = np.array([4, 5, 6, 7, 8, 9])  # train uses different positions

        # Top-k thresholds and min frequency for AUC
        top_k_thresholds = [1, 3, 5]
        min_train_freq_auc = 2

        return {
            "X_test": X_test,
            "Y_test": Y_test,
            "selected_words": selected_words,
            "test_index": test_index,
            "train_index": train_index,
            "top_k_thresholds": top_k_thresholds,
            "min_train_freq_auc": min_train_freq_auc,
            "min_test_freq_auc": 1,
        }

    def test_basic_functionality(self, mock_model, sample_data):
        """Test basic functionality with mock data."""
        device = torch.device("cpu")

        results = compute_word_embedding_task_metrics(
            sample_data["X_test"],
            sample_data["Y_test"],
            mock_model,
            device,
            sample_data["selected_words"],
            sample_data["test_index"],
            sample_data["train_index"],
            sample_data["top_k_thresholds"],
            sample_data["min_train_freq_auc"],
            sample_data["min_test_freq_auc"],
        )

        # Check that all expected metrics are returned
        expected_keys = [
            "test_occ_top_1",
            "test_occ_top_3",
            "test_occ_top_5",
            "test_word_avg_auc_roc",
            "test_word_train_weighted_auc_roc",
            "test_word_test_weighted_auc_roc",
            "test_word_top_1",
            "test_word_top_3",
            "test_word_top_5",
        ]

        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"

        # Check that metrics are reasonable values
        for key in expected_keys:
            if "auc" not in key:  # top-k accuracies should be between 0 and 1
                assert 0 <= results[key] <= 1, f"Invalid {key}: {results[key]}"

        # AUC should be between 0 and 1
        assert 0 <= results["test_word_avg_auc_roc"] <= 1
        assert 0 <= results["test_word_train_weighted_auc_roc"] <= 1
        assert 0 <= results["test_word_test_weighted_auc_roc"] <= 1

        # Model should have been put in eval mode
        assert mock_model.eval_called

    def test_perfect_predictions(self):
        """Test with perfect predictions where model predicts exact word embeddings."""
        device = torch.device("cpu")

        # Create a model that returns exact word embeddings
        class PerfectModel:
            def __init__(self):
                self.call_count = 0

            def eval(self):
                return self

            def __call__(self, x):
                # The get_predictions function calls this once per sample
                # Return the appropriate prediction for each call
                if self.call_count == 0:
                    result = torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
                    )  # word 0
                else:
                    result = torch.tensor(
                        [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
                    )  # word 1
                self.call_count += 1
                return result

        # Test data
        X_test = torch.randn(2, 5)
        Y_test = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # word 0
                [0.0, 1.0, 0.0, 0.0],  # word 1
            ],
            dtype=torch.float32,
        )

        selected_words = [
            "word0",
            "word1",
            "word0",
            "word1",
        ]  # enough for both test and train
        test_index = np.array([0, 1])  # test uses positions 0, 1
        train_index = np.array([2, 3])  # train uses positions 2, 3 (disjoint)
        top_k_thresholds = [1]
        min_train_freq_auc = 1

        results = compute_word_embedding_task_metrics(
            X_test,
            Y_test,
            PerfectModel(),
            device,
            selected_words,
            test_index,
            train_index,
            top_k_thresholds,
            min_train_freq_auc,
            1,  # min_test_freq_auc
        )

        # With perfect predictions, top-1 accuracy should be 1.0
        assert results["test_occ_top_1"] == 1.0
        assert results["test_word_top_1"] == 1.0

    def test_random_predictions(self):
        """Test with random predictions to ensure metrics are computed."""
        device = torch.device("cpu")

        class RandomModel:
            def eval(self):
                return self

            def __call__(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 4)

        # Test data
        X_test = torch.randn(6, 8)
        Y_test = torch.randn(6, 4)
        selected_words = [
            "a",
            "b",
            "c",
            "a",
            "b",
            "c",
            "a",
            "b",
            "c",
            "a",
        ]  # 10 words total
        test_index = np.array([0, 1, 2, 3, 4, 5])  # test uses first 6 positions
        train_index = np.array([6, 7, 8, 9])  # train uses last 4 positions (disjoint)
        top_k_thresholds = [1, 2]
        min_train_freq_auc = 1

        results = compute_word_embedding_task_metrics(
            X_test,
            Y_test,
            RandomModel(),
            device,
            selected_words,
            test_index,
            train_index,
            top_k_thresholds,
            min_train_freq_auc,
            1,  # min_test_freq_auc
        )

        # With random predictions, metrics should be computed but not perfect
        assert 0 <= results["test_occ_top_1"] <= 1
        assert 0 <= results["test_word_avg_auc_roc"] <= 1
        assert results["test_occ_top_2"] >= results["test_occ_top_1"]  # top-2 >= top-1

    def test_edge_case_single_sample(self):
        """Test with single sample edge case."""
        device = torch.device("cpu")

        class SimpleModel:
            def eval(self):
                return self

            def __call__(self, x):
                return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        X_test = torch.randn(1, 3)
        Y_test = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        selected_words = ["word", "word", "word"]  # 3 words total
        test_index = np.array([0])  # test uses position 0
        train_index = np.array([1, 2])  # train uses positions 1, 2 (disjoint)
        top_k_thresholds = [1]
        min_train_freq_auc = 1

        results = compute_word_embedding_task_metrics(
            X_test,
            Y_test,
            SimpleModel(),
            device,
            selected_words,
            test_index,
            train_index,
            top_k_thresholds,
            min_train_freq_auc,
            1,  # min_test_freq_auc
        )

        # Should work with single sample
        assert "test_occ_top_1" in results
        assert "test_word_avg_auc_roc" in results

    def test_different_top_k_values(self, mock_model, sample_data):
        """Test with different top-k threshold values."""
        device = torch.device("cpu")

        # Test with various k values
        top_k_thresholds = [1, 2, 3, 5, 10]
        sample_data["top_k_thresholds"] = top_k_thresholds

        results = compute_word_embedding_task_metrics(
            sample_data["X_test"],
            sample_data["Y_test"],
            mock_model,
            device,
            sample_data["selected_words"],
            sample_data["test_index"],
            sample_data["train_index"],
            top_k_thresholds,
            sample_data["min_train_freq_auc"],
            sample_data["min_test_freq_auc"],
        )

        # Check that all top-k metrics are present
        for k in top_k_thresholds:
            assert f"test_occ_top_{k}" in results
            assert f"test_word_top_{k}" in results

        # Top-k accuracy should be non-decreasing as k increases
        occ_accuracies = [results[f"test_occ_top_{k}"] for k in top_k_thresholds]
        word_accuracies = [results[f"test_word_top_{k}"] for k in top_k_thresholds]

        for i in range(1, len(occ_accuracies)):
            assert occ_accuracies[i] >= occ_accuracies[i - 1]
            assert word_accuracies[i] >= word_accuracies[i - 1]

    def test_different_vocabulary_sizes(self):
        """Test with different vocabulary sizes."""
        device = torch.device("cpu")

        class IdentityModel:
            def eval(self):
                return self

            def __call__(self, x):
                # Return first few dimensions as embedding
                return x[:, :3]

        # Test with larger vocabulary
        X_test = torch.randn(8, 5)
        Y_test = torch.randn(8, 3)
        selected_words = [
            "w1",
            "w2",
            "w3",
            "w4",
            "w1",
            "w2",
            "w3",
            "w4",
            "w1",
            "w2",
            "w3",
            "w4",
        ]  # 12 words total
        test_index = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # test uses first 8 positions
        train_index = np.array([8, 9, 10, 11])  # train uses last 4 positions (disjoint)
        top_k_thresholds = [1, 2]
        min_train_freq_auc = 1

        results = compute_word_embedding_task_metrics(
            X_test,
            Y_test,
            IdentityModel(),
            device,
            selected_words,
            test_index,
            train_index,
            top_k_thresholds,
            min_train_freq_auc,
            1,  # min_test_freq_auc
        )

        # Should handle larger vocabulary
        assert all(
            key in results
            for key in ["test_occ_top_1", "test_word_top_1", "test_word_avg_auc_roc"]
        )

    def test_integration_with_actual_functions(self):
        """Test integration with actual decoding_utils functions."""
        device = torch.device("cpu")

        # Use a simple linear model that can be called
        class SimpleLinearModel:
            def __init__(self):
                self.weight = torch.randn(3, 5)  # 5 input features -> 3 output dims

            def eval(self):
                return self

            def __call__(self, x):
                return torch.mm(x, self.weight.t())

        model = SimpleLinearModel()
        X_test = torch.randn(4, 5)
        Y_test = torch.randn(4, 3)
        selected_words = [
            "cat",
            "dog",
            "cat",
            "dog",
            "cat",
            "dog",
            "bird",
        ]  # 7 words total
        test_index = np.array([0, 1, 2, 3])  # test uses first 4 positions
        train_index = np.array([4, 5, 6])  # train uses last 3 positions (disjoint)
        top_k_thresholds = [1, 2]
        min_train_freq_auc = 1  # reduced since we have fewer train samples

        results = compute_word_embedding_task_metrics(
            X_test,
            Y_test,
            model,
            device,
            selected_words,
            test_index,
            train_index,
            top_k_thresholds,
            min_train_freq_auc,
            1,  # min_test_freq_auc
        )

        # Verify the function produces valid results
        assert isinstance(results, dict)
        assert len(results) == 9  # 2 occ + 2 word + 3 auc metrics + 2 perplexity metrics

        # All values should be numeric and finite
        for key, value in results.items():
            assert isinstance(value, (int, float, torch.Tensor))
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            else:
                assert not np.isnan(value) and not np.isinf(value)
