import pytest

import numpy as np
import torch
from scipy.spatial.distance import cosine

from metrics.utils import (
    calculate_auc_roc,
    top_k_accuracy,
    compute_class_scores,
    compute_cosine_distances,
)
from metrics.classification_metrics import perplexity, cross_entropy_metric


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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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

        probabilities, logits, unique_classes = compute_class_scores(
            cosine_distances, word_labels
        )

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
        probabilities, logits, unique_classes = compute_class_scores(
            distances, word_labels
        )

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


class TestCalculateAucRoc:
    """Test calculate_auc_roc function for frequency-based filtering behavior."""

    def test_basic_functionality_all_included(self):
        """Test basic AUC ROC calculation with all classes included."""
        predictions = torch.tensor(
            [
                [0.8, 0.1, 0.05, 0.05],  # sample 0: high prob for class 0
                [0.1, 0.8, 0.05, 0.05],  # sample 1: high prob for class 1
                [0.05, 0.1, 0.8, 0.05],  # sample 2: high prob for class 2
                [0.05, 0.05, 0.1, 0.8],  # sample 3: high prob for class 3
            ],
            dtype=torch.float32,
        )

        groundtruth = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        train_frequencies = torch.tensor([5, 5, 5, 5], dtype=torch.long)
        test_frequencies = torch.tensor([3, 3, 3, 3], dtype=torch.long)
        min_train_freq = 2  # Include all classes
        min_test_freq = 3  # Include all classes

        avg_auc, train_weighted_auc, test_weighted_auc = calculate_auc_roc(
            predictions,
            groundtruth,
            train_frequencies,
            test_frequencies,
            min_train_freq,
            min_test_freq,
        )

        # Should return valid AUC scores
        assert isinstance(avg_auc, (float, np.floating))
        assert isinstance(train_weighted_auc, (float, np.floating))
        assert isinstance(test_weighted_auc, (float, np.floating))
        assert 0.0 <= avg_auc <= 1.0
        assert 0.0 <= train_weighted_auc <= 1.0
        assert 0.0 <= test_weighted_auc <= 1.0
        # With good predictions, should be high
        assert avg_auc > 0.8
        assert train_weighted_auc > 0.8
        assert test_weighted_auc > 0.8

    def test_frequency_filtering_excludes_bad_predictions(self):
        """Test that filtering works by excluding badly predicted low-frequency classes."""
        predictions = torch.tensor(
            [
                # Perfect predictions for classes 0,1 (will be included)
                [1.0, 0.0, 0.0, 0.0],  # sample 0: perfect for class 0
                [0.0, 1.0, 0.0, 0.0],  # sample 1: perfect for class 1
                # Terrible predictions for classes 2,3 (will be excluded)
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],  # sample 2: predicts class 0, actually class 2 (wrong!)
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],  # sample 3: predicts class 1, actually class 3 (wrong!)
            ],
            dtype=torch.float32,
        )

        groundtruth = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        # Classes 0,1 have high frequency (included), classes 2,3 have low frequency (excluded)
        train_frequencies = torch.tensor([10, 10, 1, 1], dtype=torch.long)
        test_frequencies = torch.tensor([10, 10, 1, 1], dtype=torch.long)
        min_train_freq = 5  # Exclude classes 2,3
        min_test_freq = 5  # Exclude classes 2,3

        avg_auc, train_weighted_auc, test_weighted_auc = calculate_auc_roc(
            predictions,
            groundtruth,
            train_frequencies,
            test_frequencies,
            min_train_freq,
            min_test_freq,
        )

        # Since we only include the perfectly predicted samples (classes 0,1),
        # AUC should be very high despite terrible predictions for classes 2,3
        assert avg_auc > 0.99
        assert train_weighted_auc > 0.99
        assert test_weighted_auc > 0.99

    def test_correctly_weights_frequencies(self):
        """Test that filtering works by excluding badly predicted low-frequency classes."""
        predictions = torch.tensor(
            [
                # Perfect predictions for classes 0,1 (will be included)
                [1.0, 0.0, 0.0, 0.0],  # sample 0: perfect for class 0
                [0.0, 1.0, 0.0, 0.0],  # sample 1: perfect for class 1
                # Terrible predictions for classes 2,3 (will be excluded)
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],  # sample 2: predicts class 0, actually class 2 (wrong!)
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],  # sample 3: predicts class 1, actually class 3 (wrong!)
            ],
            dtype=torch.float32,
        )

        groundtruth = torch.tensor([0, 3, 2, 3], dtype=torch.long)

        # Classes 0,4 have high frequency (included), classes 1,2 have low frequency (excluded)
        train_frequencies = torch.tensor([100, 0, 0, 1], dtype=torch.long)
        test_frequencies = torch.tensor([1, 0, 0, 100], dtype=torch.long)
        min_train_freq = 1  # Exclude classes 2,3
        min_test_freq = 1  # Exclude classes 2,3

        avg_auc, train_weighted_auc, test_weighted_auc = calculate_auc_roc(
            predictions,
            groundtruth,
            train_frequencies,
            test_frequencies,
            min_train_freq,
            min_test_freq,
        )

        # Average should be somewhere in the middle. train should be high (high freq of good class).
        # test should be low (high freq of bad class)
        assert avg_auc > 0.5 and avg_auc < 0.75
        assert train_weighted_auc > 0.8
        assert test_weighted_auc < 0.3


class TestTopKAccuracy:
    """Test top_k_accuracy function for various k values and prediction scenarios."""

    def test_top_1_accuracy_perfect_predictions(self):
        """Test top-1 accuracy with perfect predictions."""
        predictions = torch.tensor(
            [
                [0.9, 0.05, 0.03, 0.02],  # class 0 is top
                [0.1, 0.8, 0.07, 0.03],  # class 1 is top
                [0.2, 0.1, 0.6, 0.1],  # class 2 is top
                [0.05, 0.05, 0.1, 0.8],  # class 3 is top
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        accuracy = top_k_accuracy(predictions, ground_truth, k=1)
        assert accuracy == 1.0

    def test_top_1_accuracy_imperfect_predictions(self):
        """Test top-1 accuracy with some incorrect predictions."""
        predictions = torch.tensor(
            [
                [0.9, 0.05, 0.03, 0.02],  # class 0 is top, correct
                [0.1, 0.8, 0.07, 0.03],  # class 1 is top, correct
                [
                    0.6,
                    0.1,
                    0.2,
                    0.1,
                ],  # class 0 is top, but ground truth is 2, incorrect
                [0.05, 0.05, 0.1, 0.8],  # class 3 is top, correct
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        accuracy = top_k_accuracy(predictions, ground_truth, k=1)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_top_2_accuracy(self):
        """Test top-2 accuracy where ground truth is in top 2 predictions."""
        predictions = torch.tensor(
            [
                [0.9, 0.05, 0.03, 0.02],  # top 2: [0, 1], ground truth: 0, correct
                [0.1, 0.8, 0.07, 0.03],  # top 2: [1, 0], ground truth: 1, correct
                [0.6, 0.1, 0.25, 0.05],  # top 2: [0, 2], ground truth: 2, correct
                [0.2, 0.3, 0.1, 0.4],  # top 2: [3, 1], ground truth: 0, incorrect
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        accuracy = top_k_accuracy(predictions, ground_truth, k=2)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_top_3_accuracy(self):
        """Test top-3 accuracy."""
        predictions = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],  # top 3: [0, 1, 2], ground truth: 3, incorrect
                [0.1, 0.8, 0.07, 0.03],  # top 3: [1, 0, 2], ground truth: 1, correct
                [0.25, 0.1, 0.6, 0.05],  # top 3: [2, 0, 1], ground truth: 2, correct
                [0.2, 0.3, 0.1, 0.4],  # top 3: [3, 1, 0], ground truth: 0, correct
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([3, 1, 2, 0], dtype=torch.long)

        accuracy = top_k_accuracy(predictions, ground_truth, k=3)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_k_equals_num_classes(self):
        """Test when k equals the number of classes (should always be 1.0)."""
        predictions = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.8, 0.1, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2], dtype=torch.long)

        accuracy = top_k_accuracy(predictions, ground_truth, k=4)
        assert accuracy == 1.0  # All samples should be correct

    def test_single_sample(self):
        """Test with a single sample."""
        predictions = torch.tensor([[0.6, 0.2, 0.1, 0.1]], dtype=torch.float32)
        ground_truth = torch.tensor([1], dtype=torch.long)

        accuracy_top1 = top_k_accuracy(predictions, ground_truth, k=1)
        assert accuracy_top1 == 0.0  # Class 0 is top, but ground truth is 1

        accuracy_top2 = top_k_accuracy(predictions, ground_truth, k=2)
        assert accuracy_top2 == 1.0  # Class 1 is in top 2

    def test_tied_predictions(self):
        """Test behavior with tied prediction scores."""
        predictions = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],  # tie between classes 0 and 1
                [0.25, 0.25, 0.25, 0.25],  # all classes tied
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([1, 2], dtype=torch.long)

        # With ties, np.argsort is stable, so earlier indices come first
        # For first sample: top 2 will be [0, 1] (or [1, 0] depending on tie-breaking)
        # For second sample: all classes in top 4, so ground truth 2 will be in top k for k>=1
        accuracy_top1 = top_k_accuracy(predictions, ground_truth, k=1)
        accuracy_top2 = top_k_accuracy(predictions, ground_truth, k=2)

        # At least the second sample should be correct for k>=1
        assert accuracy_top2 >= 0.5


class TestPerplexity:
    """Test perplexity function for LLM evaluation."""

    def test_perfect_predictions(self):
        """Test perplexity with perfect predictions (should be 1.0)."""
        # Use very high logits for the correct class (effectively perfect predictions)
        predictions = torch.tensor(
            [
                [100.0, -100.0, -100.0, -100.0],  # Very high logit for class 0
                [-100.0, 100.0, -100.0, -100.0],  # Very high logit for class 1
                [-100.0, -100.0, 100.0, -100.0],  # Very high logit for class 2
                [-100.0, -100.0, -100.0, 100.0],  # Very high logit for class 3
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        # Perfect predictions should give perplexity close to 1.0
        assert abs(ppl - 1.0) < 1e-6

    def test_uniform_predictions(self):
        """Test perplexity with uniform predictions."""
        num_classes = 4
        # Uniform logits (all zeros) give uniform probabilities after softmax
        predictions = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # Uniform logits -> uniform probabilities
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        # Uniform predictions should give perplexity equal to num_classes
        expected_ppl = num_classes
        assert abs(ppl - expected_ppl) < 1e-5

    def test_single_sample(self):
        """Test perplexity with a single sample."""
        # Convert probabilities to logits: logit = log(prob)
        # For [0.6, 0.2, 0.1, 0.1] probabilities, we use log values
        import math
        predictions = torch.tensor([[math.log(0.6), math.log(0.2), math.log(0.1), math.log(0.1)]], dtype=torch.float32)
        ground_truth = torch.tensor([0], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        # With log probabilities as logits, after softmax normalization
        # the result won't be exactly 1/0.6, but should be in a reasonable range
        assert 1.0 < ppl < 3.0  # Reasonable perplexity range

    def test_varying_quality_predictions(self):
        """Test perplexity increases with worse predictions."""
        # Good predictions: high logits for correct classes
        good_predictions = torch.tensor(
            [
                [10.0, -5.0, -5.0, -5.0],  # High logit for class 0
                [-5.0, 10.0, -5.0, -5.0],  # High logit for class 1
                [-5.0, -5.0, 10.0, -5.0],  # High logit for class 2
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2], dtype=torch.long)

        # Bad predictions: uniform logits (no confidence)
        bad_predictions = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # Uniform logits
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        good_ppl = perplexity(good_predictions, ground_truth)
        bad_ppl = perplexity(bad_predictions, ground_truth)

        # Bad predictions should have higher perplexity
        assert bad_ppl > good_ppl
        assert good_ppl < 1.5  # Good predictions should be close to 1.0
        assert bad_ppl > 3.0  # Bad predictions should be higher (close to num_classes=4)

    def test_edge_cases(self):
        """Test edge cases like very extreme logits."""
        # Predictions with very low logit for true class and high for wrong class
        predictions = torch.tensor(
            [
                [-50.0, 50.0, 0.0, 0.0],  # Very low logit for class 0, very high for class 1
                [0.0, 50.0, 0.0, 0.0],  # Normal high logit for class 1
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        # Should handle very small probabilities without numerical issues
        assert not np.isnan(ppl)
        assert not np.isinf(ppl)
        assert ppl > 1.0

    def test_empty_input(self):
        """Test perplexity with empty input."""
        predictions = torch.tensor([]).reshape(0, 4)
        ground_truth = torch.tensor([], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        assert np.isinf(ppl)

    def test_binary_classification(self):
        """Test perplexity with binary classification."""
        predictions = torch.tensor(
            [
                [5.0, -5.0],  # High logit for class 0
                [-5.0, 5.0],  # High logit for class 1
                [1.0, -1.0],  # Moderate logit for class 0
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 0], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        # Should be reasonable value
        assert 1.0 <= ppl <= 10.0
        assert not np.isnan(ppl)

    def test_numerical_stability(self):
        """Test numerical stability with extreme logits."""
        predictions = torch.tensor(
            [
                [100.0, -100.0, -100.0, -100.0],  # Very high logit
                [-100.0, 100.0, -100.0, -100.0],  # Very high logit
                [0.0, 0.0, -100.0, -100.0],  # Moderate logits with very low others
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 0], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        assert not np.isnan(ppl)
        assert not np.isinf(ppl)
        assert ppl > 0

    def test_ignore_index(self):
        """Test that perplexity correctly ignores labels with value -100."""
        # Create predictions where some labels should be ignored
        predictions = torch.tensor(
            [
                [100.0, -100.0, -100.0, -100.0],  # Perfect prediction for class 0
                [-100.0, 100.0, -100.0, -100.0],  # Perfect prediction for class 1
                [-100.0, -100.0, -100.0, 100.0],  # Wrong prediction (predicts 3, should ignore)
                [0.0, 0.0, 0.0, 0.0],  # Uniform (bad) prediction for class 2
            ],
            dtype=torch.float32,
        )

        # Ground truth with -100 values that should be ignored
        ground_truth = torch.tensor([0, 1, -100, 2], dtype=torch.long)

        ppl_with_ignore = perplexity(predictions, ground_truth)

        # Calculate expected perplexity for only the non-ignored samples
        # Samples 0, 1, 3 should be included (indices 0, 1, 3)
        predictions_no_ignore = torch.tensor(
            [
                [100.0, -100.0, -100.0, -100.0],  # Perfect prediction for class 0
                [-100.0, 100.0, -100.0, -100.0],  # Perfect prediction for class 1
                [0.0, 0.0, 0.0, 0.0],  # Uniform prediction for class 2
            ],
            dtype=torch.float32,
        )
        ground_truth_no_ignore = torch.tensor([0, 1, 2], dtype=torch.long)
        ppl_expected = perplexity(predictions_no_ignore, ground_truth_no_ignore)

        # The perplexity should match when we exclude the ignored sample manually
        assert abs(ppl_with_ignore - ppl_expected) < 1e-5

    def test_ignore_index_all_ignored(self):
        """Test perplexity when all labels are -100 (all ignored)."""
        predictions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([-100, -100], dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)
        # When all samples are ignored, should return nan or inf
        assert np.isnan(ppl) or np.isinf(ppl)

    def test_ignore_index_mixed_cases(self):
        """Test perplexity with various combinations of valid and ignored labels."""
        predictions = torch.tensor(
            [
                [10.0, -5.0, -5.0, -5.0],  # Good prediction
                [0.0, 0.0, 0.0, 0.0],  # Bad prediction (ignored)
                [-5.0, 10.0, -5.0, -5.0],  # Good prediction
                [0.0, 0.0, 0.0, 0.0],  # Bad prediction (ignored)
                [-5.0, -5.0, 10.0, -5.0],  # Good prediction
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, -100, 1, -100, 2], dtype=torch.long)

        ppl_with_ignore = perplexity(predictions, ground_truth)

        # Only samples at indices 0, 2, 4 should be included
        predictions_valid = torch.tensor(
            [
                [10.0, -5.0, -5.0, -5.0],
                [-5.0, 10.0, -5.0, -5.0],
                [-5.0, -5.0, 10.0, -5.0],
            ],
            dtype=torch.float32,
        )
        ground_truth_valid = torch.tensor([0, 1, 2], dtype=torch.long)
        ppl_expected = perplexity(predictions_valid, ground_truth_valid)

        assert abs(ppl_with_ignore - ppl_expected) < 1e-5

    def test_padded_sequences(self):
        """Test perplexity with padded sequences of variable length (common for LLMs)."""
        batch_size = 3
        max_seq_len = 5
        vocab_size = 10

        # Create predictions: shape [batch_size * max_seq_len, vocab_size]
        # Sequence 1: length 5 (no padding)
        # Sequence 2: length 3 (2 padding tokens)
        # Sequence 3: length 2 (3 padding tokens)

        # Flatten the batch for easier construction
        predictions_list = []
        ground_truth_list = []

        # Sequence 1: length 5, tokens [2, 5, 1, 8, 3]
        for token_idx in [2, 5, 1, 8, 3]:
            logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
            logits[token_idx] = 10.0  # High logit for correct token
            predictions_list.append(logits)
            ground_truth_list.append(token_idx)

        # Sequence 2: length 3, tokens [4, 7, 0], then 2 padding tokens
        for token_idx in [4, 7, 0]:
            logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
            logits[token_idx] = 10.0  # High logit for correct token
            predictions_list.append(logits)
            ground_truth_list.append(token_idx)
        for _ in range(2):  # 2 padding tokens
            predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
            ground_truth_list.append(-100)  # Padding label

        # Sequence 3: length 2, tokens [6, 9], then 3 padding tokens
        for token_idx in [6, 9]:
            logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
            logits[token_idx] = 10.0  # High logit for correct token
            predictions_list.append(logits)
            ground_truth_list.append(token_idx)
        for _ in range(3):  # 3 padding tokens
            predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
            ground_truth_list.append(-100)  # Padding label

        predictions = torch.stack(predictions_list)
        ground_truth = torch.tensor(ground_truth_list, dtype=torch.long)

        ppl_with_padding = perplexity(predictions, ground_truth)

        # Calculate expected perplexity using only non-padded tokens
        non_padded_predictions = []
        non_padded_ground_truth = []

        for i, gt in enumerate(ground_truth_list):
            if gt != -100:
                non_padded_predictions.append(predictions_list[i])
                non_padded_ground_truth.append(gt)

        predictions_no_padding = torch.stack(non_padded_predictions)
        ground_truth_no_padding = torch.tensor(non_padded_ground_truth, dtype=torch.long)
        ppl_expected = perplexity(predictions_no_padding, ground_truth_no_padding)

        # Should match closely
        assert abs(ppl_with_padding - ppl_expected) < 1e-5

        # Perplexity should be low since predictions are good
        assert ppl_with_padding < 1.5

    def test_padded_sequences_with_poor_predictions(self):
        """Test perplexity with padded sequences where some non-padded tokens have poor predictions."""
        batch_size = 2
        max_seq_len = 4
        vocab_size = 8

        predictions_list = []
        ground_truth_list = []

        # Sequence 1: length 4, mixed quality predictions
        # Token 0: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[3] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(3)

        # Token 1: bad prediction (uniform)
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(2)

        # Token 2: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[5] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(5)

        # Token 3: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[1] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(1)

        # Sequence 2: length 2, then 2 padding tokens
        # Token 0: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[7] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(7)

        # Token 1: bad prediction (uniform)
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(4)

        # Token 2: padding
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(-100)

        # Token 3: padding
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(-100)

        predictions = torch.stack(predictions_list)
        ground_truth = torch.tensor(ground_truth_list, dtype=torch.long)

        ppl = perplexity(predictions, ground_truth)

        # Should be finite and positive
        assert not np.isnan(ppl)
        assert not np.isinf(ppl)
        assert ppl > 0

        # Should be higher than perfect predictions due to the uniform (bad) predictions
        # but not as high as all-uniform
        assert 1.0 < ppl < vocab_size

    def test_3d_sequence_equivalence(self):
        """Test that 3D sequence input produces same result as manually reshaped 2D input."""
        batch_size = 3
        seq_len = 4
        vocab_size = 10

        # Create test data
        predictions_3d = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        ground_truth_2d = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Compute perplexity with 3D input
        ppl_3d = perplexity(predictions_3d, ground_truth_2d)

        # Manually reshape and compute with 2D input
        predictions_2d = predictions_3d.reshape(batch_size * seq_len, vocab_size)
        ground_truth_1d = ground_truth_2d.reshape(batch_size * seq_len)
        ppl_2d = perplexity(predictions_2d, ground_truth_1d)

        # Results should be identical
        assert abs(ppl_3d - ppl_2d) < 1e-5


class TestCrossEntropy:
    """Test cross_entropy_metric function for LLM evaluation."""

    def test_ignore_index(self):
        """Test that cross_entropy correctly ignores labels with value -100."""
        # Create predictions where some labels should be ignored
        predictions = torch.tensor(
            [
                [100.0, -100.0, -100.0, -100.0],  # Perfect prediction for class 0
                [-100.0, 100.0, -100.0, -100.0],  # Perfect prediction for class 1
                [-100.0, -100.0, -100.0, 100.0],  # Wrong prediction (predicts 3, should ignore)
                [0.0, 0.0, 0.0, 0.0],  # Uniform (bad) prediction for class 2
            ],
            dtype=torch.float32,
        )

        # Ground truth with -100 values that should be ignored
        ground_truth = torch.tensor([0, 1, -100, 2], dtype=torch.long)

        ce_with_ignore = cross_entropy_metric(predictions, ground_truth)

        # Calculate expected cross entropy for only the non-ignored samples
        # Samples 0, 1, 3 should be included (indices 0, 1, 3)
        predictions_no_ignore = torch.tensor(
            [
                [100.0, -100.0, -100.0, -100.0],  # Perfect prediction for class 0
                [-100.0, 100.0, -100.0, -100.0],  # Perfect prediction for class 1
                [0.0, 0.0, 0.0, 0.0],  # Uniform prediction for class 2
            ],
            dtype=torch.float32,
        )
        ground_truth_no_ignore = torch.tensor([0, 1, 2], dtype=torch.long)
        ce_expected = cross_entropy_metric(predictions_no_ignore, ground_truth_no_ignore)

        # The cross entropy should match when we exclude the ignored sample manually
        assert abs(ce_with_ignore - ce_expected) < 1e-5

    def test_ignore_index_all_ignored(self):
        """Test cross_entropy when all labels are -100 (all ignored)."""
        predictions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([-100, -100], dtype=torch.long)

        ce = cross_entropy_metric(predictions, ground_truth)
        # When all samples are ignored, should return nan or inf
        assert np.isnan(ce) or np.isinf(ce)

    def test_ignore_index_mixed_cases(self):
        """Test cross_entropy with various combinations of valid and ignored labels."""
        predictions = torch.tensor(
            [
                [10.0, -5.0, -5.0, -5.0],  # Good prediction
                [0.0, 0.0, 0.0, 0.0],  # Bad prediction (ignored)
                [-5.0, 10.0, -5.0, -5.0],  # Good prediction
                [0.0, 0.0, 0.0, 0.0],  # Bad prediction (ignored)
                [-5.0, -5.0, 10.0, -5.0],  # Good prediction
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, -100, 1, -100, 2], dtype=torch.long)

        ce_with_ignore = cross_entropy_metric(predictions, ground_truth)

        # Only samples at indices 0, 2, 4 should be included
        predictions_valid = torch.tensor(
            [
                [10.0, -5.0, -5.0, -5.0],
                [-5.0, 10.0, -5.0, -5.0],
                [-5.0, -5.0, 10.0, -5.0],
            ],
            dtype=torch.float32,
        )
        ground_truth_valid = torch.tensor([0, 1, 2], dtype=torch.long)
        ce_expected = cross_entropy_metric(predictions_valid, ground_truth_valid)

        assert abs(ce_with_ignore - ce_expected) < 1e-5

    def test_perfect_predictions(self):
        """Test cross_entropy with perfect predictions."""
        predictions = torch.tensor(
            [
                [100.0, -100.0, -100.0],
                [-100.0, 100.0, -100.0],
                [-100.0, -100.0, 100.0],
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2], dtype=torch.long)

        ce = cross_entropy_metric(predictions, ground_truth)
        # Perfect predictions should give cross entropy close to 0
        assert ce < 1e-5

    def test_uniform_predictions(self):
        """Test cross_entropy with uniform predictions."""
        num_classes = 4
        predictions = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ground_truth = torch.tensor([0, 1, 2], dtype=torch.long)

        ce = cross_entropy_metric(predictions, ground_truth)
        # Uniform predictions should give cross entropy = log(num_classes)
        expected_ce = np.log(num_classes)
        assert abs(ce - expected_ce) < 1e-5

    def test_padded_sequences(self):
        """Test cross_entropy with padded sequences of variable length (common for LLMs)."""
        batch_size = 3
        max_seq_len = 5
        vocab_size = 10

        # Create predictions: shape [batch_size * max_seq_len, vocab_size]
        # Sequence 1: length 5 (no padding)
        # Sequence 2: length 3 (2 padding tokens)
        # Sequence 3: length 2 (3 padding tokens)

        # Flatten the batch for easier construction
        predictions_list = []
        ground_truth_list = []

        # Sequence 1: length 5, tokens [2, 5, 1, 8, 3]
        for token_idx in [2, 5, 1, 8, 3]:
            logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
            logits[token_idx] = 10.0  # High logit for correct token
            predictions_list.append(logits)
            ground_truth_list.append(token_idx)

        # Sequence 2: length 3, tokens [4, 7, 0], then 2 padding tokens
        for token_idx in [4, 7, 0]:
            logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
            logits[token_idx] = 10.0  # High logit for correct token
            predictions_list.append(logits)
            ground_truth_list.append(token_idx)
        for _ in range(2):  # 2 padding tokens
            predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
            ground_truth_list.append(-100)  # Padding label

        # Sequence 3: length 2, tokens [6, 9], then 3 padding tokens
        for token_idx in [6, 9]:
            logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
            logits[token_idx] = 10.0  # High logit for correct token
            predictions_list.append(logits)
            ground_truth_list.append(token_idx)
        for _ in range(3):  # 3 padding tokens
            predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
            ground_truth_list.append(-100)  # Padding label

        predictions = torch.stack(predictions_list)
        ground_truth = torch.tensor(ground_truth_list, dtype=torch.long)

        ce_with_padding = cross_entropy_metric(predictions, ground_truth)

        # Calculate expected cross entropy using only non-padded tokens
        non_padded_predictions = []
        non_padded_ground_truth = []

        for i, gt in enumerate(ground_truth_list):
            if gt != -100:
                non_padded_predictions.append(predictions_list[i])
                non_padded_ground_truth.append(gt)

        predictions_no_padding = torch.stack(non_padded_predictions)
        ground_truth_no_padding = torch.tensor(non_padded_ground_truth, dtype=torch.long)
        ce_expected = cross_entropy_metric(predictions_no_padding, ground_truth_no_padding)

        # Should match closely
        assert abs(ce_with_padding - ce_expected) < 1e-5

        # Cross entropy should be low since predictions are good
        assert ce_with_padding < 0.1

    def test_padded_sequences_with_poor_predictions(self):
        """Test cross_entropy with padded sequences where some non-padded tokens have poor predictions."""
        batch_size = 2
        max_seq_len = 4
        vocab_size = 8

        predictions_list = []
        ground_truth_list = []

        # Sequence 1: length 4, mixed quality predictions
        # Token 0: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[3] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(3)

        # Token 1: bad prediction (uniform)
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(2)

        # Token 2: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[5] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(5)

        # Token 3: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[1] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(1)

        # Sequence 2: length 2, then 2 padding tokens
        # Token 0: good prediction
        logits = torch.full((vocab_size,), -5.0, dtype=torch.float32)
        logits[7] = 10.0
        predictions_list.append(logits)
        ground_truth_list.append(7)

        # Token 1: bad prediction (uniform)
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(4)

        # Token 2: padding
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(-100)

        # Token 3: padding
        predictions_list.append(torch.zeros(vocab_size, dtype=torch.float32))
        ground_truth_list.append(-100)

        predictions = torch.stack(predictions_list)
        ground_truth = torch.tensor(ground_truth_list, dtype=torch.long)

        ce = cross_entropy_metric(predictions, ground_truth)

        # Should be finite and positive
        assert not np.isnan(ce)
        assert not np.isinf(ce)
        assert ce > 0

        # Should be higher than perfect predictions due to the uniform (bad) predictions
        # but not as high as all-uniform (log(vocab_size))
        assert 0.0 < ce < np.log(vocab_size)

    def test_sequence_prediction_shape(self):
        """Test cross_entropy with sequence prediction (batch, seq_len, vocab_size)."""
        batch_size = 16
        seq_len = 16
        vocab_size = 50259

        # Create predictions: shape [batch, seq_len, vocab_size]
        predictions = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Should not raise an error
        ce = cross_entropy_metric(predictions, ground_truth)

        # Should return a valid scalar
        assert not np.isnan(ce)
        assert not np.isinf(ce)
        assert ce > 0

    def test_sequence_prediction_with_padding(self):
        """Test cross_entropy with sequence prediction and -100 padding."""
        batch_size = 4
        seq_len = 8
        vocab_size = 100

        # Create predictions
        predictions = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Add some padding tokens (-100)
        ground_truth[0, 5:] = -100  # Last 3 tokens of sequence 0 are padding
        ground_truth[1, 6:] = -100  # Last 2 tokens of sequence 1 are padding
        ground_truth[3, 7:] = -100  # Last 1 token of sequence 3 is padding

        ce = cross_entropy_metric(predictions, ground_truth)

        # Should not raise an error and should be finite
        assert not np.isnan(ce)
        assert not np.isinf(ce)
        assert ce > 0

    def test_sequence_prediction_all_padding(self):
        """Test cross_entropy when entire sequence is padded."""
        batch_size = 2
        seq_len = 5
        vocab_size = 50

        predictions = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        ground_truth = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        ce = cross_entropy_metric(predictions, ground_truth)

        # When all tokens are ignored, should return nan or inf
        assert np.isnan(ce) or np.isinf(ce)

    def test_3d_sequence_equivalence(self):
        """Test that 3D sequence input produces same result as manually reshaped 2D input."""
        batch_size = 3
        seq_len = 4
        vocab_size = 10

        # Create test data
        predictions_3d = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        ground_truth_2d = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Compute cross entropy with 3D input
        ce_3d = cross_entropy_metric(predictions_3d, ground_truth_2d)

        # Manually reshape and compute with 2D input
        predictions_2d = predictions_3d.reshape(batch_size * seq_len, vocab_size)
        ground_truth_1d = ground_truth_2d.reshape(batch_size * seq_len)
        ce_2d = cross_entropy_metric(predictions_2d, ground_truth_1d)

        # Results should be identical
        assert abs(ce_3d - ce_2d) < 1e-5
