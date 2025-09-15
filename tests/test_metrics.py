import numpy as np

from metrics import calculate_auc_roc, top_k_accuracy, perplexity


class TestCalculateAucRoc:
    """Test calculate_auc_roc function for frequency-based filtering behavior."""

    def test_basic_functionality_all_included(self):
        """Test basic AUC ROC calculation with all classes included."""
        predictions = np.array(
            [
                [0.8, 0.1, 0.05, 0.05],  # sample 0: high prob for class 0
                [0.1, 0.8, 0.05, 0.05],  # sample 1: high prob for class 1
                [0.05, 0.1, 0.8, 0.05],  # sample 2: high prob for class 2
                [0.05, 0.05, 0.1, 0.8],  # sample 3: high prob for class 3
            ]
        )

        groundtruth = np.array([0, 1, 2, 3])
        train_frequencies = np.array([5, 5, 5, 5])
        test_frequencies = np.array([3, 3, 3, 3])
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
        predictions = np.array(
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
            ]
        )

        groundtruth = np.array([0, 1, 2, 3])

        # Classes 0,1 have high frequency (included), classes 2,3 have low frequency (excluded)
        train_frequencies = np.array([10, 10, 1, 1])
        test_frequencies = np.array([10, 10, 1, 1])
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
        predictions = np.array(
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
            ]
        )

        groundtruth = np.array([0, 3, 2, 3])

        # Classes 0,4 have high frequency (included), classes 1,2 have low frequency (excluded)
        train_frequencies = np.array([100, 0, 0, 1])
        test_frequencies = np.array([1, 0, 0, 100])
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
        predictions = np.array(
            [
                [0.9, 0.05, 0.03, 0.02],  # class 0 is top
                [0.1, 0.8, 0.07, 0.03],  # class 1 is top
                [0.2, 0.1, 0.6, 0.1],  # class 2 is top
                [0.05, 0.05, 0.1, 0.8],  # class 3 is top
            ]
        )
        ground_truth = np.array([0, 1, 2, 3])

        accuracy = top_k_accuracy(predictions, ground_truth, k=1)
        assert accuracy == 1.0

    def test_top_1_accuracy_imperfect_predictions(self):
        """Test top-1 accuracy with some incorrect predictions."""
        predictions = np.array(
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
            ]
        )
        ground_truth = np.array([0, 1, 2, 3])

        accuracy = top_k_accuracy(predictions, ground_truth, k=1)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_top_2_accuracy(self):
        """Test top-2 accuracy where ground truth is in top 2 predictions."""
        predictions = np.array(
            [
                [0.9, 0.05, 0.03, 0.02],  # top 2: [0, 1], ground truth: 0, correct
                [0.1, 0.8, 0.07, 0.03],  # top 2: [1, 0], ground truth: 1, correct
                [0.6, 0.1, 0.25, 0.05],  # top 2: [0, 2], ground truth: 2, correct
                [0.2, 0.3, 0.1, 0.4],  # top 2: [3, 1], ground truth: 0, incorrect
            ]
        )
        ground_truth = np.array([0, 1, 2, 0])

        accuracy = top_k_accuracy(predictions, ground_truth, k=2)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_top_3_accuracy(self):
        """Test top-3 accuracy."""
        predictions = np.array(
            [
                [0.4, 0.3, 0.2, 0.1],  # top 3: [0, 1, 2], ground truth: 3, incorrect
                [0.1, 0.8, 0.07, 0.03],  # top 3: [1, 0, 2], ground truth: 1, correct
                [0.25, 0.1, 0.6, 0.05],  # top 3: [2, 0, 1], ground truth: 2, correct
                [0.2, 0.3, 0.1, 0.4],  # top 3: [3, 1, 0], ground truth: 0, correct
            ]
        )
        ground_truth = np.array([3, 1, 2, 0])

        accuracy = top_k_accuracy(predictions, ground_truth, k=3)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_k_equals_num_classes(self):
        """Test when k equals the number of classes (should always be 1.0)."""
        predictions = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.8, 0.1, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]]
        )
        ground_truth = np.array([0, 1, 2])

        accuracy = top_k_accuracy(predictions, ground_truth, k=4)
        assert accuracy == 1.0  # All samples should be correct

    def test_single_sample(self):
        """Test with a single sample."""
        predictions = np.array([[0.6, 0.2, 0.1, 0.1]])
        ground_truth = np.array([1])

        accuracy_top1 = top_k_accuracy(predictions, ground_truth, k=1)
        assert accuracy_top1 == 0.0  # Class 0 is top, but ground truth is 1

        accuracy_top2 = top_k_accuracy(predictions, ground_truth, k=2)
        assert accuracy_top2 == 1.0  # Class 1 is in top 2

    def test_tied_predictions(self):
        """Test behavior with tied prediction scores."""
        predictions = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],  # tie between classes 0 and 1
                [0.25, 0.25, 0.25, 0.25],  # all classes tied
            ]
        )
        ground_truth = np.array([1, 2])

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
        predictions = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Perfect prediction for class 0
                [0.0, 1.0, 0.0, 0.0],  # Perfect prediction for class 1
                [0.0, 0.0, 1.0, 0.0],  # Perfect prediction for class 2
                [0.0, 0.0, 0.0, 1.0],  # Perfect prediction for class 3
            ]
        )
        ground_truth = np.array([0, 1, 2, 3])

        ppl = perplexity(predictions, ground_truth)
        # Perfect predictions should give perplexity of 1.0
        assert abs(ppl - 1.0) < 1e-10

    def test_uniform_predictions(self):
        """Test perplexity with uniform predictions."""
        num_classes = 4
        predictions = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],  # Uniform predictions
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
        ground_truth = np.array([0, 1, 2, 3])

        ppl = perplexity(predictions, ground_truth)
        # Uniform predictions should give perplexity equal to num_classes
        expected_ppl = num_classes
        assert abs(ppl - expected_ppl) < 1e-10

    def test_single_sample(self):
        """Test perplexity with a single sample."""
        predictions = np.array([[0.6, 0.2, 0.1, 0.1]])
        ground_truth = np.array([0])

        ppl = perplexity(predictions, ground_truth)
        # Perplexity = 2^(-log2(0.6)) = 1/0.6 ≈ 1.667
        expected_ppl = 1.0 / 0.6
        assert abs(ppl - expected_ppl) < 1e-10

    def test_varying_quality_predictions(self):
        """Test perplexity increases with worse predictions."""
        # Good predictions
        good_predictions = np.array(
            [
                [0.9, 0.05, 0.03, 0.02],
                [0.05, 0.9, 0.03, 0.02],
                [0.05, 0.03, 0.9, 0.02],
            ]
        )
        ground_truth = np.array([0, 1, 2])

        # Bad predictions
        bad_predictions = np.array(
            [
                [0.3, 0.3, 0.3, 0.1],
                [0.3, 0.3, 0.3, 0.1],
                [0.3, 0.3, 0.3, 0.1],
            ]
        )

        good_ppl = perplexity(good_predictions, ground_truth)
        bad_ppl = perplexity(bad_predictions, ground_truth)

        # Bad predictions should have higher perplexity
        assert bad_ppl > good_ppl
        assert good_ppl < 2.0  # Good predictions should be close to 1.0
        assert bad_ppl > 3.0  # Bad predictions should be higher

    def test_edge_cases(self):
        """Test edge cases like very small probabilities."""
        # Predictions with very small probability for true class
        predictions = np.array(
            [
                [1e-10, 0.9999999999, 0.0, 0.0],  # Very small prob for class 0
                [0.0, 1.0, 0.0, 0.0],  # Normal prediction
            ]
        )
        ground_truth = np.array([0, 1])

        ppl = perplexity(predictions, ground_truth)
        # Should handle very small probabilities without numerical issues
        assert not np.isnan(ppl)
        assert not np.isinf(ppl)
        assert ppl > 1.0

    def test_empty_input(self):
        """Test perplexity with empty input."""
        predictions = np.array([]).reshape(0, 4)
        ground_truth = np.array([])

        ppl = perplexity(predictions, ground_truth)
        assert np.isinf(ppl)

    def test_binary_classification(self):
        """Test perplexity with binary classification."""
        predictions = np.array(
            [
                [0.8, 0.2],  # Good prediction for class 0
                [0.3, 0.7],  # Good prediction for class 1
                [0.6, 0.4],  # Moderate prediction for class 0
            ]
        )
        ground_truth = np.array([0, 1, 0])

        ppl = perplexity(predictions, ground_truth)
        # Should be reasonable value
        assert 1.0 <= ppl <= 10.0
        assert not np.isnan(ppl)

    def test_numerical_stability(self):
        """Test numerical stability with probabilities close to 0 and 1."""
        predictions = np.array(
            [
                [1.0 - 1e-15, 1e-15, 0.0, 0.0],  # Very close to 1
                [1e-15, 1.0 - 1e-15, 0.0, 0.0],  # Very close to 1
                [0.5, 0.5, 0.0, 0.0],  # Normal case
            ]
        )
        ground_truth = np.array([0, 1, 0])

        ppl = perplexity(predictions, ground_truth)
        assert not np.isnan(ppl)
        assert not np.isinf(ppl)
        assert ppl > 0
