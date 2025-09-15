import numpy as np

from metrics import calculate_auc_roc, top_k_accuracy


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
        frequencies = [
            np.array([5, 5, 5, 5]),
            np.array([3, 3, 3, 3]),
        ]  # All classes frequent
        min_frequencies = [2, 3]  # Include all classes

        auc_score = calculate_auc_roc(
            predictions, groundtruth, frequencies, min_frequencies, average="weighted"
        )

        # Should return a valid AUC score
        assert isinstance(auc_score, (float, np.floating))
        assert 0.0 <= auc_score <= 1.0
        # With good predictions, should be high
        assert auc_score > 0.8

    def test_frequency_filtering_excludes_bad_predictions(self):
        """Test that filtering works by excluding badly predicted low-frequency classes."""
        predictions = np.array(
            [
                # Perfect predictions for classes 0,1 (will be included)
                [1.0, 0.0, 0.0, 0.0],  # sample 0: perfect for class 0
                [0.0, 1.0, 0.0, 0.0],  # sample 1: perfect for class 1
                # Terrible predictions for classes 2,3 (will be excluded)
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # sample 2: predicts class 0, actually class 2 (wrong!)
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],  # sample 3: predicts class 1, actually class 3 (wrong!)
            ]
        )

        groundtruth = np.array([0, 1, 2, 3])

        # Classes 0,1 have high frequency (included), classes 2,3 have low frequency (excluded)
        frequencies = [np.array([10, 10, 1, 5]), np.array([10, 10, 1, 1])]
        min_frequencies = [5, 6]  # Exclude classes 2,3

        auc_score = calculate_auc_roc(
            predictions, groundtruth, frequencies, min_frequencies, average="weighted"
        )

        # Since we only include the perfectly predicted samples (classes 0,1),
        # AUC should be very high despite terrible predictions for classes 2,3
        assert auc_score > 0.95


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
