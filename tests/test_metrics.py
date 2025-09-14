import numpy as np

from metrics import calculate_auc_roc


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
        frequencies = np.array([5, 5, 5, 5])  # All classes frequent
        min_frequencies = 2  # Include all classes

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
        frequencies = np.array([10, 10, 1, 1])
        min_frequencies = 5  # Exclude classes 2,3

        auc_score = calculate_auc_roc(
            predictions, groundtruth, frequencies, min_frequencies, average="weighted"
        )

        # Since we only include the perfectly predicted samples (classes 0,1),
        # AUC should be very high despite terrible predictions for classes 2,3
        assert auc_score > 0.95
