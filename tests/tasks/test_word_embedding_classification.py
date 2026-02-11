"""
Tests for decoding_utils.py.

Tests the compute_cosine_distances function for computing cosine distances
between predictions and word embeddings, with support for ensemble predictions.
"""

import pytest
import torch
import numpy as np
from metrics.embedding_metrics import (
    build_vocabulary,
    compute_word_embedding_task_metrics,
)


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
            "test_occurence_top_1",
            "test_occurence_top_3",
            "test_occurence_top_5",
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
                pass

            def eval(self):
                return self

            def __call__(self, x):
                # The get_predictions function passes batched input
                # Return predictions for all samples in the batch
                batch_size = x.shape[0]
                if batch_size == 1:
                    # Single sample prediction
                    return torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
                    )  # word 0
                else:
                    # Multiple samples - return perfect predictions for each
                    return torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0],  # word 0 for first sample
                            [0.0, 1.0, 0.0, 0.0],  # word 1 for second sample
                        ],
                        dtype=torch.float32,
                    )

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
        assert results["test_occurence_top_1"] == 1.0
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
        assert 0 <= results["test_occurence_top_1"] <= 1
        assert 0 <= results["test_word_avg_auc_roc"] <= 1
        assert (
            results["test_occurence_top_2"] >= results["test_occurence_top_1"]
        )  # top-2 >= top-1

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
        assert "test_occurence_top_1" in results
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
            assert f"test_occurence_top_{k}" in results
            assert f"test_word_top_{k}" in results

        # Top-k accuracy should be non-decreasing as k increases
        occurence_accuracies = [
            results[f"test_occurence_top_{k}"] for k in top_k_thresholds
        ]
        word_accuracies = [results[f"test_word_top_{k}"] for k in top_k_thresholds]

        for i in range(1, len(occurence_accuracies)):
            assert occurence_accuracies[i] >= occurence_accuracies[i - 1]
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
            for key in [
                "test_occurence_top_1",
                "test_word_top_1",
                "test_word_avg_auc_roc",
            ]
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
        assert (
            len(results) == 9
        )  # 2 occ + 2 word + 3 auc metrics + 2 perplexity metrics

        # All values should be numeric and finite
        for key, value in results.items():
            assert isinstance(value, (int, float, torch.Tensor))
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            else:
                assert not np.isnan(value) and not np.isinf(value)
