"""
Tests for decoding setup functionality from decoding_utils.py.

Tests loss setup, metrics setup, early stopping configuration, and registry function resolution.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

import registry
from config import TrainingParams
from registry import metric_registry
import decoding_utils


class TestLossAndMetricSetup:
    """Test loss and metrics setup logic from training parameters."""

    def setup_method(self):
        """Set up test metrics."""

        @registry.register_metric("mse")
        def mse_metric(pred, true):
            return torch.nn.functional.mse_loss(pred, true)

        @registry.register_metric("cosine_sim")
        def cosine_sim_metric(pred, true):
            return torch.nn.functional.cosine_similarity(pred, true, dim=-1).mean()

        @registry.register_metric("custom_loss")
        def custom_loss_metric(pred, true):
            return torch.tensor(0.1)

    def test_setup_metrics_and_loss(self):
        """Test setup_metrics_and_loss function."""
        training_params = TrainingParams(
            loss_name="mse", metrics=["cosine_sim", "custom_loss"]
        )

        all_fns = decoding_utils.setup_metrics_and_loss(training_params)

        assert len(all_fns) == 3
        assert "mse" in all_fns  # Loss function
        assert "cosine_sim" in all_fns  # Metric
        assert "custom_loss" in all_fns  # Metric

        # Test that they're all callable
        for _, fn in all_fns.items():
            assert callable(fn)

    def test_duplicate_loss_and_metric(self):
        """Test when loss function is also listed in metrics."""
        training_params = TrainingParams(loss_name="mse", metrics=["mse", "cosine_sim"])

        all_fns = decoding_utils.setup_metrics_and_loss(training_params)

        assert len(all_fns) == 2
        assert "mse" in all_fns  # Loss function
        assert "cosine_sim" in all_fns  # Metric

        # Test that they're all callable
        for _, fn in all_fns.items():
            assert callable(fn)

    def test_empty_metrics_list(self):
        """Test with empty metrics list."""
        training_params = TrainingParams(loss_name="mse", metrics=[])

        all_fns = decoding_utils.setup_metrics_and_loss(training_params)

        assert len(all_fns) == 1
        assert "mse" in all_fns  # Loss function

        # Test that they're all callable
        for _, fn in all_fns.items():
            assert callable(fn)


class TestEarlyStoppingSetup:
    """Test early stopping configuration and validation."""

    def test_validate_early_stopping_config_valid(self):
        """Test that valid early stopping configurations pass validation."""
        # Valid: early stopping metric is the loss
        params1 = TrainingParams(
            loss_name="mse", metrics=["cosine_sim"], early_stopping_metric="mse"
        )

        # Should not raise exception
        decoding_utils.validate_early_stopping_config(params1)

        # Valid: early stopping metric is in metrics list
        params2 = TrainingParams(
            loss_name="mse", metrics=["cosine_sim"], early_stopping_metric="cosine_sim"
        )

        # Should not raise exception
        decoding_utils.validate_early_stopping_config(params2)

    def test_validate_early_stopping_config_invalid(self):
        """Test that invalid early stopping configurations raise ValueError."""
        params = TrainingParams(
            loss_name="mse",
            metrics=["cosine_sim"],
            early_stopping_metric="nonexistent_metric",
        )

        with pytest.raises(ValueError, match="Early stopping metric.*must be either"):
            decoding_utils.validate_early_stopping_config(params)

    def test_setup_early_stopping_state(self):
        """Test setup_early_stopping_state function."""
        # Test smaller_is_better=True (e.g., loss)
        params_minimize = TrainingParams(smaller_is_better=True)
        best_val, patience = decoding_utils.setup_early_stopping_state(params_minimize)

        assert best_val == float("inf")
        assert patience == 0

        # Test smaller_is_better=False (e.g., accuracy)
        params_maximize = TrainingParams(smaller_is_better=False)
        best_val, patience = decoding_utils.setup_early_stopping_state(params_maximize)

        assert best_val == -float("inf")
        assert patience == 0

    def test_should_update_best(self):
        """Test should_update_best function."""
        # Test smaller_is_better=True
        assert decoding_utils.should_update_best(0.5, 1.0, True) == True
        assert decoding_utils.should_update_best(1.5, 1.0, True) == False

        # Test smaller_is_better=False
        assert decoding_utils.should_update_best(0.9, 0.5, False) == True
        assert decoding_utils.should_update_best(0.3, 0.5, False) == False

    def test_early_stopping_metric_validation(self):
        """Test that early stopping metric must be in loss or metrics."""
        # Valid: early stopping metric is the loss
        params1 = TrainingParams(
            loss_name="mse", metrics=["cosine_sim"], early_stopping_metric="mse"
        )

        all_metrics = [params1.loss_name] + params1.metrics
        assert params1.early_stopping_metric in all_metrics

        # Valid: early stopping metric is in metrics list
        params2 = TrainingParams(
            loss_name="mse", metrics=["cosine_sim"], early_stopping_metric="cosine_sim"
        )

        all_metrics = [params2.loss_name] + params2.metrics
        assert params2.early_stopping_metric in all_metrics


class TestGradientAccumulationSetup:
    """Test gradient accumulation configuration."""

    def test_should_update_gradient_accumulation(self):
        """Test should_update_gradient_accumulation function."""
        grad_steps = 4
        total_batches = 8

        # Test various batch indices
        assert (
            decoding_utils.should_update_gradient_accumulation(
                3, total_batches, grad_steps
            )
            == True
        )  # 4th batch
        assert (
            decoding_utils.should_update_gradient_accumulation(
                7, total_batches, grad_steps
            )
            == True
        )  # Last batch
        assert (
            decoding_utils.should_update_gradient_accumulation(
                1, total_batches, grad_steps
            )
            == False
        )  # 2nd batch
        assert (
            decoding_utils.should_update_gradient_accumulation(
                5, total_batches, grad_steps
            )
            == False
        )  # 6th batch
