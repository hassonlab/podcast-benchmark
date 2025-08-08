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

    def test_setup_metrics_and_loss_with_loss_name_override(self):
        """Test setup_metrics_and_loss correctly overrides losses with loss_name."""
        training_params = TrainingParams(
            loss_name="mse", 
            losses=["custom_loss"],  # This should be overridden
            loss_weights=[0.5],  # This should be overridden
            metrics=["cosine_sim"]
        )

        all_fns = decoding_utils.setup_metrics_and_loss(training_params)

        # Verify loss_name override worked
        assert training_params.losses == ["mse"]
        assert training_params.loss_weights == [1]
        
        # Verify all functions are in all_fns
        assert len(all_fns) == 2  # mse (from loss_name) + cosine_sim (from metrics)
        assert "mse" in all_fns
        assert "cosine_sim" in all_fns
        assert "custom_loss" not in all_fns  # Should not be there due to override

    def test_setup_metrics_and_loss_with_multiple_losses(self):
        """Test setup_metrics_and_loss with multiple losses (no loss_name override)."""
        training_params = TrainingParams(
            loss_name=None,  # No override
            losses=["mse", "custom_loss"],
            loss_weights=[0.7, 0.3],
            metrics=["cosine_sim"]
        )

        all_fns = decoding_utils.setup_metrics_and_loss(training_params)

        # Verify no override occurred
        assert training_params.losses == ["mse", "custom_loss"]
        assert training_params.loss_weights == [0.7, 0.3]
        
        # Verify all functions are in all_fns
        assert len(all_fns) == 3  # mse + custom_loss (from losses) + cosine_sim (from metrics)
        assert "mse" in all_fns
        assert "custom_loss" in all_fns
        assert "cosine_sim" in all_fns

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


class TestComputeLoss:
    """Test compute_loss function for multiple loss support."""

    def setup_method(self):
        """Set up mock loss functions that return fixed values."""

        @registry.register_metric("mock_loss_1")
        def mock_loss_1(pred, true):
            return torch.tensor(1.0)

        @registry.register_metric("mock_loss_2")
        def mock_loss_2(pred, true):
            return torch.tensor(2.0)

        @registry.register_metric("mock_loss_3")
        def mock_loss_3(pred, true):
            return torch.tensor(3.0)

    def test_compute_loss_single_loss(self):
        """Test compute_loss with a single loss function."""
        pred = torch.tensor([[1.0, 2.0]])
        true = torch.tensor([[1.1, 1.9]])
        
        training_params = TrainingParams(
            losses=["mock_loss_1"],
            loss_weights=[1.0]
        )
        
        all_fns = {"mock_loss_1": registry.metric_registry["mock_loss_1"]}
        
        loss = decoding_utils.compute_loss(pred, true, training_params, all_fns)
        
        # Should be 1.0 * 1.0 = 1.0
        expected_loss = torch.tensor(1.0)
        assert torch.allclose(loss, expected_loss)

    def test_compute_loss_multiple_losses_equal_weights(self):
        """Test compute_loss with multiple losses and equal weights."""
        pred = torch.tensor([[1.0, 2.0]])
        true = torch.tensor([[1.1, 1.9]])
        
        training_params = TrainingParams(
            losses=["mock_loss_1", "mock_loss_2"],
            loss_weights=[0.5, 0.5]
        )
        
        all_fns = {
            "mock_loss_1": registry.metric_registry["mock_loss_1"],
            "mock_loss_2": registry.metric_registry["mock_loss_2"]
        }
        
        loss = decoding_utils.compute_loss(pred, true, training_params, all_fns)
        
        # Should be 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        expected_loss = torch.tensor(1.5)
        assert torch.allclose(loss, expected_loss)

    def test_compute_loss_multiple_losses_different_weights(self):
        """Test compute_loss with multiple losses and different weights."""
        pred = torch.tensor([[1.0, 2.0]])
        true = torch.tensor([[1.1, 1.9]])
        
        training_params = TrainingParams(
            losses=["mock_loss_1", "mock_loss_2", "mock_loss_3"],
            loss_weights=[0.2, 0.3, 0.5]
        )
        
        all_fns = {
            "mock_loss_1": registry.metric_registry["mock_loss_1"],
            "mock_loss_2": registry.metric_registry["mock_loss_2"],
            "mock_loss_3": registry.metric_registry["mock_loss_3"]
        }
        
        loss = decoding_utils.compute_loss(pred, true, training_params, all_fns)
        
        # Should be 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0 = 0.2 + 0.6 + 1.5 = 2.3
        expected_loss = torch.tensor(2.3)
        assert torch.allclose(loss, expected_loss)

    def test_compute_loss_zero_weight(self):
        """Test compute_loss with a zero weight (loss should be ignored)."""
        pred = torch.tensor([[1.0, 2.0]])
        true = torch.tensor([[1.1, 1.9]])
        
        training_params = TrainingParams(
            losses=["mock_loss_1", "mock_loss_2"],
            loss_weights=[1.0, 0.0]  # mock_loss_2 has zero weight
        )
        
        all_fns = {
            "mock_loss_1": registry.metric_registry["mock_loss_1"],
            "mock_loss_2": registry.metric_registry["mock_loss_2"]
        }
        
        loss = decoding_utils.compute_loss(pred, true, training_params, all_fns)
        
        # Should be 1.0 * 1.0 + 0.0 * 2.0 = 1.0
        expected_loss = torch.tensor(1.0)
        assert torch.allclose(loss, expected_loss)
