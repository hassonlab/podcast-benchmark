#!/usr/bin/env python3

"""
Test script to verify that the refactored plot_utils.py functions work correctly
with various metric dictionary configurations.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys

# Use non-interactive backend when running under pytest
if 'pytest' in sys.modules or '--pytest' in sys.argv:
    matplotlib.use('Agg')

# Import the refactored functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import (
    plot_training_history,
    plot_cv_results,
    plot_ridge_results,
    extract_metric_names,
    format_metric_name,
    get_subplot_layout,
)


def test_extract_metric_names():
    """Test the extract_metric_names function."""
    print("Testing extract_metric_names...")

    # Test with legacy format (should work with existing code)
    history_legacy = {
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.2, 0.9, 0.7],
        'train_cosine': [0.5, 0.6, 0.7],
        'val_cosine': [0.4, 0.5, 0.6],
        'num_epochs': 3
    }

    metrics = extract_metric_names(history_legacy)
    expected = ['cosine', 'loss']
    assert metrics == expected, f"Expected {expected}, got {metrics}"
    print("✓ Legacy format test passed")

    # Test with new registry-based format
    history_new = {
        'train_mse': [1.0, 0.8, 0.6],
        'val_mse': [1.2, 0.9, 0.7],
        'train_cosine_sim': [0.5, 0.6, 0.7],
        'val_cosine_sim': [0.4, 0.5, 0.6],
        'train_nll_embedding': [2.1, 1.8, 1.5],
        'val_nll_embedding': [2.3, 2.0, 1.7],
        'num_epochs': 3
    }

    metrics = extract_metric_names(history_new)
    expected = ['cosine_sim', 'mse', 'nll_embedding']
    assert metrics == expected, f"Expected {expected}, got {metrics}"
    print("✓ New format test passed")


def test_format_metric_name():
    """Test the format_metric_name function."""
    print("Testing format_metric_name...")

    test_cases = [
        ('mse', 'MSE'),
        ('cosine_sim', 'Cosine Similarity'),
        ('nll_embedding', 'NLL Embedding'),
        ('custom_metric', 'Custom Metric'),
        ('loss', 'Loss')
    ]

    for input_name, expected in test_cases:
        result = format_metric_name(input_name)
        assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_name}'"

    print("✓ All format tests passed")


def test_get_subplot_layout():
    """Test the get_subplot_layout function."""
    print("Testing get_subplot_layout...")

    test_cases = [
        (1, (1, 1)),
        (2, (1, 2)),
        (3, (2, 2)),
        (4, (2, 2)),
        (5, (2, 3)),
        (6, (2, 3)),
        (9, (3, 3))
    ]

    for n_metrics, expected in test_cases:
        result = get_subplot_layout(n_metrics)
        assert result == expected, f"Expected {expected} for {n_metrics} metrics, got {result}"

    print("✓ All layout tests passed")


def test_plot_training_history(show_plots=False):
    """Test plot_training_history with different metric configurations."""
    print("Testing plot_training_history...")

    # Test with legacy format
    history_legacy = {
        'train_loss': [1.0, 0.8, 0.6, 0.5],
        'val_loss': [1.2, 0.9, 0.7, 0.6],
        'train_cosine': [0.5, 0.6, 0.7, 0.75],
        'val_cosine': [0.4, 0.5, 0.6, 0.65],
        'num_epochs': 4
    }

    # This should not raise an error
    plot_training_history(history_legacy, fold=1)
    if show_plots:
        input("Press Enter to continue to next plot...")
    plt.close('all')  # Close the plot
    print("✓ Legacy format plot test passed")

    # Test with new registry format
    history_new = {
        'train_mse': [1.0, 0.8, 0.6, 0.5],
        'val_mse': [1.2, 0.9, 0.7, 0.6],
        'train_cosine_sim': [0.5, 0.6, 0.7, 0.75],
        'val_cosine_sim': [0.4, 0.5, 0.6, 0.65],
        'train_nll_embedding': [2.1, 1.8, 1.5, 1.2],
        'val_nll_embedding': [2.3, 2.0, 1.7, 1.4],
        'num_epochs': 4
    }

    # This should not raise an error
    plot_training_history(history_new, fold=2)
    if show_plots:
        input("Press Enter to continue to next plot...")
    plt.close('all')  # Close the plot
    print("✓ New format plot test passed")


def test_plot_cv_results(show_plots=False):
    """Test plot_cv_results with different metric configurations."""
    print("Testing plot_cv_results...")

    # Test with legacy format
    cv_results_legacy = {
        'train_loss': [1.0, 0.9, 0.8, 0.85, 0.75],
        'val_loss': [1.2, 1.1, 0.95, 1.0, 0.9],
        'test_loss': [1.1, 1.0, 0.9, 0.95, 0.85],
        'train_cosine': [0.5, 0.55, 0.6, 0.58, 0.65],
        'val_cosine': [0.45, 0.5, 0.55, 0.53, 0.6],
        'test_cosine': [0.48, 0.52, 0.58, 0.55, 0.62],
        'num_epochs': [10, 8, 12, 9, 11]
    }

    # This should not raise an error
    plot_cv_results(cv_results_legacy)
    if show_plots:
        input("Press Enter to continue to next plot...")
    plt.close('all')  # Close the plot
    print("✓ Legacy CV format plot test passed")

    # Test with new registry format
    cv_results_new = {
        'train_mse': [1.0, 0.9, 0.8, 0.85, 0.75],
        'val_mse': [1.2, 1.1, 0.95, 1.0, 0.9],
        'test_mse': [1.1, 1.0, 0.9, 0.95, 0.85],
        'train_cosine_sim': [0.5, 0.55, 0.6, 0.58, 0.65],
        'val_cosine_sim': [0.45, 0.5, 0.55, 0.53, 0.6],
        'test_cosine_sim': [0.48, 0.52, 0.58, 0.55, 0.62],
        'train_nll_embedding': [2.1, 1.9, 1.7, 1.8, 1.6],
        'val_nll_embedding': [2.3, 2.1, 1.9, 2.0, 1.8],
        'test_nll_embedding': [2.2, 2.0, 1.8, 1.9, 1.7],
        'num_epochs': [10, 8, 12, 9, 11]
    }

    # This should not raise an error
    plot_cv_results(cv_results_new)
    if show_plots:
        input("Press Enter to continue to next plot...")
    plt.close('all')  # Close the plot
    print("✓ New CV format plot test passed")


def test_plot_ridge_results(show_plots=False):
    """Test ridge plotting helper with synthetic results."""
    print("Testing plot_ridge_results...")

    results = {
        'lag_ms': [-100.0, 0.0, 100.0, 200.0],
        'r2': [0.15, 0.42, 0.35, 0.1],
        'train_r2': [0.6, 0.82, 0.75, 0.5],
        'alpha': [0.1, 0.01, 0.05, 0.2],
        'coef_norm': [2.0, 1.5, 1.7, 2.2],
        'n_samples': [500, 520, 510, 480],
        'n_features': [128, 128, 128, 128],
    }

    fig, axes = plot_ridge_results(results, show=show_plots)
    assert fig is not None
    assert len(axes) == 4
    if not show_plots:
        plt.close(fig)
    print("✓ Ridge plot test passed")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test plot_utils.py functions')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively (requires GUI environment)')
    args = parser.parse_args()

    # Set matplotlib backend based on whether plots should be shown
    if not args.show_plots:
        matplotlib.use('Agg')  # Use non-interactive backend for automated testing

    print("Running plot_utils refactoring tests...\n")
    if args.show_plots:
        print("🖼️  Interactive plot viewing enabled. Close plot windows or press Enter to continue.\n")

    test_extract_metric_names()
    test_format_metric_name()
    test_get_subplot_layout()
    test_plot_training_history(show_plots=args.show_plots)
    test_plot_cv_results(show_plots=args.show_plots)
    test_plot_ridge_results(show_plots=args.show_plots)

    print("\n✅ All tests passed! The refactored plot_utils.py functions work correctly.")


if __name__ == "__main__":
    main()