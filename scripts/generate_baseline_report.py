#!/usr/bin/env python3
"""
Generate baseline results report for all tasks.

This script processes baseline results from the baseline-results directory,
creates plots of performance metrics across lags, and generates a markdown
documentation page with the results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import re


# Create a custom YAML loader that ignores Python-specific tags
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    pass

def construct_undefined(self, node):
    if isinstance(node, yaml.MappingNode):
        return self.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        return self.construct_sequence(node)
    else:
        return None

SafeLoaderIgnoreUnknown.add_constructor(None, construct_undefined)


# Define metric configuration for each task type
TASK_METRICS = {
    'content_noncontent_task': {
        'primary': 'test_roc_auc_mean',
        'secondary': None,
        'higher_is_better': True,
        'ylabel': 'Test ROC AUC'
    },
    'ensemble_model': {
        'primary': 'test_word_avg_auc_roc_mean',
        'secondary': 'test_word_top_5_mean',
        'higher_is_better': True,
        'ylabel': 'Test Word Avg AUC ROC'
    },
    'gpt_surprise': {
        'primary': 'test_corr_mean',
        'secondary': None,
        'higher_is_better': True,
        'ylabel': 'Test Correlation'
    },
    'gpt_surprise_multiclass': {
        'primary': 'test_roc_auc_multiclass_mean',
        'secondary': None,
        'higher_is_better': True,
        'ylabel': 'Test ROC AUC (Multiclass)'
    },
    'pos_task': {
        'primary': 'test_roc_auc_multiclass_mean',
        'secondary': None,
        'higher_is_better': True,
        'ylabel': 'Test ROC AUC (Multiclass)'
    },
    'sentence_onset': {
        'primary': 'test_roc_auc_mean',
        'secondary': None,
        'higher_is_better': True,
        'ylabel': 'Test ROC AUC'
    },
    'volume_level': {
        'primary': 'test_corr_mean',
        'secondary': None,
        'higher_is_better': True,
        'ylabel': 'Test Correlation'
    }
}


def identify_task_type(task_name, config):
    """Identify the task type from the task name and config."""
    # Check config task_name first for more specific identification
    config_task_name = config.get('task_name', '')

    if 'gpt_surprise_multiclass' in config_task_name:
        return 'gpt_surprise_multiclass'
    elif 'content_noncontent' in task_name or 'content_noncontent' in config_task_name:
        return 'content_noncontent_task'
    elif 'ensemble_model' in task_name:
        return 'ensemble_model'
    elif 'gpt_surprise' in task_name or 'gpt_surprise' in config_task_name:
        return 'gpt_surprise'
    elif 'pos_task' in task_name or 'pos' in config_task_name:
        return 'pos_task'
    elif 'sentence_onset' in task_name:
        return 'sentence_onset'
    elif 'volume_level' in task_name:
        return 'volume_level'
    else:
        return None


def get_config_path(result_dir, task_name):
    """Find the corresponding config file for a task."""
    # Try to read the config from the result directory first
    config_in_results = result_dir / 'config.yml'
    trial_name = ''
    task_name_from_config = ''

    if config_in_results.exists():
        with open(config_in_results, 'r') as f:
            config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
            trial_name = config.get('trial_name', '')
            task_name_from_config = config.get('task_name', '')

    if not trial_name and not task_name_from_config:
        return None

    # Search for matching config in configs/
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    configs_dir = project_root / 'configs'

    # Search in all subdirectories (both .yml and .yaml)
    for ext in ['*.yml', '*.yaml']:
        for config_file in configs_dir.rglob(ext):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
                    if config and (config.get('trial_name') == trial_name or config.get('task_name') == task_name_from_config):
                        return config_file
            except Exception as e:
                continue

    return None


def process_baseline_result(result_dir):
    """Process a single baseline result directory."""
    lag_perf_file = result_dir / 'lag_performance.csv'
    config_file = result_dir / 'config.yml'

    if not lag_perf_file.exists() or not config_file.exists():
        return None

    # Read lag performance data
    df = pd.read_csv(lag_perf_file)

    # Read config
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    task_name = result_dir.name.rsplit('_2025', 1)[0]  # Remove timestamp
    trial_name = config.get('trial_name', task_name)
    task_type = identify_task_type(task_name, config)

    if task_type is None:
        print(f"Warning: Could not identify task type for {task_name}")
        return None

    # Get the config path
    config_path = get_config_path(result_dir, trial_name)

    return {
        'task_name': task_name,
        'trial_name': trial_name,
        'task_type': task_type,
        'df': df,
        'config': config,
        'config_path': config_path,
        'result_dir': result_dir
    }


def create_plot(result_data, output_dir):
    """Create a plot for a baseline result."""
    task_name = result_data['task_name']
    task_type = result_data['task_type']
    df = result_data['df']
    result_dir_name = result_data['result_dir'].name  # Use full directory name for uniqueness

    metric_config = TASK_METRICS[task_type]
    primary_metric = metric_config['primary']
    secondary_metric = metric_config['secondary']
    higher_is_better = metric_config['higher_is_better']
    ylabel = metric_config['ylabel']

    # Check if primary metric exists
    if primary_metric not in df.columns:
        print(f"Warning: Metric {primary_metric} not found in {task_name}")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot primary metric
    ax.plot(df['lags'], df[primary_metric], marker='o', linewidth=2, markersize=6, label=ylabel)

    # Plot secondary metric if it exists
    if secondary_metric and secondary_metric in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df['lags'], df[secondary_metric], marker='s', linewidth=2, markersize=6,
                color='orange', label='Test Word Top 5')
        ax2.set_ylabel('Test Word Top 5', fontsize=12)
        ax2.legend(loc='upper right')

    # Find best lag
    if higher_is_better:
        best_idx = df[primary_metric].idxmax()
    else:
        best_idx = df[primary_metric].idxmin()

    best_lag = df.loc[best_idx, 'lags']
    best_value = df.loc[best_idx, primary_metric]

    # Mark the best point
    ax.axvline(best_lag, color='red', linestyle='--', alpha=0.5, label=f'Best lag: {best_lag}ms')

    ax.set_xlabel('Lag (ms)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{task_name}\nBest {ylabel}: {best_value:.4f} at {best_lag}ms lag', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()

    # Save plot - use full result directory name to ensure uniqueness
    # Include the timestamp to differentiate between multiple runs of the same task
    plot_filename = result_dir_name.replace(':', '-') + '_lag_performance.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'plot_path': plot_path,
        'best_lag': int(best_lag),
        'best_value': best_value,
        'primary_metric': primary_metric,
        'secondary_metric': secondary_metric,
        'secondary_value': df.loc[best_idx, secondary_metric] if secondary_metric and secondary_metric in df.columns else None
    }


def generate_markdown_report(results, output_file):
    """Generate a markdown report with all baseline results."""

    with open(output_file, 'w') as f:
        f.write("# Baseline Results Summary\n\n")
        f.write("This page summarizes the baseline results for all tasks in the podcast benchmark.\n\n")
        f.write("## Overview\n\n")
        f.write(f"Total number of baseline results: {len(results)}\n\n")

        # Group by task type
        task_types = {}
        for result in results:
            task_type = result['task_type']
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(result)

        # Write results by task type
        for task_type, task_results in sorted(task_types.items()):
            f.write(f"## {task_type.replace('_', ' ').title()}\n\n")

            for result in sorted(task_results, key=lambda x: x['task_name']):
                f.write(f"### {result['task_name']}\n\n")

                # Add config reference
                if result['config_path']:
                    f.write(f"**Config:** [`{result['config_path']}`]({result['config_path']})\n\n")
                else:
                    f.write("**Config:** Not found in configs/ directory\n\n")

                # Add plot
                if result['plot_info']:
                    plot_path = result['plot_info']['plot_path']
                    f.write(f"![{result['task_name']} Performance]({plot_path})\n\n")

                    # Add best performance info
                    f.write("**Best Performance:**\n\n")
                    f.write(f"- **Lag:** {result['plot_info']['best_lag']}ms\n")
                    f.write(f"- **{result['plot_info']['primary_metric']}:** {result['plot_info']['best_value']:.4f}\n")

                    if result['plot_info']['secondary_metric'] and result['plot_info']['secondary_value'] is not None:
                        f.write(f"- **{result['plot_info']['secondary_metric']}:** {result['plot_info']['secondary_value']:.4f}\n")

                    f.write("\n")
                else:
                    f.write("*Plot generation failed for this task.*\n\n")

                f.write("---\n\n")


def main():
    """Main function to generate baseline report."""
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / 'baseline-results'
    docs_dir = project_root / 'docs'
    plots_dir = docs_dir / 'baseline_plots'

    # Create plots directory
    plots_dir.mkdir(exist_ok=True)

    # Process all baseline results
    print("Processing baseline results...")
    results = []

    for result_dir in sorted(baseline_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith('.'):
            continue

        print(f"Processing {result_dir.name}...")
        result_data = process_baseline_result(result_dir)

        if result_data is None:
            continue

        # Create plot
        plot_info = create_plot(result_data, plots_dir)

        results.append({
            'task_name': result_data['task_name'],
            'trial_name': result_data['trial_name'],
            'task_type': result_data['task_type'],
            'config_path': result_data['config_path'],
            'plot_info': plot_info
        })

    # Generate markdown report
    print("\nGenerating markdown report...")
    report_file = docs_dir / 'baseline-results.md'
    generate_markdown_report(results, report_file)

    print(f"\nReport generated: {report_file}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Total tasks processed: {len(results)}")


if __name__ == '__main__':
    main()
