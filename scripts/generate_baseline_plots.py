#!/usr/bin/env python3
"""
Generate baseline results plots for all tasks.

This script processes baseline results from the baseline-results directory
and creates plots of performance metrics across lags.
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

# Define human-readable names for task types
TASK_TYPE_NAMES = {
    'content_noncontent_task': 'Content/Non-Content Classification',
    'ensemble_model': 'Word Embedding Decoding',
    'gpt_surprise': 'GPT Surprisal (Regression)',
    'gpt_surprise_multiclass': 'GPT Surprisal (Multiclass)',
    'pos_task': 'Part of Speech',
    'sentence_onset': 'Sentence Onset Detection',
    'volume_level': 'Volume Level Prediction'
}

# Define human-readable names for metrics
METRIC_DISPLAY_NAMES = {
    'test_roc_auc_mean': 'ROC-AUC',
    'test_roc_auc_multiclass_mean': 'ROC-AUC (Multiclass)',
    'test_word_avg_auc_roc_mean': 'AUC-ROC',
    'test_word_top_5_mean': 'Top-5 Accuracy',
    'test_corr_mean': 'Correlation'
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
    # First try to match on both task_name AND trial_name for best accuracy
    for ext in ['*.yml', '*.yaml']:
        for config_file in configs_dir.rglob(ext):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
                    if config and config.get('task_name') == task_name_from_config and config.get('trial_name') == trial_name:
                        return config_file
            except Exception as e:
                continue

    # If no exact match, try matching on task_name only (more specific than trial_name)
    for ext in ['*.yml', '*.yaml']:
        for config_file in configs_dir.rglob(ext):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
                    if config and config.get('task_name') == task_name_from_config:
                        return config_file
            except Exception as e:
                continue

    # Finally, try matching on trial_name
    for ext in ['*.yml', '*.yaml']:
        for config_file in configs_dir.rglob(ext):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
                    if config and config.get('trial_name') == trial_name:
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
        'result_dir': result_dir,
        'result_dir_name': result_dir.name
    }


def create_combined_ensemble_plots(ensemble_results, output_dir):
    """Create combined AUC ROC plot for all ensemble models."""
    if not ensemble_results:
        return {}

    # Define colors and markers for each model type
    model_styles = {
        'arbitrary': {'color': 'blue', 'marker': 'o', 'label': 'Arbitrary'},
        'glove': {'color': 'green', 'marker': 's', 'label': 'GloVe'},
        'gpt2': {'color': 'red', 'marker': '^', 'label': 'GPT-2'}
    }

    # Create figure for AUC ROC
    fig_auc, ax_auc = plt.subplots(figsize=(10, 6))

    best_performances = {}

    for result in ensemble_results:
        task_name = result['task_name']
        df = result['df']

        # Determine model type from task name
        model_type = None
        for key in model_styles.keys():
            if key in task_name:
                model_type = key
                break

        if model_type is None:
            continue

        style = model_styles[model_type]

        # Plot AUC ROC
        if 'test_word_avg_auc_roc_mean' in df.columns:
            ax_auc.plot(df['lags'], df['test_word_avg_auc_roc_mean'],
                       color=style['color'], marker=style['marker'],
                       linewidth=2, markersize=6, label=style['label'])

            # Find best performance
            best_idx = df['test_word_avg_auc_roc_mean'].idxmax()
            best_performances[task_name] = {
                'best_lag': int(df.loc[best_idx, 'lags']),
                'test_word_avg_auc_roc_mean': df.loc[best_idx, 'test_word_avg_auc_roc_mean'],
                'test_word_top_5_mean': df.loc[best_idx, 'test_word_top_5_mean'] if 'test_word_top_5_mean' in df.columns else None
            }

    # Finalize AUC ROC plot
    ax_auc.set_xlabel('Lag (ms)', fontsize=12)
    ax_auc.set_ylabel('Test Word Avg AUC ROC', fontsize=12)
    ax_auc.set_title('Word Embedding Decoding - AUC ROC Performance', fontsize=14)
    ax_auc.grid(True, alpha=0.3)
    ax_auc.legend(loc='best')
    plt.tight_layout()

    # Save AUC ROC plot
    auc_plot_path = output_dir / 'ensemble_models_auc_roc.png'
    fig_auc.savefig(auc_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_auc)

    return {
        'auc_plot_path': auc_plot_path,
        'best_performances': best_performances
    }


def create_llm_comparison_plot(baseline_dir, output_dir):
    """Create comparison plot for LLM decoding with and without brain data."""
    # Find the most recent llm_token_finetune and llm_decoding_control results
    finetune_dirs = sorted([d for d in baseline_dir.iterdir() if 'llm_token_finetune' in d.name and d.is_dir()], reverse=True)
    control_dirs = sorted([d for d in baseline_dir.iterdir() if 'llm_decoding_control' in d.name and d.is_dir()], reverse=True)

    if not finetune_dirs or not control_dirs:
        print("Warning: Could not find both llm_token_finetune and llm_decoding_control results")
        return None

    finetune_dir = finetune_dirs[0]
    control_dir = control_dirs[0]

    finetune_file = finetune_dir / 'lag_performance.csv'
    control_file = control_dir / 'lag_performance.csv'

    if not finetune_file.exists() or not control_file.exists():
        print("Warning: Missing lag_performance.csv files for LLM comparison")
        return None

    # Read the data
    finetune_data = pd.read_csv(finetune_file)
    control_data = pd.read_csv(control_file)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot test perplexity for both experiments
    ax.plot(finetune_data['lags'], finetune_data['test_perplexity_mean'],
            marker='o', linewidth=2, markersize=6, label='LLM Token Finetuning (Brain Data)')

    ax.plot(control_data['lags'], control_data['test_perplexity_mean'],
            marker='s', linewidth=2, markersize=6, label='LLM Decoding (No Brain Data)')

    ax.set_xlabel('Lag (ms)', fontsize=12)
    ax.set_ylabel('Test Perplexity', fontsize=12)
    ax.set_title('LLM Decoding: Brain Data vs Control', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / 'llm_decoding_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Get best performances
    best_finetune_idx = finetune_data['test_perplexity_mean'].idxmin()
    best_control_idx = control_data['test_perplexity_mean'].idxmin()

    return {
        'plot_path': plot_path,
        'finetune_dir': finetune_dir.name,
        'control_dir': control_dir.name,
        'finetune_best_lag': int(finetune_data.loc[best_finetune_idx, 'lags']),
        'finetune_best_perplexity': finetune_data.loc[best_finetune_idx, 'test_perplexity_mean'],
        'control_best_lag': int(control_data.loc[best_control_idx, 'lags']),
        'control_best_perplexity': control_data.loc[best_control_idx, 'test_perplexity_mean']
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

    # Get human-readable task name
    task_display_name = TASK_TYPE_NAMES.get(task_type, task_type.replace('_', ' ').title())

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

    # Use human-readable metric name in title
    metric_display_name = METRIC_DISPLAY_NAMES.get(primary_metric, ylabel)
    ax.set_title(f'{task_display_name}\nBest {metric_display_name}: {best_value:.4f} at {best_lag}ms lag', fontsize=14)
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


def main():
    """Main function to generate baseline plots."""
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / 'baseline-results'
    docs_dir = project_root / 'docs'
    plots_dir = docs_dir / 'baseline_plots'

    # Create plots directory
    plots_dir.mkdir(exist_ok=True)

    # Process all baseline results
    print("Processing baseline results...")
    print("=" * 80)
    results = []
    ensemble_results = []

    for result_dir in sorted(baseline_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith('.'):
            continue

        # Skip torch_ridge volume level model - only show simple
        if 'volume_level_torch_ridge' in result_dir.name:
            print(f"Skipping {result_dir.name} (using simple model only)...")
            continue

        print(f"\nProcessing {result_dir.name}...")
        result_data = process_baseline_result(result_dir)

        if result_data is None:
            continue

        # Collect ensemble model results separately
        if result_data['task_type'] == 'ensemble_model':
            ensemble_results.append(result_data)

        # Create plot for non-ensemble models only
        plot_info = None
        if result_data['task_type'] != 'ensemble_model':
            plot_info = create_plot(result_data, plots_dir)
            if plot_info:
                print(f"  Task Type: {TASK_TYPE_NAMES.get(result_data['task_type'], result_data['task_type'])}")
                print(f"  Config: {result_data['config_path'].relative_to(project_root) if result_data['config_path'] else 'Not found'}")
                print(f"  Data: baseline-results/{result_data['result_dir_name']}/lag_performance.csv")
                print(f"  Plot: {plot_info['plot_path'].name}")
                print(f"  Best Lag: {plot_info['best_lag']}ms")
                primary_metric_display = METRIC_DISPLAY_NAMES.get(plot_info['primary_metric'], plot_info['primary_metric'])
                print(f"  Best {primary_metric_display}: {plot_info['best_value']:.4f}")

        results.append({
            'task_name': result_data['task_name'],
            'trial_name': result_data['trial_name'],
            'task_type': result_data['task_type'],
            'config_path': result_data['config_path'],
            'plot_info': plot_info,
            'result_dir_name': result_data['result_dir_name']
        })

    # Create combined plots for ensemble models
    ensemble_plot_info = None
    if ensemble_results:
        print("\n" + "=" * 80)
        print("\nCreating combined ensemble model plots...")
        ensemble_plot_info = create_combined_ensemble_plots(ensemble_results, plots_dir)
        print(f"\nWord Embedding Decoding:")
        print(f"  Plot: {ensemble_plot_info['auc_plot_path'].name}")

        for result_data in ensemble_results:
            task_name = result_data['task_name']
            if task_name in ensemble_plot_info['best_performances']:
                perf = ensemble_plot_info['best_performances'][task_name]

                model_name = None
                if 'arbitrary' in task_name:
                    model_name = 'Arbitrary'
                elif 'glove' in task_name:
                    model_name = 'GloVe'
                elif 'gpt2' in task_name:
                    model_name = 'GPT-2'

                if model_name:
                    print(f"\n  {model_name}:")
                    print(f"    Config: {result_data['config_path'].relative_to(project_root) if result_data['config_path'] else 'Not found'}")
                    print(f"    Data: baseline-results/{result_data['result_dir_name']}/lag_performance.csv")
                    print(f"    Best Lag: {perf['best_lag']}ms")
                    print(f"    Best AUC-ROC: {perf['test_word_avg_auc_roc_mean']:.4f}")

    # Create LLM comparison plot
    print("\n" + "=" * 80)
    print("\nCreating LLM decoding comparison plot...")
    llm_plot_info = create_llm_comparison_plot(baseline_dir, plots_dir)
    if llm_plot_info:
        print(f"\nLLM Decoding Comparison:")
        print(f"  Plot: {llm_plot_info['plot_path'].name}")
        print(f"\n  LLM Token Finetuning (Brain Data):")
        print(f"    Data: baseline-results/{llm_plot_info['finetune_dir']}/lag_performance.csv")
        print(f"    Best Lag: {llm_plot_info['finetune_best_lag']}ms")
        print(f"    Best Perplexity: {llm_plot_info['finetune_best_perplexity']:.4f}")
        print(f"\n  LLM Decoding (No Brain Data - Control):")
        print(f"    Data: baseline-results/{llm_plot_info['control_dir']}/lag_performance.csv")
        print(f"    Best Lag: {llm_plot_info['control_best_lag']}ms")
        print(f"    Best Perplexity: {llm_plot_info['control_best_perplexity']:.4f}")

    print("\n" + "=" * 80)
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Total tasks processed: {len(results)}")
    print("\nNOTE: Plot generation complete. Update docs/baseline-results.md manually if needed.")


if __name__ == '__main__':
    main()
