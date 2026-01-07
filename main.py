import os
from datetime import datetime

import numpy as np
import torch

from utils import data_utils
from utils import decoding_utils
import random
from utils.module_loader_utils import import_all_from_package
from core import registry
from core.config import TaskConfig, DataParams, MultiTaskConfig, ExperimentConfig, dict_to_config
from utils.config_utils import (
    parse_known_args,
    load_config,
    apply_overrides,
    get_nested_value,
    interpolate_prev_checkpoint_dir,
)

import_all_from_package("models", recursive=True)
import_all_from_package("tasks", recursive=True)
import_all_from_package("metrics", recursive=True)


def set_seed(seed=42, cudnn_deterministic=False):
    """
    Set random seeds for reproducibility across numpy, pytorch, and python's random module.

    Args:
        seed (int): Random seed value
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_task(experiment_config: ExperimentConfig) -> str:
    """Run a single training task.

    Args:
        experiment_config: Configuration for this task

    Returns:
        str: Checkpoint directory for this task (with timestamp)
    """
    task_name = experiment_config.task_config.task_name
    task_info = registry.task_registry[task_name]

    # Load all data
    raws = data_utils.load_raws(experiment_config.task_config.data_params)
    task_getter = task_info["getter"]
    task_df = task_getter(experiment_config.task_config)

    # Apply config setters
    if experiment_config.config_setter_name:
        for config_setter_name in experiment_config.config_setter_name:
            config_setter_fn = registry.config_setter_registry[config_setter_name]
            experiment_config = config_setter_fn(experiment_config, raws, task_df)

    # User defined preprocessing function
    preprocessing_fns = None
    if experiment_config.task_config.data_params.preprocessing_fn_name:
        if not isinstance(
            experiment_config.task_config.data_params.preprocessing_fn_name, list
        ):
            experiment_config.task_config.data_params.preprocessing_fn_name = [
                experiment_config.task_config.data_params.preprocessing_fn_name
            ]
        preprocessing_fns = []
        for fn_name in experiment_config.task_config.data_params.preprocessing_fn_name:
            preprocessing_fns.append(registry.data_preprocessor_registry[fn_name])

    # User defined model specification - will be built at each lag
    model_spec = experiment_config.model_spec

    # Generate trial name if user specified format string
    trial_name = experiment_config.trial_name
    if experiment_config.format_fields:
        format_values = [
            get_nested_value(experiment_config, s)
            for s in experiment_config.format_fields
        ]
        trial_name = trial_name.format(*format_values)
    # Append timestamp to prevent accidental overwriting
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_name = f"{trial_name}_{timestamp}"
    output_dir = os.path.join(experiment_config.output_dir, trial_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(experiment_config.checkpoint_dir, trial_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tensorboard_dir = os.path.join(experiment_config.tensorboard_dir, trial_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Write config to output_dir so it is easy to tell what parameters led to these results
    import yaml
    from dataclasses import asdict

    with open(os.path.join(output_dir, "config.yml"), "w") as fp:
        yaml.dump(asdict(experiment_config), fp, default_flow_style=False)

    # Decide what lags we need to train over
    if experiment_config.training_params.lag is not None:
        lags = [experiment_config.training_params.lag]
    else:
        lags = np.arange(
            experiment_config.training_params.min_lag,
            experiment_config.training_params.max_lag,
            experiment_config.training_params.lag_step_size,
        )

    decoding_utils.run_training_over_lags(
        lags,
        raws,
        task_df,
        preprocessing_fns,
        model_spec,
        experiment_config.task_config.task_name,
        training_params=experiment_config.training_params,
        task_config=experiment_config.task_config,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        write_to_tensorboard=experiment_config.training_params.tensorboard_logging,
    )

    return checkpoint_dir


def run_multi_task(multi_config: MultiTaskConfig):
    """Run multiple tasks sequentially.

    Args:
        multi_config: Multi-task configuration
    """
    prev_checkpoint_dir = None

    for task_idx, task_config in enumerate(multi_config.tasks):
        print("\n" + "=" * 80)
        print(f"RUNNING TASK {task_idx + 1}/{len(multi_config.tasks)}: {task_config.trial_name}")
        print("=" * 80 + "\n")

        # Interpolate {prev_checkpoint_dir} in model_spec if this is not the first task
        if prev_checkpoint_dir:
            task_config.model_spec = interpolate_prev_checkpoint_dir(
                task_config.model_spec, prev_checkpoint_dir
            )

        # Apply shared params AFTER checkpoint interpolation but BEFORE other processing
        # This allows shared params to override values set in individual task configs
        if multi_config.shared_params:
            task_config = apply_overrides(task_config, multi_config.shared_params)

        # Run this task
        checkpoint_dir = run_single_task(task_config)

        # Update prev_checkpoint_dir for next task
        prev_checkpoint_dir = checkpoint_dir

        print(f"\nTask {task_idx + 1} completed. Checkpoint directory: {checkpoint_dir}\n")


def main():
    args, overrides = parse_known_args()
    config = load_config(args.config, overrides)  # Auto-detect single/multi

    # Set random seed
    if isinstance(config, MultiTaskConfig):
        # Use seed from first task
        seed = config.tasks[0].training_params.random_seed
        cudnn_det = config.tasks[0].training_params.cudnn_deterministic
    else:
        seed = config.training_params.random_seed
        cudnn_det = config.training_params.cudnn_deterministic

    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed, cudnn_det)

    # Run single or multi-task
    if isinstance(config, MultiTaskConfig):
        run_multi_task(config)
    else:
        run_single_task(config)


if __name__ == "__main__":
    main()
