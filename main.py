import os
from datetime import datetime
from dataclasses import asdict
from copy import deepcopy

import numpy as np
import torch

from utils import data_utils
from utils import decoding_utils
from utils.atlas_utils import (
    REGION_GROUPS,
    build_electrode_region_map,
    slugify_region_name,
)
import random
from utils.module_loader_utils import import_all_from_package
from core import registry
from core.config import (
    TaskConfig,
    DataParams,
    MultiTaskConfig,
    ExperimentConfig,
    RunMode,
    dict_to_config,
)
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

    # Shared setup
    base_raws = data_utils.load_raws(experiment_config.task_config.data_params)
    task_getter = task_info["getter"]
    base_task_df = task_getter(experiment_config.task_config)

    model_spec = experiment_config.model_spec
    if (
        experiment_config.run_mode != RunMode.COMBINED
        and model_spec.per_subject_feature_concat
    ):
        raise ValueError(
            "Split run modes only support non-concat runs; "
            "set model_spec.per_subject_feature_concat to False."
        )

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

    subject_ids = experiment_config.task_config.data_params.subject_ids
    if len(base_raws) != len(subject_ids):
        raise ValueError(
            "Loaded raws must align one-to-one with configured subject_ids."
        )

    run_units = _build_run_units(experiment_config, base_raws)
    for run_unit in run_units:
        unit_config = deepcopy(experiment_config)
        unit_task_df = base_task_df.copy(deep=True)
        unit_config.task_config.data_params.subject_ids = list(run_unit["subject_ids"])
        unit_config.task_config.data_params.per_subject_electrodes = deepcopy(
            run_unit["per_subject_electrodes"]
        )

        unit_raws = run_unit["raws"]

        if unit_config.config_setter_name:
            for config_setter_name in _iter_config_setter_names(
                unit_config.config_setter_name
            ):
                config_setter_fn = registry.config_setter_registry[config_setter_name]
                unit_config = config_setter_fn(unit_config, unit_raws, unit_task_df)

        unit_model_spec = unit_config.model_spec
        model_info = registry.model_constructor_registry.get(
            unit_model_spec.constructor_name, {}
        )
        getter_name = (
            unit_model_spec.model_data_getter
            or model_info.get("required_data_getter")
        )

        if getter_name:
            if getter_name not in registry.model_data_getter_registry:
                raise ValueError(
                    f"Model '{unit_model_spec.constructor_name}' requires data getter "
                    f"'{getter_name}' but it is not registered. "
                    f"Available getters: {list(registry.model_data_getter_registry.keys())}"
                )

            getter_fn = registry.model_data_getter_registry[getter_name]
            unit_task_df, added_columns = getter_fn(
                unit_task_df, unit_raws, unit_model_spec.params
            )
            existing_fields = (
                unit_config.task_config.task_specific_config.input_fields or []
            )
            unit_config.task_config.task_specific_config.input_fields = (
                existing_fields + added_columns
            )
            print(f"Model data getter '{getter_name}' added columns: {added_columns}")

        preprocessing_fns = _resolve_preprocessing_fns(unit_config)

        unit_output_dir = (
            output_dir
            if run_unit["dir_name"] is None
            else os.path.join(output_dir, run_unit["dir_name"])
        )
        unit_checkpoint_dir = (
            checkpoint_dir
            if run_unit["dir_name"] is None
            else os.path.join(checkpoint_dir, run_unit["dir_name"])
        )
        unit_tensorboard_dir = (
            tensorboard_dir
            if run_unit["dir_name"] is None
            else os.path.join(tensorboard_dir, run_unit["dir_name"])
        )
        os.makedirs(unit_output_dir, exist_ok=True)
        os.makedirs(unit_checkpoint_dir, exist_ok=True)
        os.makedirs(unit_tensorboard_dir, exist_ok=True)

        decoding_utils.run_training_over_lags(
            lags,
            unit_raws,
            unit_task_df,
            preprocessing_fns,
            unit_model_spec,
            unit_config.task_config.task_name,
            training_params=unit_config.training_params,
            task_config=unit_config.task_config,
            output_dir=unit_output_dir,
            checkpoint_dir=unit_checkpoint_dir,
            tensorboard_dir=unit_tensorboard_dir,
            write_to_tensorboard=unit_config.training_params.tensorboard_logging,
        )

    return checkpoint_dir


def _resolve_preprocessing_fns(experiment_config: ExperimentConfig):
    preprocessing_names = experiment_config.task_config.data_params.preprocessing_fn_name
    if not preprocessing_names:
        return None

    if not isinstance(preprocessing_names, list):
        preprocessing_names = [preprocessing_names]
        experiment_config.task_config.data_params.preprocessing_fn_name = preprocessing_names

    return [registry.data_preprocessor_registry[fn_name] for fn_name in preprocessing_names]


def _iter_config_setter_names(config_setter_name):
    if not config_setter_name:
        return []
    if isinstance(config_setter_name, list):
        return config_setter_name
    return [config_setter_name]


def _build_run_units(experiment_config: ExperimentConfig, base_raws):
    subject_ids = experiment_config.task_config.data_params.subject_ids
    per_subject_electrodes = experiment_config.task_config.data_params.per_subject_electrodes

    if experiment_config.run_mode == RunMode.COMBINED:
        return [
            {
                "dir_name": None,
                "subject_ids": list(subject_ids),
                "per_subject_electrodes": deepcopy(per_subject_electrodes),
                "raws": list(base_raws),
            }
        ]

    if experiment_config.run_mode == RunMode.PER_SUBJECT:
        return [
            {
                "dir_name": f"subject_{subject_id}",
                "subject_ids": [subject_id],
                "per_subject_electrodes": (
                    {subject_id: per_subject_electrodes[subject_id]}
                    if per_subject_electrodes and subject_id in per_subject_electrodes
                    else None
                ),
                "raws": [raw],
            }
            for subject_id, raw in zip(subject_ids, base_raws)
        ]

    if experiment_config.run_mode == RunMode.PER_REGION:
        region_map = build_electrode_region_map(
            subject_ids=subject_ids,
            raws=base_raws,
            region_groups=REGION_GROUPS,
        )
        if experiment_config.regions is not None:
            requested_regions = set(experiment_config.regions)
            available_regions = set(region_map.keys())
            unknown_regions = requested_regions - available_regions
            if unknown_regions:
                raise ValueError(
                    "Unknown regions requested: "
                    f"{sorted(unknown_regions)}. Available regions: "
                    f"{sorted(available_regions)}"
                )
            region_map = {
                region_name: region_subjects
                for region_name, region_subjects in region_map.items()
                if region_name in requested_regions
            }

        run_units = []
        for region_name, region_subjects in region_map.items():
            region_raws = []
            region_subject_ids = []
            for subject_id, raw in zip(subject_ids, base_raws):
                if subject_id in region_subjects:
                    region_raws.append(raw)
                    region_subject_ids.append(subject_id)

            if not region_raws:
                continue

            run_units.append(
                {
                    "dir_name": f"region_{slugify_region_name(region_name)}",
                    "subject_ids": region_subject_ids,
                    "per_subject_electrodes": deepcopy(region_subjects),
                    "raws": region_raws,
                }
            )

        return run_units

    raise ValueError(f"Unsupported run mode: {experiment_config.run_mode}")


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
