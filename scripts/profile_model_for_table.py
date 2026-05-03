#!/usr/bin/env python
"""Profile one config/model pair for comparison-table reporting."""

from __future__ import annotations

import argparse
import csv
import io
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.config import ExperimentConfig, MultiTaskConfig
from core import registry
from utils import data_utils
from utils.config_utils import load_config, get_nested_value
from utils.dataset import RawNeuralDataset
from utils.decoding_utils import (
    CachedFeatureModel,
    _build_cached_fold_loaders,
    _build_fold_loaders,
    _build_full_lag_loader,
    _create_optimizer,
    _get_fold_indices,
    _maybe_shuffle_targets,
    _maybe_prepare_feature_cache_model,
    _maybe_prepare_per_subject_concat_model,
    _normalize_fold_targets,
    _run_epoch,
    _select_requested_folds,
    setup_metrics_and_loss,
)
from main import _build_run_units
from utils.model_utils import build_model_from_spec
from utils.module_loader_utils import import_all_from_package


import_all_from_package("models", recursive=True)
import_all_from_package("tasks", recursive=True)
import_all_from_package("metrics", recursive=True)


REPORT_COLUMNS = [
    "config_path",
    "trial_name",
    "task_name",
    "constructor_name",
    "model_class",
    "device",
    "dtype",
    "lag",
    "fold",
    "total_params",
    "trainable_params",
    "frozen_params",
    "parameter_memory_mb",
    "input_shape",
    "cached_feature_shape",
    "output_shape",
    "feature_cache",
    "per_subject_feature_concat",
    "num_subjects",
    "num_channels",
    "cache_build_seconds",
    "cache_samples_per_second",
    "cache_ms_per_sample",
    "finetune_epoch_seconds",
    "finetune_ms_per_sample",
    "finetune_ms_per_step",
    "inference_ms_per_sample",
    "cache_peak_cuda_memory_mb",
    "finetune_peak_cuda_memory_mb",
]


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_cuda_memory_mb(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    return f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.3f}"


def _reset_cuda_peak(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _select_config(config: ExperimentConfig | MultiTaskConfig) -> ExperimentConfig:
    if isinstance(config, MultiTaskConfig):
        if not config.tasks:
            raise ValueError("MultiTaskConfig must contain at least one task")
        return config.tasks[0]
    return config


def _require_model_spec(config: ExperimentConfig):
    model_spec = getattr(config, "model_spec", None)
    if model_spec is None or not getattr(model_spec, "constructor_name", None):
        raise ValueError(
            "This profiler requires a current-style config with model_spec.constructor_name; "
            "legacy flat model configs are not supported."
        )


def _trial_name(config: ExperimentConfig) -> str:
    if not config.format_fields:
        return config.trial_name
    values = [get_nested_value(config, path) for path in config.format_fields]
    return config.trial_name.format(*values)


def _iter_config_setter_names(config_setter_name):
    if not config_setter_name:
        return []
    if isinstance(config_setter_name, list):
        return config_setter_name
    return [config_setter_name]


def _resolve_preprocessing_fns(experiment_config: ExperimentConfig):
    preprocessing_names = experiment_config.task_config.data_params.preprocessing_fn_name
    if not preprocessing_names:
        return None
    if not isinstance(preprocessing_names, list):
        preprocessing_names = [preprocessing_names]
        experiment_config.task_config.data_params.preprocessing_fn_name = preprocessing_names
    return [registry.data_preprocessor_registry[name] for name in preprocessing_names]


def _prepare_first_run_unit(experiment_config: ExperimentConfig):
    task_name = experiment_config.task_config.task_name
    task_info = registry.task_registry[task_name]

    base_raws = data_utils.load_raws(experiment_config.task_config.data_params)
    base_task_df = task_info["getter"](experiment_config.task_config)
    run_units = _build_run_units(experiment_config, base_raws)
    if not run_units:
        raise ValueError("No run units were created for this config")
    run_unit = run_units[0]

    unit_config = deepcopy(experiment_config)
    unit_task_df = base_task_df.copy(deep=True)
    unit_config.task_config.data_params.subject_ids = list(run_unit["subject_ids"])
    unit_config.task_config.data_params.per_subject_electrodes = deepcopy(
        run_unit["per_subject_electrodes"]
    )
    unit_raws = list(run_unit["raws"])

    for config_setter_name in _iter_config_setter_names(unit_config.config_setter_name):
        config_setter_fn = registry.config_setter_registry[config_setter_name]
        unit_config = config_setter_fn(unit_config, unit_raws, unit_task_df)

    model_spec = unit_config.model_spec
    model_info = registry.model_constructor_registry.get(model_spec.constructor_name, {})
    getter_name = model_spec.model_data_getter or model_info.get("required_data_getter")
    if getter_name:
        if getter_name not in registry.model_data_getter_registry:
            raise ValueError(
                f"Model '{model_spec.constructor_name}' requires data getter "
                f"'{getter_name}' but it is not registered."
            )
        getter_fn = registry.model_data_getter_registry[getter_name]
        unit_task_df, added_columns = getter_fn(unit_task_df, unit_raws, model_spec.params)
        existing_fields = unit_config.task_config.task_specific_config.input_fields or []
        unit_config.task_config.task_specific_config.input_fields = (
            existing_fields + added_columns
        )

    preprocessing_fns = _resolve_preprocessing_fns(unit_config)
    return unit_config, unit_raws, unit_task_df, preprocessing_fns


def _default_lag(config: ExperimentConfig) -> int:
    training_params = config.training_params
    return (
        int(training_params.lag)
        if training_params.lag is not None
        else int(training_params.min_lag)
    )


def _pick_fold(fold_indices, fold_nums, requested_fold: int):
    if requested_fold not in fold_nums:
        raise ValueError(f"Requested fold {requested_fold} is not available: {fold_nums}")
    index = fold_nums.index(requested_fold)
    return fold_nums[index], fold_indices[index]


def _count_params(model: torch.nn.Module) -> tuple[int, int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return total, trainable, frozen, memory


def _same_cached_feature_model(
    model: torch.nn.Module, cache_model: torch.nn.Module | None
) -> bool:
    if cache_model is None or not isinstance(model, CachedFeatureModel):
        return False
    return model.model.__class__ is cache_model.__class__


def _count_profile_params(
    model: torch.nn.Module, cache_model: torch.nn.Module | None = None
) -> tuple[int, int, int, float]:
    model_total, model_trainable, model_frozen, model_memory = _count_params(model)
    if cache_model is None or _same_cached_feature_model(model, cache_model):
        return model_total, model_trainable, model_frozen, model_memory

    cache_total, _, _, cache_memory = _count_params(cache_model)
    total = cache_total + model_total
    trainable = model_trainable
    frozen = total - trainable
    return total, trainable, frozen, cache_memory + model_memory


def _first_parameter_dtype(model: torch.nn.Module) -> str:
    for param in model.parameters():
        return str(param.dtype).replace("torch.", "")
    return "n/a"


def _shape(value: Any) -> str:
    if value is None:
        return "n/a"
    if torch.is_tensor(value):
        return "x".join(str(dim) for dim in value.shape)
    return str(value)


def _format_number(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    return value


def _forward_output_shape(model, loader, device) -> str:
    batch = next(iter(loader))
    Xb, inputs_dict, _ = batch
    Xb = Xb.to(device)
    inputs_dict = {
        key: val.to(device) if torch.is_tensor(val) else val
        for key, val in inputs_dict.items()
    }
    model.eval()
    with torch.no_grad():
        out = model(Xb, **inputs_dict)
    return _shape(out)


def _time_inference_ms_per_sample(model, loader, device, warmup_iters, timing_iters):
    batch = next(iter(loader))
    Xb, inputs_dict, _ = batch
    batch_size = Xb.shape[0]
    Xb = Xb.to(device)
    inputs_dict = {
        key: val.to(device) if torch.is_tensor(val) else val
        for key, val in inputs_dict.items()
    }
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(Xb, **inputs_dict)
        _sync(device)
        start = time.perf_counter()
        for _ in range(timing_iters):
            model(Xb, **inputs_dict)
        _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, timing_iters * batch_size)


def _warmup_train_batch(
    model, loader, device, training_params, all_fns, optimizer, warmup_iters
):
    if warmup_iters <= 0:
        return
    batch = next(iter(loader))
    Xb, inputs_dict, yb = batch
    Xb = Xb.to(device)
    yb = yb.to(device)
    inputs_dict = {
        key: val.to(device) if torch.is_tensor(val) else val
        for key, val in inputs_dict.items()
    }
    model.train()
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        out = model(Xb, **inputs_dict)
        loss = None
        for i, loss_name in enumerate(training_params.losses):
            loss_val = training_params.loss_weights[i] * all_fns[loss_name](out, yb)
            loss = loss_val if loss is None else loss + loss_val
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    _sync(device)


def _prepare_trainable_model(model_spec, lag, fold, loaders, training_params, device):
    model = build_model_from_spec(model_spec, lag=lag, fold=fold).to(device)
    optimizer = _create_optimizer(model, training_params)
    model, loaders, probe_optimizer = _maybe_prepare_per_subject_concat_model(
        model, loaders, model_spec, training_params, device
    )
    if probe_optimizer is not None:
        optimizer = probe_optimizer
    elif getattr(model_spec, "feature_cache", False):
        model = CachedFeatureModel(model).to(device)
    return model, loaders, optimizer


def profile_config(
    config_path: str,
    lag: int | None = None,
    fold: int = 1,
    device_name: str = "auto",
    warmup_iters: int = 3,
    timing_iters: int = 20,
):
    loaded_config = load_config(config_path, {})
    experiment_config = _select_config(loaded_config)
    _require_model_spec(experiment_config)

    lag = _default_lag(experiment_config) if lag is None else int(lag)
    device = _resolve_device(device_name)
    training_params = experiment_config.training_params
    model_spec = experiment_config.model_spec

    unit_config, raws, task_df, preprocessing_fns = _prepare_first_run_unit(
        experiment_config
    )
    training_params = unit_config.training_params
    model_spec = unit_config.model_spec

    raw_dataset = RawNeuralDataset(
        raws,
        task_df,
        unit_config.task_config.data_params.window_width,
        preprocessing_fns,
        unit_config.task_config.data_params.preprocessor_params,
    )
    neural_data, Y, data_df, subject_channel_counts = raw_dataset.get_data_for_lag(lag)
    Y = _maybe_shuffle_targets(Y, training_params)

    fold_indices = _get_fold_indices(
        neural_data, data_df, unit_config.task_config, training_params
    )
    fold_indices, fold_nums = _select_requested_folds(fold_indices, training_params)
    fold_num, (tr_idx, va_idx, te_idx) = _pick_fold(fold_indices, fold_nums, fold)
    split_indices = {"train": tr_idx, "val": va_idx, "test": te_idx}
    target_splits = _normalize_fold_targets(Y, tr_idx, va_idx, te_idx, training_params)

    use_cache = getattr(model_spec, "feature_cache", False) or getattr(
        model_spec, "per_subject_feature_concat", False
    )

    cached_features = None
    cached_extra_inputs = None
    cache_model = None
    cache_build_seconds = 0.0
    cache_peak = "n/a"
    if use_cache:
        full_lag_loader = _build_full_lag_loader(
            neural_data, data_df, Y, unit_config.task_config, training_params
        )
        _reset_cuda_peak(device)
        _sync(device)
        start = time.perf_counter()
        (
            cached_features,
            cached_extra_inputs,
            cache_model,
        ) = _maybe_prepare_feature_cache_model(
            model_spec,
            lag,
            full_lag_loader,
            training_params,
            device,
            subject_channel_counts=(
                subject_channel_counts
                if getattr(model_spec, "per_subject_feature_concat", False)
                else None
            ),
            return_cache_model=True,
        )
        _sync(device)
        cache_build_seconds = time.perf_counter() - start
        cache_peak = _peak_cuda_memory_mb(device)
        loaders = _build_cached_fold_loaders(
            cached_features,
            cached_extra_inputs,
            split_indices,
            target_splits,
            training_params,
        )
    else:
        loaders = _build_fold_loaders(
            neural_data,
            data_df,
            unit_config.task_config,
            split_indices,
            target_splits,
            training_params,
        )

    all_fns = setup_metrics_and_loss(training_params)
    metric_names = all_fns.keys()
    model, loaders, optimizer = _prepare_trainable_model(
        model_spec, lag, fold_num, loaders, training_params, device
    )

    input_shape = _shape(neural_data)
    cached_feature_shape = _shape(cached_features)
    output_shape = _forward_output_shape(model, loaders["train"], device)

    _warmup_train_batch(
        model,
        loaders["train"],
        device,
        training_params,
        all_fns,
        optimizer,
        warmup_iters,
    )

    _reset_cuda_peak(device)
    _sync(device)
    start = time.perf_counter()
    _run_epoch(
        model,
        loaders["train"],
        device,
        training_params,
        all_fns,
        metric_names,
        model_spec.params,
        optimizer=optimizer,
    )
    _sync(device)
    finetune_epoch_seconds = time.perf_counter() - start
    finetune_peak = _peak_cuda_memory_mb(device)

    inference_ms_per_sample = _time_inference_ms_per_sample(
        model, loaders["test"], device, warmup_iters, timing_iters
    )

    total_params, trainable_params, frozen_params, param_memory = _count_profile_params(
        model, cache_model
    )
    num_train_samples = len(loaders["train"].dataset)
    num_steps = len(loaders["train"])
    num_samples = len(neural_data)
    num_channels = sum(subject_channel_counts) if subject_channel_counts else neural_data.shape[1]
    cache_sps = num_samples / cache_build_seconds if cache_build_seconds > 0 else "n/a"
    cache_ms = cache_build_seconds * 1000.0 / num_samples if cache_build_seconds > 0 else "n/a"

    return {
        "config_path": config_path,
        "trial_name": _trial_name(unit_config),
        "task_name": unit_config.task_config.task_name,
        "constructor_name": model_spec.constructor_name,
        "model_class": model.__class__.__name__,
        "device": str(device),
        "dtype": _first_parameter_dtype(model),
        "lag": lag,
        "fold": fold_num,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "parameter_memory_mb": param_memory,
        "input_shape": input_shape,
        "cached_feature_shape": cached_feature_shape,
        "output_shape": output_shape,
        "feature_cache": bool(getattr(model_spec, "feature_cache", False)),
        "per_subject_feature_concat": bool(
            getattr(model_spec, "per_subject_feature_concat", False)
        ),
        "num_subjects": len(subject_channel_counts),
        "num_channels": num_channels,
        "cache_build_seconds": cache_build_seconds if use_cache else "n/a",
        "cache_samples_per_second": cache_sps,
        "cache_ms_per_sample": cache_ms,
        "finetune_epoch_seconds": finetune_epoch_seconds,
        "finetune_ms_per_sample": finetune_epoch_seconds * 1000.0 / num_train_samples,
        "finetune_ms_per_step": finetune_epoch_seconds * 1000.0 / num_steps,
        "inference_ms_per_sample": inference_ms_per_sample,
        "cache_peak_cuda_memory_mb": cache_peak,
        "finetune_peak_cuda_memory_mb": finetune_peak,
    }


def row_to_csv(row: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REPORT_COLUMNS)
    writer.writeheader()
    writer.writerow({key: _format_number(row.get(key, "n/a")) for key in REPORT_COLUMNS})
    return output.getvalue().strip()


def row_to_latex(row: dict[str, Any]) -> str:
    def esc(value):
        value = str(_format_number(value)).replace("\\", "\\textbackslash{}")
        return value.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

    headers = " & ".join(esc(col) for col in REPORT_COLUMNS)
    values = " & ".join(esc(row.get(col, "n/a")) for col in REPORT_COLUMNS)
    column_spec = "l" * len(REPORT_COLUMNS)
    return "\n".join(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            headers + r" \\",
            r"\hline",
            values + r" \\",
            r"\end{tabular}",
        ]
    )


def write_csv(path: str, row: dict[str, Any]):
    path_obj = Path(path)
    exists = path_obj.exists() and path_obj.stat().st_size > 0
    with path_obj.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REPORT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: _format_number(row.get(key, "n/a")) for key in REPORT_COLUMNS})


def write_latex(path: str, row: dict[str, Any]):
    Path(path).write_text(row_to_latex(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile the first task/model in a model_spec config."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--lag", type=int, default=None)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-latex", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    row = profile_config(
        args.config,
        lag=args.lag,
        fold=args.fold,
        device_name=args.device,
        warmup_iters=args.warmup_iters,
        timing_iters=args.timing_iters,
    )

    print(row_to_csv(row))
    print()
    print(row_to_latex(row))

    if args.output_csv:
        write_csv(args.output_csv, row)
    if args.output_latex:
        write_latex(args.output_latex, row)


if __name__ == "__main__":
    main()
