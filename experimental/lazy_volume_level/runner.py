from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from mup import MuAdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import ModelSpec, TaskConfig, TrainingParams
from experimental.lazy_volume_level.collate import LazyEpochCollator
from experimental.lazy_volume_level.dataset import LazyWindowDataset
from utils import data_utils
from utils.decoding_utils import (
    TENSORBOARD_AVAILABLE,
    SummaryWriter,
    compute_all_metrics,
    compute_loss,
    create_lr_scheduler,
    log_metrics_to_tensorboard,
    setup_early_stopping_state,
    setup_metrics_and_loss,
    should_update_best,
    should_update_gradient_accumulation,
)
from utils.fold_utils import get_sequential_folds, get_zero_shot_folds
from utils.model_utils import build_model_from_spec


def _filter_valid_rows(raws, task_df: pd.DataFrame, lag: int, window_width: float):
    tmin = lag / 1000 - window_width / 2
    tmax = lag / 1000 + window_width / 2 - 2e-3
    valid_mask = np.ones(len(task_df), dtype=bool)
    for raw in raws:
        data_duration = raw.times[-1]
        valid_mask &= (task_df.start + tmin >= 0) & (task_df.start + tmax <= data_duration)

    filtered_df = task_df[valid_mask].reset_index(drop=True)
    if len(filtered_df) == 0:
        raise ValueError("No valid events found within data time bounds")

    return filtered_df, filtered_df.target.to_numpy()


def train_decoding_model_lazy(
    raws,
    Y: torch.Tensor,
    data_df: pd.DataFrame,
    preprocessing_fns,
    preprocessor_params,
    window_width: float,
    model_spec: ModelSpec,
    task_name: str,
    task_config: TaskConfig,
    lag: int,
    training_params: TrainingParams,
    checkpoint_dir: str,
    write_to_tensorboard: bool = False,
    tensorboard_dir: str = "event_logs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if training_params.fold_type == "sequential_folds":
        fold_indices = get_sequential_folds(Y, num_folds=training_params.n_folds)
    elif training_params.fold_type == "zero_shot_folds":
        fold_indices = get_zero_shot_folds(
            data_df[task_config.data_params.word_column].values,
            num_folds=training_params.n_folds,
        )
    else:
        raise ValueError(f"Unknown fold_type: {training_params.fold_type}")

    fold_nums_all = list(range(1, len(fold_indices) + 1))
    fold_ids = getattr(training_params, "fold_ids", None)
    if fold_ids is not None:
        seen = set()
        selected_fold_nums = [
            k for k in fold_ids if not (k in seen or seen.add(k))
        ]
        fold_indices = [fold_indices[k - 1] for k in selected_fold_nums]
        fold_nums_all = selected_fold_nums

    all_fns = setup_metrics_and_loss(training_params)
    metric_names = all_fns.keys()
    phases = ("train", "val", "test")
    cv_results = {
        f"{phase}_{name}": []
        for phase in phases
        for name in metric_names
        if name != "confusion_matrix"
    }
    cv_results["num_epochs"] = []
    cv_results["fold_nums"] = []

    models, histories = [], []

    def run_epoch(model, loader, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        sums = {name: None if name == "confusion_matrix" else 0.0 for name in metric_names}
        sums["loss"] = 0.0
        grad_steps = training_params.grad_accumulation_steps

        if is_train:
            optimizer.zero_grad()

        for i, batch_data in enumerate(loader):
            Xb, inputs_dict, yb = batch_data
            Xb = Xb.to(device)
            inputs_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in inputs_dict.items()
            }
            yb = yb.to(device)

            if is_train:
                out = model(Xb, **inputs_dict)
                loss = compute_loss(out, yb, training_params, all_fns) / grad_steps
                loss.backward()
                if should_update_gradient_accumulation(i, len(loader), grad_steps):
                    if (
                        getattr(training_params, "clip_grad_norm", 0.0)
                        and float(training_params.clip_grad_norm) > 0.0
                    ):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=float(training_params.clip_grad_norm),
                        )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    out = model(Xb, **inputs_dict)
                    loss = compute_loss(out, yb, training_params, all_fns)

            batch_metrics = compute_all_metrics(out, yb, all_fns, model_spec.params)
            for name, val in batch_metrics.items():
                if sums[name] is None:
                    sums[name] = val
                else:
                    sums[name] += val

            if torch.is_tensor(loss):
                loss = loss.detach().mean().item()
            sums["loss"] += loss

        result = {
            name: (
                sums[name] if name == "confusion_matrix" else sums[name] / len(loader)
            )
            for name in sums
        }
        if "cross_entropy" in result:
            result["perplexity"] = np.exp(result["cross_entropy"])
        return result

    collate_fn = LazyEpochCollator(
        raws,
        lag=lag,
        window_width=window_width,
        preprocessing_fns=preprocessing_fns,
        preprocessor_params=preprocessor_params,
    )

    for fold, (tr_idx, va_idx, te_idx) in zip(fold_nums_all, fold_indices):
        print(f"Fold {fold}")
        print(f"Train size: {len(tr_idx)}")
        print(f"Validation size: {len(va_idx)}")
        print(f"Test size: {len(te_idx)}")
        cv_results["fold_nums"].append(fold)
        model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pt")

        if write_to_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                raise ImportError("TensorBoard is not available. Please install tensorboard.")
            tb_path = os.path.join(tensorboard_dir, f"lag_{lag}", f"fold_{fold}")
            writer = SummaryWriter(log_dir=tb_path)

        if training_params.normalize_targets:
            Y_train = Y[tr_idx]
            Y_val = Y[va_idx]
            Y_test = Y[te_idx]
            y_mean = Y_train.mean(dim=0, keepdim=True)
            y_std = Y_train.std(dim=0, keepdim=True)
            y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
            Y_train_norm = (Y_train - y_mean) / y_std
            Y_val_norm = (Y_val - y_mean) / y_std
            Y_test_norm = (Y_test - y_mean) / y_std
        else:
            Y_train_norm = Y[tr_idx]
            Y_val_norm = Y[va_idx]
            Y_test_norm = Y[te_idx]

        extra_train_inputs = data_utils.df_columns_to_tensors(
            data_df, task_config.task_specific_config.input_fields, tr_idx
        )
        extra_val_inputs = data_utils.df_columns_to_tensors(
            data_df, task_config.task_specific_config.input_fields, va_idx
        )
        extra_test_inputs = data_utils.df_columns_to_tensors(
            data_df, task_config.task_specific_config.input_fields, te_idx
        )

        datasets = {
            "train": LazyWindowDataset(
                data_df.iloc[tr_idx].start.to_numpy(),
                extra_train_inputs,
                Y_train_norm,
            ),
            "val": LazyWindowDataset(
                data_df.iloc[va_idx].start.to_numpy(),
                extra_val_inputs,
                Y_val_norm,
            ),
            "test": LazyWindowDataset(
                data_df.iloc[te_idx].start.to_numpy(),
                extra_test_inputs,
                Y_test_norm,
            ),
        }
        loaders = {
            phase: DataLoader(
                ds,
                batch_size=training_params.batch_size,
                shuffle=(phase == "train"),
                collate_fn=collate_fn,
            )
            for phase, ds in datasets.items()
        }

        model = build_model_from_spec(model_spec, lag=lag, fold=fold).to(device)

        if training_params.optimizer == "MuAdamW":
            print("Using MuAdamW optimizer")
            optimizer = MuAdamW(
                model.parameters(),
                lr=float(training_params.learning_rate),
                weight_decay=float(training_params.weight_decay),
            )
        else:
            print("Using AdamW optimizer")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=float(training_params.learning_rate),
                weight_decay=float(training_params.weight_decay),
            )

        scheduler = create_lr_scheduler(optimizer, training_params)
        best_val, patience = setup_early_stopping_state(training_params)
        best_epoch = 0

        history = {
            f"{phase}_{name}": [] for phase in ("train", "val") for name in metric_names
        }
        if "cross_entropy" in metric_names:
            for phase in ("train", "val"):
                history[f"{phase}_perplexity"] = []
        history["train_loss"] = []
        history["val_loss"] = []
        history["num_epochs"] = None

        loop = tqdm(range(training_params.epochs), desc=f"Lag {lag}, Fold {fold}")
        for epoch in loop:
            train_mets = run_epoch(model, loaders["train"], optimizer)
            val_mets = run_epoch(model, loaders["val"])

            for name, val in train_mets.items():
                history[f"train_{name}"].append(val)
            for name, val in val_mets.items():
                history[f"val_{name}"].append(val)

            if write_to_tensorboard:
                log_metrics_to_tensorboard(writer, train_mets, "model", "train", epoch)
                log_metrics_to_tensorboard(writer, val_mets, "model", "val", epoch)

            cur = val_mets[training_params.early_stopping_metric]
            if should_update_best(cur, best_val, training_params.smaller_is_better):
                best_val = cur
                best_epoch = epoch
                if hasattr(model, "save_checkpoint") and callable(getattr(model, "save_checkpoint")):
                    model.save_checkpoint(model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                patience = 0
            else:
                patience += 1
                if patience >= training_params.early_stopping_patience:
                    break

            if scheduler is not None and not hasattr(scheduler, "step_per_batch"):
                scheduler.step(cur)

            if write_to_tensorboard:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("learning_rate", current_lr, epoch)

            loop.set_postfix(
                {
                    training_params.early_stopping_metric: f"{best_val:.4f}",
                    **{f"train_{name}": val for name, val in train_mets.items()},
                    **{f"val_{name}": val for name, val in val_mets.items()},
                }
            )

        history["num_epochs"] = best_epoch + 1

        if hasattr(model, "load_checkpoint") and callable(getattr(model, "load_checkpoint")):
            model.load_checkpoint(model_path)
        else:
            model.load_state_dict(torch.load(model_path))
        test_mets = run_epoch(model, loaders["test"])

        for name in metric_names:
            if name != "confusion_matrix":
                cv_results[f"train_{name}"].append(history[f"train_{name}"][best_epoch])
                cv_results[f"val_{name}"].append(history[f"val_{name}"][best_epoch])
                cv_results[f"test_{name}"].append(test_mets[name])
        cv_results["num_epochs"].append(history["num_epochs"])

        if write_to_tensorboard:
            log_metrics_to_tensorboard(writer, test_mets, "model", "test", fold)
            writer.close()

        models.append(model)
        histories.append(history)

    print("\n" + "=" * 60)
    print("MAIN MODEL CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for phase in ("train", "val", "test"):
        for name in metric_names:
            if name == "confusion_matrix":
                continue
            vals = cv_results[f"{phase}_{name}"]
            print(f"--- Individual Folds ({phase}_{name}) ---")
            fold_nums = cv_results.get("fold_nums", list(range(1, len(vals) + 1)))
            for i, val in enumerate(vals):
                fold_num = fold_nums[i]
                print(f"Fold {fold_num}: {val:.4f}")
            print(f"Mean {phase} {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")

    if "cross_entropy" in metric_names:
        for phase in ("train", "val", "test"):
            ce_vals = cv_results[f"{phase}_cross_entropy"]
            ppl_vals = np.exp(ce_vals)
            print(
                f"Mean {phase} perplexity: {np.mean(ppl_vals):.4f} ± {np.std(ppl_vals):.4f}"
            )

    return models, histories, cv_results


def run_training_over_lags_lazy_stft(
    lags,
    raws,
    task_df: pd.DataFrame,
    preprocessing_fns,
    model_spec: ModelSpec,
    task_name: str,
    training_params: TrainingParams,
    task_config: TaskConfig,
    output_dir="results/",
    checkpoint_dir="checkpoints/",
    write_to_tensorboard=False,
    tensorboard_dir="event_log",
):
    data_params = task_config.data_params
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, "lag_performance.csv")
    if os.path.exists(filename):
        roc_df = pd.read_csv(filename)
        already_read_lags = roc_df["lags"].tolist()
        existing_df = roc_df
    else:
        already_read_lags = []
        existing_df = pd.DataFrame()

    for lag in lags:
        if lag in already_read_lags:
            print(f"Lag {lag} already done, skipping...")
            continue

        print("=" * 60)
        print("running experimental lazy lag:", lag)
        print("=" * 60)

        data_df, targets = _filter_valid_rows(raws, task_df, lag, data_params.window_width)
        targets_tensor = torch.FloatTensor(np.stack(targets) if getattr(targets, "dtype", None) == object else targets)
        print(f"lazy data_df rows: {len(data_df)}")

        _, _, cv_results = train_decoding_model_lazy(
            raws,
            targets_tensor,
            data_df,
            preprocessing_fns,
            data_params.preprocessor_params,
            data_params.window_width,
            model_spec,
            task_name,
            task_config,
            lag,
            training_params=training_params,
            checkpoint_dir=os.path.join(checkpoint_dir, f"lag_{lag}"),
            write_to_tensorboard=write_to_tensorboard,
            tensorboard_dir=tensorboard_dir,
        )

        lag_metrics = {"lags": lag}
        fold_nums = cv_results.get("fold_nums", None)
        for metric, values in cv_results.items():
            if metric == "fold_nums":
                continue
            if len(values) > 0:
                lag_metrics[f"{metric}_mean"] = np.mean(values)
                lag_metrics[f"{metric}_std"] = np.std(values)
                for i, val in enumerate(values):
                    fold_num = (
                        fold_nums[i]
                        if (fold_nums is not None and i < len(fold_nums))
                        else (i + 1)
                    )
                    lag_metrics[f"{metric}_fold_{fold_num}"] = val
            else:
                lag_metrics[f"{metric}_mean"] = np.nan
                lag_metrics[f"{metric}_std"] = np.nan

        existing_df = pd.concat(
            [existing_df, pd.DataFrame([lag_metrics])], ignore_index=True
        )
        existing_df.to_csv(filename, index=False)
