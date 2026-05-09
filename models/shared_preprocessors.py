import copy
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import core.registry as registry
from utils.dataset import _apply_preprocessing

_DEFAULT_PREPROCESSOR_CACHE_DIR = ".cache/preprocessors"


@registry.register_data_preprocessor()
def window_average_neural_data(data, preprocessor_params):
    num_average_samples = preprocessor_params["num_average_samples"]
    if preprocessor_params and preprocessor_params.get("num_average_samples"):
        data = window_data(data, num_average_samples)

    data = data.mean(-1)

    return data


@registry.register_data_preprocessor("window_rms")
def window_rms_preprocessor(
    data: np.ndarray, preprocessor_params: Optional[dict] = None
) -> np.ndarray:
    """Reduce each neural window to a root-mean-square amplitude."""

    if data.ndim != 3:
        raise ValueError(
            "window_rms_preprocessor expects data with shape (examples, channels, samples)."
        )

    squared = np.square(data, dtype=np.float64)
    if preprocessor_params and preprocessor_params.get("num_average_samples"):
        squared = window_data(squared, preprocessor_params["num_average_samples"])
    mean_sq = squared.mean(axis=-1)
    rms = np.sqrt(np.maximum(mean_sq, 0.0))
    return rms.astype(np.float32, copy=False)


@registry.register_data_preprocessor("log_transform")
def log_transform_preprocessor(
    data: np.ndarray, preprocessor_params: Optional[dict] = None
) -> np.ndarray:
    """Apply a logarithmic compression to neural amplitudes."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(
            "log_transform_preprocessor expects data with at least two dimensions."
        )

    params = preprocessor_params if preprocessor_params is not None else {}
    epsilon_param = params.get("epsilon")
    if epsilon_param is None:
        epsilon_scale = float(params.get("epsilon_scale", 1e-6))
        epsilon_floor = float(params.get("epsilon_floor", 1e-12))
        max_val = float(np.max(arr)) if arr.size else 0.0
        epsilon = max(epsilon_floor, max_val * epsilon_scale)
        if preprocessor_params is not None:
            preprocessor_params["epsilon"] = epsilon
    else:
        epsilon = float(epsilon_param)

    if epsilon <= 0:
        raise ValueError("epsilon must be positive for log_transform_preprocessor.")

    clip_min = float(params.get("clip_min", 0.0))
    scale = float(params.get("scale", 1.0))
    base = params.get("log_base", 10.0)

    arr64 = arr.astype(np.float64, copy=False)
    clipped = np.clip(arr64, clip_min, None)
    shifted = clipped + epsilon

    if isinstance(base, str):
        base_lower = base.lower()
        if base_lower == "e":
            log_values = np.log(shifted)
        elif base_lower == "10":
            log_values = np.log10(shifted)
        else:
            raise ValueError("log_base string must be 'e' or '10'.")
    else:
        base = float(base)
        if base <= 0 or np.isclose(base, 1.0):
            raise ValueError("log_base must be > 0 and != 1.")
        log_values = np.log(shifted) / np.log(base)

    if scale != 1.0:
        log_values *= scale

    return log_values.astype(np.float32, copy=False)


@registry.register_data_preprocessor("zscore")
def zscore_preprocessor(
    data: np.ndarray, preprocessor_params: Optional[dict] = None
) -> np.ndarray:
    """Standardize each channel independently across all observations."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(
            "zscore_preprocessor expects data with at least two dimensions."
        )

    params = preprocessor_params if preprocessor_params is not None else {}
    epsilon = float(params.get("epsilon", 1e-6))

    channel_axis = 1 if arr.ndim >= 2 else 0
    channel_first = np.moveaxis(arr, channel_axis, 0)
    flat = channel_first.reshape(channel_first.shape[0], -1)
    flat64 = flat.astype(np.float64, copy=False)

    means = params.get("channel_means")
    stds = params.get("channel_stds")
    if means is not None and stds is not None:
        means = np.asarray(means, dtype=np.float64).reshape(-1, 1)
        stds = np.asarray(stds, dtype=np.float64).reshape(-1, 1)
        if means.shape[0] != flat64.shape[0] or stds.shape[0] != flat64.shape[0]:
            raise ValueError(
                "channel_means and channel_stds must match the number of channels in data."
            )
    else:
        means = np.nanmean(flat64, axis=1, keepdims=True)
        stds = np.nanstd(flat64, axis=1, keepdims=True)
        if preprocessor_params is not None:
            preprocessor_params["channel_means"] = means.astype(np.float32).reshape(-1)
            preprocessor_params["channel_stds"] = stds.astype(np.float32).reshape(-1)

    stds = np.where(stds < epsilon, epsilon, stds)
    standardized = (flat64 - means) / stds
    standardized = standardized.reshape(channel_first.shape)
    standardized = np.moveaxis(standardized, 0, channel_axis)
    return standardized.astype(np.float32, copy=False)


@registry.register_data_preprocessor("disk_cache_preprocessor")
def disk_cache_preprocessor(
    data: np.ndarray,
    preprocessor_params: Optional[dict] = None,
    **context,
) -> np.ndarray:
    """Cache the output of one wrapped preprocessor or an ordered preprocessor chain."""

    params = preprocessor_params if preprocessor_params is not None else {}
    if not isinstance(params, dict):
        raise ValueError("disk_cache_preprocessor params must be a dictionary.")

    names, wrapped_params = _resolve_wrapped_preprocessor_config(params)
    wrapped_fns = []
    for name in names:
        if name not in registry.data_preprocessor_registry:
            raise ValueError(f"Unknown wrapped preprocessor '{name}'.")
        wrapped_fns.append(registry.data_preprocessor_registry[name])

    cache_dir = Path(params.get("cache_dir", _DEFAULT_PREPROCESSOR_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_identity = _build_cache_identity(
        names=names,
        wrapped_fns=wrapped_fns,
        wrapped_params=wrapped_params,
        mode=params.get("mode", "normal"),
        context=context,
    )
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    cache_path = cache_dir / f"{cache_key}.npz"

    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["data"].astype(np.float32, copy=False)

    output = _apply_preprocessing(
        data,
        wrapped_fns,
        copy.deepcopy(wrapped_params),
        **context,
    )
    metadata = json.dumps(cache_identity, sort_keys=True)
    _atomic_save_npz(cache_path, data=np.asarray(output), metadata=metadata)
    return output


def _resolve_wrapped_preprocessor_config(params: dict) -> tuple[list[str], list]:
    names = params.get("base_preprocessing_fn_name")
    if names is None:
        raise ValueError("disk_cache_preprocessor requires base_preprocessing_fn_name.")

    if isinstance(names, str):
        names_list = [names]
        wrapped_params = [params.get("base_preprocessor_params")]
    elif isinstance(names, list):
        names_list = names
        raw_params = params.get("base_preprocessor_params")
        if isinstance(raw_params, list):
            wrapped_params = [
                raw_params[i] if i < len(raw_params) else None
                for i in range(len(names_list))
            ]
        else:
            wrapped_params = [raw_params for _ in names_list]
    else:
        raise ValueError(
            "base_preprocessing_fn_name must be a string or list of strings."
        )

    if not all(isinstance(name, str) for name in names_list):
        raise ValueError("base_preprocessing_fn_name entries must be strings.")

    return names_list, wrapped_params


def _build_cache_identity(
    names: list[str],
    wrapped_fns: list,
    wrapped_params: list,
    mode: str,
    context: dict,
) -> dict:
    selected_rows = context.get("selected_rows_df", context.get("selected_rows"))
    starts = []
    if selected_rows is not None and "start" in selected_rows:
        starts = _normalize_for_json(selected_rows["start"].to_list())

    return {
        "version": 1,
        "mode": mode,
        "pipeline": [
            {
                "name": name,
                "source_hash": _source_hash(fn),
                "params": _normalize_for_json(params),
            }
            for name, fn, params in zip(names, wrapped_fns, wrapped_params)
        ],
        "lag": _normalize_for_json(context.get("lag")),
        "selected_rows_start": starts,
        "subject_channel_names": _normalize_for_json(
            context.get("subject_channel_names")
        ),
        "subject_channel_counts": _normalize_for_json(
            context.get("subject_channel_counts")
        ),
    }


def _source_hash(fn) -> Optional[str]:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalize_for_json(value):
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_json(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_save_npz(cache_path: Path, **arrays) -> None:
    temp_file = tempfile.NamedTemporaryFile(
        dir=cache_path.parent,
        prefix=f".{cache_path.stem}.",
        suffix=".tmp.npz",
        delete=False,
    )
    temp_name = temp_file.name
    temp_file.close()
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, cache_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def window_data(data: np.ndarray, num_average_samples: int) -> np.ndarray:
    """Trim data to the largest size divisible by num_average_samples and reshape into windows."""
    if data.ndim < 3:
        raise ValueError("window_data expects data with at least three dimensions.")

    num_samples = data.shape[2]
    num_to_keep = (num_samples // num_average_samples) * num_average_samples
    data_trimmed = data[:, :, :num_to_keep]
    return data_trimmed.reshape(
        data_trimmed.shape[0],
        data_trimmed.shape[1],
        -1,
        num_average_samples,
    )
