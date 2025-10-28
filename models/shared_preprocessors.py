from typing import Optional

import numpy as np

import core.registry as registry


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
