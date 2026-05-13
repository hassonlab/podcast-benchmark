import time

import numpy as np
import torch
from tqdm import tqdm

from core import registry
from core.config import ModelSpec, dict_to_config
from utils import data_utils
from utils.model_utils import build_model_from_spec


_FEATURE_CACHE_OMITTED_PARAM_KEYS = {
    "output_dim",
    "embedding_dim",
    "output_activation",
    "dropout",
    "mlp_layer_sizes",
    "freeze_foundation",
    "frozen_upstream",
    "num_frozen_layers",
}


def _as_model_spec(spec):
    if isinstance(spec, ModelSpec):
        return spec
    if isinstance(spec, dict):
        return dict_to_config(spec, ModelSpec)
    raise TypeError(
        f"foundation_model_spec must be a dict or ModelSpec, got {type(spec)}"
    )


def _resolve_device(params):
    configured = params.get("device")
    if configured:
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_input_tensors(selected_rows, input_fields, device):
    if selected_rows is None or not input_fields:
        return {}
    indices = np.arange(len(selected_rows))
    inputs = data_utils.df_columns_to_tensors(selected_rows, input_fields, indices)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def _batch_kwargs(kwargs, start, stop):
    out = {}
    for key, value in kwargs.items():
        if torch.is_tensor(value) and value.shape[0] >= stop:
            out[key] = value[start:stop]
        else:
            out[key] = value
    return out


def _split_coordinate_kwargs(kwargs, subject_channel_counts):
    chunks = {}
    for key in ("xyz_id", "lip_coords"):
        value = kwargs.get(key)
        if torch.is_tensor(value):
            chunks[key] = torch.split(value, subject_channel_counts, dim=1)
    return chunks


def _encode_normal(model, data_tensor, kwargs, batch_size, device):
    features = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(data_tensor), batch_size),
            desc="Caching foundation features",
        ):
            stop = min(start + batch_size, len(data_tensor))
            batch = data_tensor[start:stop].to(device)
            batch_kwargs = _batch_kwargs(kwargs, start, stop)
            features.append(model.encode_features(batch, **batch_kwargs).detach().cpu())
    return torch.cat(features, dim=0)


def _encode_per_subject_concat(
    model,
    data_tensor,
    kwargs,
    subject_channel_counts,
    batch_size,
    device,
):
    if not subject_channel_counts or len(subject_channel_counts) <= 1:
        raise ValueError(
            "foundation_feature_cache per_subject_feature_concat mode requires "
            f"multiple subject_channel_counts, got {subject_channel_counts}."
        )

    features = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(data_tensor), batch_size),
            desc="Caching per-subject foundation features",
        ):
            stop = min(start + batch_size, len(data_tensor))
            batch = data_tensor[start:stop].to(device)
            batch_kwargs = _batch_kwargs(kwargs, start, stop)
            coord_chunks = _split_coordinate_kwargs(
                batch_kwargs, subject_channel_counts
            )
            subject_embeddings = []
            for subject_idx, chunk in enumerate(
                torch.split(batch, subject_channel_counts, dim=1)
            ):
                subject_kwargs = {
                    key: chunks[subject_idx] for key, chunks in coord_chunks.items()
                }
                subject_embeddings.append(
                    model.encode_features(chunk, **subject_kwargs).detach().cpu()
                )
            features.append(torch.cat(subject_embeddings, dim=-1))
    return torch.cat(features, dim=0)


@registry.register_data_preprocessor("foundation_feature_cache")
def foundation_feature_cache(
    data,
    preprocessor_params,
    selected_rows=None,
    selected_rows_df=None,
    subject_channel_counts=None,
    **kwargs,
):
    """Cache frozen foundation-model embeddings as the neural tensor."""
    params = dict(preprocessor_params or {})
    foundation_spec = _as_model_spec(params["foundation_model_spec"])
    mode = params.get("mode", "normal")
    batch_size = int(params.get("batch_size", params.get("cache_batch_size", 32)))
    input_fields = params.get("input_fields") or params.get("model_input_fields") or []
    rows = selected_rows if selected_rows is not None else selected_rows_df
    if rows is not None:
        input_fields = list(input_fields)
        for field in ("xyz_id", "lip_coords"):
            if field in rows.columns and field not in input_fields:
                input_fields.append(field)
    device = _resolve_device(params)

    model_inputs = _build_input_tensors(rows, input_fields, device)
    data_tensor = torch.as_tensor(data, dtype=torch.float32)

    model = build_model_from_spec(foundation_spec, lag=params.get("lag"), fold=1).to(
        device
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "encode_features"):
        raise NotImplementedError(
            "foundation_feature_cache requires the foundation model to implement "
            f"encode_features(...). Got model: {model.__class__.__name__}"
        )

    start_time = time.time()
    if mode == "normal":
        features = _encode_normal(model, data_tensor, model_inputs, batch_size, device)
    elif mode == "per_subject_feature_concat":
        features = _encode_per_subject_concat(
            model,
            data_tensor,
            model_inputs,
            subject_channel_counts,
            batch_size,
            device,
        )
    else:
        raise ValueError(
            "foundation_feature_cache mode must be 'normal' or "
            f"'per_subject_feature_concat', got {mode!r}."
        )

    print(
        "Cached foundation features: "
        f"{tuple(data_tensor.shape)} -> {tuple(features.shape)} "
        f"in {time.time() - start_time:.2f}s"
    )
    return features.numpy().astype(np.float32, copy=False)


def _iter_preprocessor_entries(data_params):
    names = data_params.preprocessing_fn_name
    params = data_params.preprocessor_params
    if names is None:
        return []
    if not isinstance(names, list):
        names = [names]
        params = [params]
    elif params is None:
        params = [None] * len(names)
    elif not isinstance(params, list):
        params = [params]
    return list(zip(names, params))


def _set_preprocessor_entries(data_params, entries):
    data_params.preprocessing_fn_name = [name for name, _ in entries]
    data_params.preprocessor_params = [params for _, params in entries]


def _iter_wrapped_preprocessor_entries(wrapper_params):
    names = wrapper_params.get("base_preprocessing_fn_name")
    params = wrapper_params.get("base_preprocessor_params")
    if names is None:
        return []
    if not isinstance(names, list):
        names = [names]
        params = [params]
    elif params is None:
        params = [None] * len(names)
    elif not isinstance(params, list):
        params = [params]
    return list(zip(names, params))


def _set_wrapped_preprocessor_entries(wrapper_params, entries):
    wrapper_params["base_preprocessing_fn_name"] = [name for name, _ in entries]
    wrapper_params["base_preprocessor_params"] = [params for _, params in entries]


def _configure_foundation_feature_params(
    experiment_config, data_params, params, raws, task_df
):
    params = dict(params or {})
    foundation_spec = _as_model_spec(params["foundation_model_spec"])
    foundation_spec.feature_cache = True
    foundation_spec.params["feature_cache"] = True
    foundation_setters = params.get("foundation_config_setter_name")

    if foundation_setters:
        original_model_spec = experiment_config.model_spec
        experiment_config.model_spec = foundation_spec
        try:
            setter_names = (
                foundation_setters
                if isinstance(foundation_setters, list)
                else [foundation_setters]
            )
            for setter_name in setter_names:
                setter = registry.config_setter_registry[setter_name]
                experiment_config = setter(experiment_config, raws, task_df)
        finally:
            foundation_spec = experiment_config.model_spec
            experiment_config.model_spec = original_model_spec

    foundation_spec.feature_cache = True
    foundation_spec.params["feature_cache"] = True
    for key in _FEATURE_CACHE_OMITTED_PARAM_KEYS:
        foundation_spec.params.pop(key, None)

    input_fields = list(
        experiment_config.task_config.task_specific_config.input_fields or []
    )
    params["input_fields"] = params.get("input_fields", input_fields)
    params["foundation_model_spec"] = foundation_spec
    foundation_info = registry.model_constructor_registry.get(
        foundation_spec.constructor_name, {}
    )
    getter_name = foundation_spec.model_data_getter or foundation_info.get(
        "required_data_getter"
    )
    if getter_name:
        experiment_config.model_spec.model_data_getter = getter_name
        experiment_config.model_spec.params.setdefault(
            "data_root", data_params.data_root
        )
    return experiment_config, params


def _configure_disk_cache_wrapper_params(
    experiment_config, data_params, params, raws, task_df
):
    wrapper_params = dict(params or {})
    wrapped_entries = _iter_wrapped_preprocessor_entries(wrapper_params)
    if not wrapped_entries:
        return experiment_config, wrapper_params

    updated_entries = []
    for wrapped_name, wrapped_params in wrapped_entries:
        if wrapped_name == "foundation_feature_cache":
            experiment_config, wrapped_params = _configure_foundation_feature_params(
                experiment_config,
                data_params,
                wrapped_params,
                raws,
                task_df,
            )
        updated_entries.append((wrapped_name, wrapped_params))
    _set_wrapped_preprocessor_entries(wrapper_params, updated_entries)
    return experiment_config, wrapper_params


@registry.register_config_setter("foundation_feature_cache")
def set_foundation_feature_cache_config(experiment_config, raws, task_df):
    """Configure nested frozen foundation spec before generic feature caching."""
    data_params = experiment_config.task_config.data_params
    entries = _iter_preprocessor_entries(data_params)
    updated_entries = []
    for name, params in entries:
        if name == "foundation_feature_cache":
            experiment_config, params = _configure_foundation_feature_params(
                experiment_config,
                data_params,
                params,
                raws,
                task_df,
            )
        elif name == "disk_cache_preprocessor":
            experiment_config, params = _configure_disk_cache_wrapper_params(
                experiment_config,
                data_params,
                params,
                raws,
                task_df,
            )
        updated_entries.append((name, params))

    _set_preprocessor_entries(data_params, updated_entries)
    return experiment_config
