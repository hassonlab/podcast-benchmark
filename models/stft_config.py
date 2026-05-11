DEFAULT_STFT_CONFIG = {
    "freq_channel_cutoff": 40,
    "nperseg": 400,
    "noverlap": 350,
    "normalizing": "zscore",
}


def configure_explicit_stft_preprocessor(
    data_params,
    *,
    sample_rate: int,
    model_name: str,
    extra_defaults: dict | None = None,
) -> dict:
    """Fill defaults for an explicitly configured STFT preprocessor."""

    entry = _find_stft_preprocessor_params(data_params)
    if entry is None:
        raise ValueError(
            f"{model_name} requires an explicit 'stft_preprocessing' entry in "
            "data_params.preprocessing_fn_name or inside a disk_cache_preprocessor "
            "base_preprocessing_fn_name pipeline."
        )

    params, setter = entry
    stft_config = dict(params or {})
    for key, value in DEFAULT_STFT_CONFIG.items():
        stft_config.setdefault(key, value)
    stft_config.setdefault("fs", int(sample_rate))
    for key, value in (extra_defaults or {}).items():
        stft_config.setdefault(key, value)
    setter(stft_config)
    return stft_config


def _find_stft_preprocessor_params(data_params):
    for idx, (name, params) in enumerate(_iter_entries(data_params)):
        if name == "stft_preprocessing":
            return params, lambda updated, idx=idx: _set_entry_param(
                data_params, idx, updated
            )
        if name == "disk_cache_preprocessor" and isinstance(params, dict):
            wrapped_entry = _find_wrapped_stft_preprocessor_params(params)
            if wrapped_entry is not None:
                return wrapped_entry
    return None


def _find_wrapped_stft_preprocessor_params(wrapper_params: dict):
    entries = _iter_wrapped_entries(wrapper_params)
    for idx, (name, params) in enumerate(entries):
        if name == "stft_preprocessing":
            return params, lambda updated, idx=idx: _set_wrapped_entry_param(
                wrapper_params, idx, updated
            )
    return None


def _iter_entries(data_params):
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


def _set_entry_param(data_params, idx: int, updated: dict) -> None:
    names = data_params.preprocessing_fn_name
    params = data_params.preprocessor_params
    if not isinstance(names, list):
        data_params.preprocessing_fn_name = names
        data_params.preprocessor_params = updated
        return
    if params is None:
        params = [None] * len(names)
    elif not isinstance(params, list):
        params = [params]
    while len(params) <= idx:
        params.append(None)
    params[idx] = updated
    data_params.preprocessor_params = params


def _iter_wrapped_entries(wrapper_params: dict):
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


def _set_wrapped_entry_param(wrapper_params: dict, idx: int, updated: dict) -> None:
    names = wrapper_params.get("base_preprocessing_fn_name")
    params = wrapper_params.get("base_preprocessor_params")
    if not isinstance(names, list):
        wrapper_params["base_preprocessor_params"] = updated
        return
    if params is None:
        params = [None] * len(names)
    elif not isinstance(params, list):
        params = [params]
    while len(params) <= idx:
        params.append(None)
    params[idx] = updated
    wrapper_params["base_preprocessor_params"] = params
