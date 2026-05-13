import json
from pathlib import Path

import yaml


FOUNDATION_CONFIG_ROOT = Path("configs/foundation_models")


def _foundation_feature_cache_params(config):
    data_params = config.get("task_config", {}).get("data_params", {})
    names = data_params.get("preprocessing_fn_name") or []
    params = data_params.get("preprocessor_params") or []
    if isinstance(names, str):
        names = [names]
        params = [params]

    for name, preprocessor_params in zip(names, params):
        assert name == "disk_cache_preprocessor"
        wrapped_names = preprocessor_params["base_preprocessing_fn_name"]
        wrapped_params = preprocessor_params["base_preprocessor_params"]
        assert "foundation_feature_cache" in wrapped_names
        assert len(wrapped_names) == len(wrapped_params)
        for wrapped_name, wrapped_param in zip(wrapped_names, wrapped_params):
            if wrapped_name == "foundation_feature_cache":
                yield wrapped_param


def _disk_cache_preprocessor_params(config):
    data_params = config.get("task_config", {}).get("data_params", {})
    names = data_params.get("preprocessing_fn_name") or []
    params = data_params.get("preprocessor_params") or []
    if isinstance(names, str):
        names = [names]
        params = [params]

    for name, preprocessor_params in zip(names, params):
        if name == "disk_cache_preprocessor":
            yield preprocessor_params


def test_foundation_feature_cache_configs_are_family_stable_except_mode():
    omitted_nested_keys = {
        "output_dim",
        "embedding_dim",
        "output_activation",
        "dropout",
        "mlp_layer_sizes",
        "freeze_foundation",
        "frozen_upstream",
        "num_frozen_layers",
    }

    for family in ("brainbert", "popt", "diver"):
        identities = {}
        for config_path in sorted((FOUNDATION_CONFIG_ROOT / family).glob("**/*.yml")):
            config = yaml.safe_load(config_path.read_text())
            for params in _foundation_feature_cache_params(config):
                params_for_identity = dict(params)
                params_for_identity.pop("mode", None)
                spec = params_for_identity["foundation_model_spec"]
                spec_params = spec["params"]

                assert params_for_identity["batch_size"] == 4
                assert spec["feature_cache"] is True
                assert spec_params["feature_cache"] is True
                assert omitted_nested_keys.isdisjoint(spec_params)

                identity = json.dumps(params_for_identity, sort_keys=True)
                identities.setdefault(identity, []).append(str(config_path))

        assert len(identities) == 1, (
            f"{family} foundation_feature_cache params differ across configs: "
            f"{list(identities.values())[:3]}"
        )


def test_foundation_disk_cache_updates_missing_rows_except_volume_level():
    for config_path in sorted(FOUNDATION_CONFIG_ROOT.glob("**/*.yml")):
        config = yaml.safe_load(config_path.read_text())
        is_volume_level = (
            "volume_level" in config_path.parts
            or config.get("task_config", {}).get("task_name")
            == "volume_level_decoding_task"
        )

        for params in _disk_cache_preprocessor_params(config):
            if is_volume_level:
                assert "update_cache_with_missing" not in params
            else:
                assert params["update_cache_with_missing"] is True
