#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG_ROOT = REPO_ROOT / "configs" / "foundation_models"

ENCODER_CONSTRUCTOR_BY_MODEL = {
    "brainbert_finetune": "brainbert_encoder",
    "diver_finetune": "diver_encoder",
    "popt_finetune": "popt_encoder",
}


def _is_cached_encoder_wrapper(model_spec: dict, encoder_constructor_name: str) -> bool:
    encoder_spec = model_spec.get("sub_models", {}).get("encoder_model")
    if not isinstance(encoder_spec, dict):
        return False
    if encoder_spec.get("constructor_name") != "caching_model":
        return False
    inner_spec = encoder_spec.get("sub_models", {}).get("inner_model")
    return (
        isinstance(inner_spec, dict)
        and inner_spec.get("constructor_name") == encoder_constructor_name
    )


def _wrap_model_spec_with_cached_encoder(
    model_spec: dict, encoder_constructor_name: str
) -> bool:
    if _is_cached_encoder_wrapper(model_spec, encoder_constructor_name):
        return False

    sub_models = model_spec.setdefault("sub_models", {})
    if "encoder_model" in sub_models:
        raise ValueError(
            "Refusing to overwrite existing encoder_model sub_model while migrating to caching."
        )

    sub_models["encoder_model"] = {
        "constructor_name": "caching_model",
        "params": {},
        "sub_models": {
            "inner_model": {
                "constructor_name": encoder_constructor_name,
                "params": {},
                "sub_models": {},
            }
        },
    }
    return True


def migrate_config(path: Path, dry_run: bool) -> bool:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected YAML root in {path}")

    model_spec = config.get("model_spec")
    if not isinstance(model_spec, dict):
        return False

    constructor_name = model_spec.get("constructor_name")
    encoder_constructor_name = ENCODER_CONSTRUCTOR_BY_MODEL.get(constructor_name)
    if encoder_constructor_name is None:
        return False

    if model_spec.get("params", {}).get("freeze_foundation") is not True:
        return False

    changed = _wrap_model_spec_with_cached_encoder(
        model_spec, encoder_constructor_name
    )
    if changed and not dry_run:
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap frozen foundation-model finetune configs with the new caching_model "
            "encoder path."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which configs would change without writing them.",
    )
    args = parser.parse_args()

    targets = sorted(FOUNDATION_CONFIG_ROOT.glob("*/*/*.yml"))
    changed_paths: list[Path] = []
    for path in targets:
        if migrate_config(path, dry_run=args.dry_run):
            changed_paths.append(path)

    print(
        f"Processed {len(targets)} config files. "
        f"{'Would update' if args.dry_run else 'Updated'} {len(changed_paths)} files."
    )
    for path in changed_paths[:20]:
        print(path)
    if len(changed_paths) > 20:
        print(f"... and {len(changed_paths) - 20} more")


if __name__ == "__main__":
    main()
