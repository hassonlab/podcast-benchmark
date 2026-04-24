#!/usr/bin/env python3
"""Add the caching_model encoder wrapper to foundation-model configs.

Handles two config shapes:

1. Simple finetune (top-level is brainbert_finetune / diver_finetune / popt_finetune,
   freeze_foundation=True):

     model_spec:
       constructor_name: brainbert_finetune
       params:
         freeze_foundation: true
       sub_models:
         encoder_model:           ← added / verified
           constructor_name: caching_model
           sub_models:
             inner_model:
               constructor_name: brainbert_encoder

2. LLM-decoding (top-level is gpt2_brain, encoder_model is a finetune model):

     model_spec:
       constructor_name: gpt2_brain
       sub_models:
         encoder_model:
           constructor_name: brainbert_finetune
           params:
             freeze_foundation: true   ← set to true
           sub_models:
             encoder_model:            ← added / verified
               constructor_name: caching_model
               sub_models:
                 inner_model:
                   constructor_name: brainbert_encoder

For each case three sub-cases apply:
  A. encoder_model already wrapped with caching_model → skip (idempotent).
  B. No encoder_model sub_model yet → add the full caching wrapper.
  C. encoder_model exists with the raw encoder constructor → wrap it.
  D. Anything else → skip with a warning (unknown pattern, don't touch).
"""

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


def _caching_wrapper(encoder_constructor: str) -> dict:
    return {
        "constructor_name": "caching_model",
        "params": {},
        "sub_models": {
            "inner_model": {
                "constructor_name": encoder_constructor,
                "params": {},
                "sub_models": {},
            }
        },
    }


def _apply_caching_to_finetune_spec(
    finetune_spec: dict,
    encoder_constructor: str,
    path: Path,
    verbose: bool,
) -> bool:
    """Ensure finetune_spec has encoder_model wrapped with caching_model.

    Returns True if a change was made.
    """
    sub_models = finetune_spec.setdefault("sub_models", {})
    encoder_spec = sub_models.get("encoder_model")

    if encoder_spec is None:
        sub_models["encoder_model"] = _caching_wrapper(encoder_constructor)
        return True

    if not isinstance(encoder_spec, dict):
        return False

    existing = encoder_spec.get("constructor_name")
    if existing == "caching_model":
        if verbose:
            print(f"  [skip] already cached: {path}")
        return False
    if existing == encoder_constructor:
        sub_models["encoder_model"] = _caching_wrapper(encoder_constructor)
        return True

    print(
        f"  [warn] unexpected encoder_model constructor '{existing}' in {path} — skipping"
    )
    return False


def _migrate_simple_finetune(model_spec: dict, path: Path, verbose: bool) -> bool:
    """Handle top-level brainbert/diver/popt_finetune with freeze_foundation=True."""
    outer_constructor = model_spec.get("constructor_name")
    encoder_constructor = ENCODER_CONSTRUCTOR_BY_MODEL.get(outer_constructor)
    if encoder_constructor is None:
        return False
    if model_spec.get("params", {}).get("freeze_foundation") is not True:
        return False
    return _apply_caching_to_finetune_spec(model_spec, encoder_constructor, path, verbose)


def _migrate_llm_decoding(model_spec: dict, path: Path, verbose: bool) -> bool:
    """Handle gpt2_brain configs: add caching inside the nested finetune encoder_model."""
    if model_spec.get("constructor_name") != "gpt2_brain":
        return False

    sub_models = model_spec.get("sub_models") or {}
    finetune_spec = sub_models.get("encoder_model")
    if not isinstance(finetune_spec, dict):
        return False

    finetune_constructor = finetune_spec.get("constructor_name")
    encoder_constructor = ENCODER_CONSTRUCTOR_BY_MODEL.get(finetune_constructor)
    if encoder_constructor is None:
        return False

    changed = False

    # Ensure the foundation is frozen (required for caching to be valid).
    params = finetune_spec.setdefault("params", {})
    if params.get("freeze_foundation") is not True:
        params["freeze_foundation"] = True
        changed = True

    if _apply_caching_to_finetune_spec(finetune_spec, encoder_constructor, path, verbose):
        changed = True

    return changed


def migrate_config(path: Path, dry_run: bool, verbose: bool) -> bool:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        return False

    model_spec = config.get("model_spec")
    if not isinstance(model_spec, dict):
        return False

    changed = _migrate_simple_finetune(model_spec, path, verbose) or \
              _migrate_llm_decoding(model_spec, path, verbose)

    if changed and not dry_run:
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which configs would change without writing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print configs that are skipped because they are already up to date.",
    )
    args = parser.parse_args()

    targets = sorted(FOUNDATION_CONFIG_ROOT.glob("*/*/*.yml"))
    changed: list[Path] = []
    for path in targets:
        if migrate_config(path, dry_run=args.dry_run, verbose=args.verbose):
            changed.append(path)
            action = "Would update" if args.dry_run else "Updated"
            print(f"  [{action}] {path.relative_to(REPO_ROOT)}")

    print(
        f"\nProcessed {len(targets)} config files. "
        f"{'Would update' if args.dry_run else 'Updated'} {len(changed)} files."
    )


if __name__ == "__main__":
    main()
