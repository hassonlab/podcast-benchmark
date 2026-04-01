#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path("/storage/connectome/jmhan/podcast-benchmark")
FOUNDATION_ROOT = PROJECT_ROOT / "configs" / "foundation_models"
EXPERIMENTAL_LAZY_ROOT = (
    PROJECT_ROOT / "configs" / "experimental" / "lazy_volume_level"
)

DEFAULT_FOLD_IDS = [1, 5]
DEFAULT_LAGS = [-1000, -500, 0, 500, 1000]


def lag_range_from_points(lags: list[int]) -> tuple[int, int, int]:
    if len(lags) < 2:
        raise ValueError("Need at least two lag points to derive step size.")

    steps = [b - a for a, b in zip(lags, lags[1:])]
    if len(set(steps)) != 1:
        raise ValueError(f"Lags are not evenly spaced: {lags}")

    step = steps[0]
    return lags[0], lags[-1] + step, step


def update_training_params(path: Path, dry_run: bool) -> bool:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected YAML root in {path}")

    training_params = config.setdefault("training_params", {})

    min_lag, max_lag, lag_step = lag_range_from_points(DEFAULT_LAGS)

    changed = False
    updates = {
        "fold_ids": list(DEFAULT_FOLD_IDS),
        "lag": None,
        "min_lag": min_lag,
        "max_lag": max_lag,
        "lag_step_size": lag_step,
    }

    for key, value in updates.items():
        if training_params.get(key) != value:
            training_params[key] = value
            changed = True

    if changed and not dry_run:
        path.write_text(yaml.safe_dump(config, sort_keys=False))

    return changed


def collect_targets(include_experimental: bool) -> list[Path]:
    roots = [FOUNDATION_ROOT]
    if include_experimental:
        roots.append(EXPERIMENTAL_LAZY_ROOT)

    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("*/*/*.yml")))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set global fold/lag defaults for foundation-model configs and optional "
            "experimental lazy-volume configs."
        )
    )
    parser.add_argument(
        "--skip-experimental",
        action="store_true",
        help="Only update configs/foundation_models and skip configs/experimental/lazy_volume_level.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would change without writing them.",
    )
    args = parser.parse_args()

    targets = collect_targets(include_experimental=not args.skip_experimental)
    changed_paths: list[Path] = []

    for path in targets:
        if update_training_params(path, dry_run=args.dry_run):
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
