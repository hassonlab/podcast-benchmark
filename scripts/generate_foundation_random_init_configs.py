#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG_ROOT = REPO_ROOT / "configs" / "foundation_models"
CONTROL_CONFIG_ROOT = REPO_ROOT / "configs" / "controls" / "foundation_random_init"


def iter_control_templates() -> list[Path]:
    templates = list(FOUNDATION_CONFIG_ROOT.glob("*/*/supersubject.yml"))
    templates.extend(FOUNDATION_CONFIG_ROOT.glob("*/*/subject[1-9]_full.yml"))
    return sorted(templates)


def build_random_init_config(template_cfg: dict) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))
    preprocessor_params = cfg["task_config"]["data_params"]["preprocessor_params"]
    foundation_params = next(
        params for params in preprocessor_params if "foundation_model_spec" in params
    )
    foundation_params["foundation_model_spec"]["random_init"] = True
    cfg["trial_name"] = f"{cfg['trial_name']}_random_init"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random-init controls from foundation supersubject and "
            "individual-subject configs."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    args = parser.parse_args()

    templates = iter_control_templates()
    if not templates:
        raise SystemExit(f"No control templates found under {FOUNDATION_CONFIG_ROOT}")

    written = 0
    for template_path in templates:
        model = template_path.parents[1].name
        task = template_path.parent.name
        output_path = CONTROL_CONFIG_ROOT / f"{model}_{task}_{template_path.name}"

        if args.dry_run:
            print(output_path)
            written += 1
            continue

        with open(template_path) as f:
            template_cfg = yaml.safe_load(f)
        cfg = build_random_init_config(template_cfg)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        written += 1

    mode = "Would write" if args.dry_run else "Wrote"
    print(f"{mode} {written} random-init foundation control configs.")


if __name__ == "__main__":
    main()
