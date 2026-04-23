#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG_ROOT = REPO_ROOT / "configs" / "foundation_models"

ENCODER_CONSTRUCTOR_BY_MODEL = {
    "brainbert_finetune": "brainbert_encoder",
    "diver_finetune": "diver_encoder",
    "popt_finetune": "popt_encoder",
}


def normalize_foundation_model_spec_for_caching(cfg: dict) -> None:
    model_spec = cfg.get("model_spec", {})
    constructor_name = model_spec.get("constructor_name")
    encoder_constructor_name = ENCODER_CONSTRUCTOR_BY_MODEL.get(constructor_name)
    if encoder_constructor_name is None:
        return
    if model_spec.get("params", {}).get("freeze_foundation") is not True:
        return

    sub_models = model_spec.setdefault("sub_models", {})
    encoder_spec = sub_models.get("encoder_model")
    if isinstance(encoder_spec, dict) and encoder_spec.get("constructor_name") == "caching_model":
        return

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


def iter_supersubject_templates() -> list[Path]:
    return sorted(FOUNDATION_CONFIG_ROOT.glob("*/*/supersubject.yml"))


def build_subject_variant_config(template_cfg: dict, model: str, task: str, subject_id: int) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))
    normalize_foundation_model_spec_for_caching(cfg)

    task_config = cfg.setdefault("task_config", {})
    data_params = task_config.setdefault("data_params", {})

    data_params["subject_ids"] = [subject_id]
    data_params["electrode_file_path"] = None
    data_params["channel_reg_ex"] = None
    data_params["per_subject_electrodes"] = None

    cfg["trial_name"] = f"{model}_subject{subject_id}_full_{task}"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate subjectN_full.yml variants from supersubject templates."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, 10)),
        help="Subject ids to generate. Default: 1 2 3 4 5 6 7 8 9",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    args = parser.parse_args()

    templates = iter_supersubject_templates()
    if not templates:
        raise SystemExit(f"No supersubject templates found under {FOUNDATION_CONFIG_ROOT}")

    written = 0
    for template_path in templates:
        model = template_path.parents[1].name
        task = template_path.parent.name

        with open(template_path, "r") as f:
            template_cfg = yaml.safe_load(f)

        for subject_id in args.subjects:
            output_path = template_path.with_name(f"subject{subject_id}_full.yml")
            cfg = build_subject_variant_config(template_cfg, model, task, subject_id)

            if args.dry_run:
                print(output_path)
                written += 1
                continue

            with open(output_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written += 1

    mode = "Would write" if args.dry_run else "Wrote"
    print(f"{mode} {written} subject-full config files from {len(templates)} templates.")
    print("Note: experimental lazy-volume configs are intentionally not generated here.")


if __name__ == "__main__":
    main()
