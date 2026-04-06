#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG_ROOT = REPO_ROOT / "configs" / "foundation_models"

# Models that support per-subject variant. Convention-based naming:
#   config_setter:  {model}_per_subject
#   preprocessor:   {model}_per_subject_feature_extraction
#   constructor:    linear_probe  (shared, model-agnostic)
PERSUBJECT_MODELS = {"brainbert", "popt"}


def iter_supersubject_templates() -> list[Path]:
    return sorted(FOUNDATION_CONFIG_ROOT.glob("*/*/supersubject.yml"))


def build_subject_variant_config(template_cfg: dict, model: str, task: str, subject_id: int) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))

    task_config = cfg.setdefault("task_config", {})
    data_params = task_config.setdefault("data_params", {})

    data_params["subject_ids"] = [subject_id]
    data_params["electrode_file_path"] = None
    data_params["channel_reg_ex"] = None
    data_params["per_subject_electrodes"] = None

    cfg["trial_name"] = f"{model}_subject{subject_id}_full_{task}"
    return cfg


def build_persubject_variant_config(template_cfg: dict, model: str, task: str) -> dict | None:
    """Generate a persubject variant from a supersubject template."""
    if model not in PERSUBJECT_MODELS:
        return None

    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))

    cfg["config_setter_name"] = f"{model}_per_subject"

    task_config = cfg.setdefault("task_config", {})
    data_params = task_config.setdefault("data_params", {})
    data_params["per_subject_preprocessing"] = True
    data_params["preprocessing_fn_name"] = [f"{model}_per_subject_feature_extraction"]

    # Linear probe on concatenated per-subject embeddings
    model_spec = cfg.setdefault("model_spec", {})
    model_spec["constructor_name"] = "linear_probe"
    old_params = model_spec.get("params", {})
    model_spec["params"] = {
        "model_dir": old_params.get("model_dir"),
        "layer_sizes": [old_params.get("output_dim", 1)],
        "dropout": 0.0,
    }
    model_spec.pop("sub_models", None)

    training = cfg.setdefault("training_params", {})
    training["batch_size"] = 32
    training["learning_rate"] = 1.0e-03

    cfg["trial_name"] = f"{model}_persubject_{task}"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate config variants from supersubject templates."
    )
    parser.add_argument(
        "--variant",
        choices=["subject-full", "persubject", "all"],
        default="subject-full",
        help="Which variant type to generate. Default: subject-full",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, 10)),
        help="Subject ids to generate (for subject-full variant). Default: 1 2 3 4 5 6 7 8 9",
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

        # --- subject-full variants ---
        if args.variant in ("subject-full", "all"):
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

        # --- persubject variant ---
        if args.variant in ("persubject", "all"):
            cfg = build_persubject_variant_config(template_cfg, model, task)
            if cfg is None:
                continue
            output_path = template_path.with_name("persubject.yml")

            if args.dry_run:
                print(output_path)
                written += 1
                continue

            with open(output_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written += 1

    mode = "Would write" if args.dry_run else "Wrote"
    print(f"{mode} {written} config files from {len(templates)} templates.")
    if args.variant != "all":
        print(f"Note: generated {args.variant} variant only.")


if __name__ == "__main__":
    main()
