"""Generate persubject_concat.yml configs from existing supersubject.yml configs.

Usage:
    python generate_config_for_persubject_feature_concat.py

Finds all supersubject.yml under configs/foundation_models/, and creates a
persubject_concat.yml next to each one with:
  - model_spec.per_subject_feature_concat: true
  - trial_name: s/supersubject/persubject_concat/
"""

import glob
import os

import yaml


def generate(supersubject_path: str) -> str:
    with open(supersubject_path) as f:
        cfg = yaml.safe_load(f)

    # Add per_subject_feature_concat flag
    cfg.setdefault("model_spec", {})["per_subject_feature_concat"] = True

    # Update trial_name
    if "trial_name" in cfg:
        cfg["trial_name"] = cfg["trial_name"].replace("supersubject", "persubject_concat")

    out_path = os.path.join(os.path.dirname(supersubject_path), "persubject_concat.yml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return out_path


if __name__ == "__main__":
    pattern = "configs/foundation_models/**/supersubject.yml"
    paths = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(paths)} supersubject configs")
    for p in paths:
        out = generate(p)
        print(f"  {p} -> {out}")
