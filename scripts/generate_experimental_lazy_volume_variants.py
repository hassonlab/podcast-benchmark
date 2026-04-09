#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_ROOT = REPO_ROOT / "configs" / "foundation_models"
EXPERIMENTAL_ROOT = REPO_ROOT / "configs" / "experimental" / "lazy_volume_level"
MODELS = ["brainbert", "popt", "diver"]


def iter_standard_volume_variants(model: str) -> list[Path]:
    return sorted((STANDARD_ROOT / model / "volume_level").glob("*.yml"))


def main() -> None:
    written = 0
    for model in MODELS:
        src_dir = STANDARD_ROOT / model / "volume_level"
        dst_dir = EXPERIMENTAL_ROOT / model / "volume_level"
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src_path in iter_standard_volume_variants(model):
            with open(src_path, "r") as f:
                cfg = yaml.safe_load(f)

            trial_name = cfg.get("trial_name", f"{model}_{src_path.stem}_volume_level")
            if not str(trial_name).endswith("_exp_lazy"):
                cfg["trial_name"] = f"{trial_name}_exp_lazy"

            dst_path = dst_dir / src_path.name
            with open(dst_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written += 1

    print(f"Wrote {written} experimental lazy-volume variant configs.")


if __name__ == "__main__":
    main()
