"""Generate Brain Treebank configs for the supported foundation model families."""

from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "configs" / "foundation_models"
OUTPUT_ROOT = PROJECT_ROOT / "configs" / "brain_treebank"
MODELS = ("brainbert", "diver", "popt")
TASKS = {
    "content_noncontent": "content_noncontent_task",
    "gpt_surprise": "gpt_surprise_task",
    "gpt_surprise_multiclass": "gpt_surprise_multiclass_task",
    "pos": "pos_task",
    "sentence_onset": "sentence_onset_task",
}


def _task_specific_config(task: str) -> dict:
    config = {"labels_path": "processed_data/brain_treebank/cars-2/word_labels.csv"}
    if task == "sentence_onset":
        config["negatives_per_positive"] = 5
    return config


def _configure_foundation_paths(model: str, config: dict) -> None:
    foundation = config["task_config"]["data_params"]["preprocessor_params"][-1]
    params = foundation["foundation_model_spec"]["params"]
    if model == "brainbert":
        params["model_dir"] = (
            "/scratch/gpfs/HASSON/lucy/PODCAST/podcast-benchmark/"
            "models/brainbert/pretrained_model"
        )
    elif model == "diver":
        params["foundation_dir"] = (
            "/scratch/gpfs/HASSON/lucy/PODCAST/podcast-benchmark/"
            "models/diver/pretrained_model/256_mp_rank_00_model_states.pt"
        )
    else:
        params["model_dir"] = (
            "/scratch/gpfs/HASSON/lucy/PODCAST/podcast-benchmark/"
            "models/popt/pretrained_model"
        )
        params["brainbert_model_dir"] = (
            "/scratch/gpfs/HASSON/lucy/PODCAST/podcast-benchmark/"
            "models/brainbert/pretrained_model"
        )


def build_config(model: str, task: str) -> dict:
    source = SOURCE_ROOT / model / task / "subject3_full.yml"
    with source.open() as stream:
        config = yaml.safe_load(stream)
    config = deepcopy(config)

    data_params = config["task_config"]["data_params"]
    data_params.update(
        {
            "dataset_name": "brain_treebank",
            "dataset_params": {"movie": "cars-2"},
            "data_root": "data/brain-treebank",
            "subject_ids": [3, 7, 10],
            "electrode_file_path": None,
            "channel_reg_ex": None,
            "per_subject_electrodes": None,
            "use_high_gamma": False,
        }
    )
    data_params["preprocessor_params"][-1]["mode"] = "normal"

    config["task_config"]["task_name"] = TASKS[task]
    config["task_config"]["task_specific_config"] = _task_specific_config(task)
    config["training_params"].update(
        {"min_lag": -1000, "max_lag": 1100, "lag_step_size": 100}
    )
    config["model_spec"]["params"].pop("embedding_dim", None)
    config["run_mode"] = "per_subject"
    config["trial_name"] = f"brain_treebank_cars2_{model}_{task}_per_subject"
    _configure_foundation_paths(model, config)
    return config


def main() -> None:
    for model in MODELS:
        output_dir = OUTPUT_ROOT / model
        output_dir.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            path = output_dir / f"{task}.yml"
            with path.open("w") as stream:
                yaml.safe_dump(build_config(model, task), stream, sort_keys=False)
            print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
