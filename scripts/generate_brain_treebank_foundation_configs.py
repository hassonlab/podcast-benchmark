"""Generate Brain Treebank configs for the supported foundation model families."""

from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "configs" / "foundation_models"
OUTPUT_ROOT = PROJECT_ROOT / "configs" / "brain_treebank"
MODELS = ("brainbert", "diver", "popt")
SUBJECTS = (3, 7, 10)
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


def build_config(model: str, task: str, subject_id: int) -> dict:
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
            "subject_ids": [subject_id],
            "electrode_file_path": None,
            "channel_reg_ex": None,
            "per_subject_electrodes": None,
            "use_high_gamma": False,
        }
    )
    data_params["chunked_preprocessing"] = {
        "enabled": True,
        "num_chunks": 20,
        "cache_dir": ".cache/preprocessed_chunks",
    }
    if model == "popt":
        # The public Brain Treebank localization table contains MNI XYZ but not
        # the L/I/P integer indices used to pretrain PopT's position lookup.
        data_params["use_lip_coords"] = False
    data_params["preprocessor_params"][-1]["mode"] = "normal"

    config["task_config"]["task_name"] = TASKS[task]
    config["task_config"]["task_specific_config"] = _task_specific_config(task)
    config["training_params"].update(
        {
            "min_lag": -1000,
            "max_lag": 1100,
            "lag_step_size": 100,
            "weight_decay": 1.0e-3,
        }
    )
    config["model_spec"]["params"].pop("embedding_dim", None)
    config["model_spec"]["params"]["input_dropout"] = 0.5
    if model == "diver":
        config["model_spec"]["params"]["coordinate_root"] = (
            "processed_data/brain_treebank/coordinates"
        )
    config["run_mode"] = "per_subject"
    config["trial_name"] = f"brain_treebank_cars2_{model}_{task}_subject{subject_id}"
    _configure_foundation_paths(model, config)
    if model == "popt":
        foundation_params = data_params["preprocessor_params"][-1][
            "foundation_model_spec"
        ]["params"]
        foundation_params["use_lip_coords"] = False
    return config


def main() -> None:
    for model in MODELS:
        output_dir = OUTPUT_ROOT / model
        output_dir.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            for subject_id in SUBJECTS:
                suffix = "" if subject_id == 3 else f"_subject{subject_id}"
                path = output_dir / f"{task}{suffix}.yml"
                with path.open("w") as stream:
                    yaml.safe_dump(
                        build_config(model, task, subject_id), stream, sort_keys=False
                    )
                print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
