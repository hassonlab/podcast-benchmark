from pathlib import Path

from scripts.submit_foundation_random_init_lags import (
    command_for_condition,
    control_path,
)


def test_command_runs_one_repeated_null_lag_job():
    config = Path("configs/controls/foundation_random_init/example.yml")

    command = command_for_condition(
        config,
        lag=-500,
        sbatch_flags="-p debug",
        config_overrides="--training_params.epochs=1",
    )

    assert "--dependency=singleton" in command
    assert "--training_params.lag=-500" in command
    assert "--trial_name=example_random_init_lag_-500" in command
    assert command.count("-p") == 1
    assert command.count("debug") == 1


def test_control_path_maps_benchmark_entity_to_generated_config():
    path = control_path(
        Path("configs/controls/foundation_random_init"),
        "brainbert",
        "word_embedding",
        "subject3_full",
    )

    assert path == Path(
        "configs/controls/foundation_random_init/"
        "brainbert_word_embedding_subject3_full.yml"
    )
