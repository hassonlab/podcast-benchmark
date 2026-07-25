from pathlib import Path

from scripts.submit_foundation_random_init_lags import (
    command_for_condition,
    control_path,
)


def test_command_runs_one_five_lag_condition_job():
    config = Path("configs/controls/foundation_random_init/example.yml")

    command = command_for_condition(
        config,
        sbatch_flags="-p debug",
        config_overrides="--training_params.epochs=1",
    )

    assert "--dependency=singleton" in command
    assert "--training_params.min_lag=-1000" in command
    assert "--training_params.max_lag=1500" in command
    assert "--training_params.lag_step_size=500" in command
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
