from pathlib import Path

from scripts.submit_foundation_control_ranges import (
    command_for_range,
    eligible_control_configs,
)


def test_command_submits_one_repetition_as_an_ordinary_lag_range():
    command = command_for_range(
        Path("configs/controls/foundation_random_init/brainbert_pos_supersubject.yml"),
        "random-init",
        -1000,
        0,
        sbatch_flags="-p debug",
    )

    assert "--training_params.num_null_repetitions=1" in command
    assert "--training_params.lag=null" in command
    assert "--training_params.min_lag=-1000" in command
    assert "--training_params.max_lag=0" in command
    assert "--training_params.lag_step_size=500" in command
    assert "-p" in command
    assert "debug" in command


def test_matrix_selects_supersubject_and_nine_subjects_without_llm_decoding():
    configs = eligible_control_configs(
        Path("configs/controls/foundation_random_init")
    )

    assert len(configs) == 300
    assert not any("llm_decoding" in config.name for config in configs)
    assert not any("persubject_concat" in config.name for config in configs)
    assert sum("word_embedding" in config.name for config in configs) == 30
