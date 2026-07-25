from pathlib import Path

import pytest

from scripts.submit_foundation_random_init_lags import (
    arithmetic_batches,
    command_for_batch,
    control_path,
)


def test_arithmetic_batches_are_exact_and_capped_at_five():
    lags = (-1000, -900, -800, -700, -600, -400, -200, -100, 0)

    batches = arithmetic_batches(lags, max_batch_size=5)

    assert batches == (
        (-1000, -900, -800, -700, -600),
        (-400, -200),
        (-100, 0),
    )
    assert [lag for batch in batches for lag in batch] == list(lags)
    assert max(map(len, batches)) == 5


@pytest.mark.parametrize("size", [0, 6])
def test_arithmetic_batches_rejects_sizes_outside_hard_limit(size):
    with pytest.raises(ValueError, match="between 1 and 5"):
        arithmetic_batches((0,), max_batch_size=size)


def test_command_uses_exclusive_max_lag_and_singleton_dependency():
    config = Path("configs/controls/foundation_random_init/example.yml")

    command = command_for_batch(
        config,
        (-1000, -900, -800, -700, -600),
        default_step=100,
        sbatch_flags="-p debug",
        config_overrides="--training_params.epochs=1",
    )

    assert "--dependency=singleton" in command
    assert "--training_params.min_lag=-1000" in command
    assert "--training_params.max_lag=-500" in command
    assert "--training_params.lag_step_size=100" in command
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
