from pathlib import Path

import yaml

from scripts.submit_foundation_random_init_lags import (
    command_for_condition,
    control_path,
    main,
)


def test_command_runs_one_configured_multi_lag_null_job():
    config = Path("configs/controls/foundation_random_init/example.yml")

    command = command_for_condition(
        config,
        sbatch_flags="-p debug",
        config_overrides="--training_params.epochs=1",
    )

    assert "--dependency=singleton" in command
    assert not any(arg.startswith("--training_params.lag=") for arg in command)
    assert not any(arg.startswith("--trial_name=") for arg in command)
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


def test_main_ignores_nested_random_init_entries(tmp_path, monkeypatch, capsys):
    trained = tmp_path / "trained"
    trained.mkdir()
    (trained / "lag_performance.csv").write_text("lags\n0\n")
    results = tmp_path / "results.yml"
    results.write_text(
        yaml.safe_dump(
            {
                "results": {
                    "diver": {
                        "pos": {
                            "super_subject": str(trained),
                            "random_init": {"super_subject": "existing"},
                        }
                    }
                }
            }
        )
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    (controls / "diver_pos_supersubject.yml").write_text("{}")
    monkeypatch.setattr(
        "sys.argv",
        [
            "submit",
            f"--results-config={results}",
            f"--controls-root={controls}",
            "--dry-run",
        ],
    )

    assert main() == 0
    assert "would submit 1 jobs for 1 control configs" in capsys.readouterr().out


def test_plan_lag_chunks_balances_and_covers_every_lag():
    from scripts.submit_foundation_random_init_lags import (
        lag_grid,
        lags_per_job,
        plan_lag_chunks,
    )

    lags = lag_grid(-1000, 1000, 100)
    chunks = plan_lag_chunks(lags, lags_per_job("diver", "volume_level", 10.0))

    assert len(chunks) == 4
    assert [lag for chunk in chunks for lag in chunk] == lags
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1


def test_chunk_overrides_set_exclusive_max_lag_and_unique_trial_name():
    from scripts.submit_foundation_random_init_lags import chunk_overrides

    overrides = chunk_overrides("trial", [-1000, -900], 100, "--x=1")

    assert "--training_params.min_lag=-1000" in overrides
    assert "--training_params.max_lag=-800" in overrides
    assert "--trial_name=trial_lags_-1000_-900" in overrides
    assert overrides.startswith("--x=1")
