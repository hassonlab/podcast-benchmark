from pathlib import Path

from scripts.find_lag_gaps import expected_lags, find_gaps


def write_lag_csv(path: Path, lags: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("lags,score\n" + "".join(f"{lag},0.0\n" for lag in lags))


def test_finds_gaps_in_nested_per_subject_results(tmp_path):
    run_dir = tmp_path / "results" / "baseline_task_per_subject"
    write_lag_csv(run_dir / "subject_1" / "lag_performance.csv", [-25, 0, 25])
    write_lag_csv(run_dir / "subject_2" / "lag_performance.csv", [-25, 25])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_subject": "results/baseline_task_per_subject",
                }
            }
        }
    }

    gaps, issues = find_gaps(
        config=config,
        models=["baseline"],
        root=tmp_path,
        expected=expected_lags(-25, 25, 25),
        ignored=set(),
    )

    assert issues == []
    assert len(gaps) == 1
    assert gaps[0].model == "baseline"
    assert gaps[0].entity == "subject_2"
    assert gaps[0].missing == (0,)


def test_finds_gaps_in_root_per_subject_results(tmp_path):
    write_lag_csv(
        tmp_path / "results" / "brainbert_subject1_full_task" / "lag_performance.csv",
        [-100, 0, 100],
    )
    write_lag_csv(
        tmp_path / "results" / "brainbert_subject2_full_task" / "lag_performance.csv",
        [-100, 100],
    )
    config = {
        "results": {
            "brainbert": {
                "task": {
                    "per_subject": [
                        "results/brainbert_subject1_full_task",
                        "results/brainbert_subject2_full_task",
                    ],
                }
            }
        }
    }

    gaps, issues = find_gaps(
        config=config,
        models=["brainbert"],
        root=tmp_path,
        expected=expected_lags(-100, 100, 100),
        ignored=set(),
    )

    assert issues == []
    assert len(gaps) == 1
    assert gaps[0].model == "brainbert"
    assert gaps[0].entity == "subject_2"
    assert gaps[0].missing == (0,)
