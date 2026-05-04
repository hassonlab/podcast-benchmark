import numpy as np
import pandas as pd

from core.config import DataParams, TaskConfig, dict_to_config
from tasks.sentence_onset import SentenceOnsetConfig, sentence_onset_task


def _write_csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _task_config(sentence_csv_path, word_csv_path, negatives_per_positive=1):
    return TaskConfig(
        task_name="sentence_onset_task",
        data_params=DataParams(),
        task_specific_config=SentenceOnsetConfig(
            sentence_csv_path=sentence_csv_path,
            word_csv_path=word_csv_path,
            negatives_per_positive=negatives_per_positive,
        ),
    )


def test_sentence_onset_task_uses_only_matching_word_onsets_as_positives(tmp_path):
    sentence_path = _write_csv(
        tmp_path,
        "sentences.csv",
        [
            {"sentence_onset": 1.0},
            {"sentence_onset": 3.0000004},
            {"sentence_onset": 9.0},
        ],
    )
    word_path = _write_csv(
        tmp_path,
        "words.csv",
        [{"onset": 0.5}, {"onset": 1.0}, {"onset": 2.0}, {"onset": 3.0}],
    )

    df = sentence_onset_task(
        _task_config(sentence_path, word_path, negatives_per_positive=1)
    )

    positives = df.loc[df["target"] == 1.0, "start"].to_numpy()
    assert np.allclose(np.sort(positives), [1.0, 3.0])
    assert not np.isclose(positives, 9.0).any()


def test_sentence_onset_task_samples_negatives_only_from_non_sentence_word_onsets(tmp_path):
    sentence_path = _write_csv(
        tmp_path,
        "sentences.csv",
        [{"sentence_onset": 1.0}, {"sentence_onset": 3.0}],
    )
    word_path = _write_csv(
        tmp_path,
        "words.csv",
        [
            {"onset": 0.5},
            {"onset": 1.0},
            {"onset": 2.0},
            {"onset": 3.0},
            {"onset": 4.0},
        ],
    )

    df = sentence_onset_task(
        _task_config(sentence_path, word_path, negatives_per_positive=1)
    )

    negatives = df.loc[df["target"] == 0.0, "start"].to_numpy()
    assert len(negatives) == 2
    assert set(negatives).issubset({0.5, 2.0, 4.0})
    assert not set(negatives) & {1.0, 3.0}


def test_sentence_onset_task_negatives_per_positive_controls_negative_count(tmp_path):
    sentence_path = _write_csv(
        tmp_path,
        "sentences.csv",
        [{"sentence_onset": 1.0}, {"sentence_onset": 3.0}],
    )
    word_path = _write_csv(
        tmp_path,
        "words.csv",
        [
            {"onset": 0.5},
            {"onset": 1.0},
            {"onset": 2.0},
            {"onset": 3.0},
            {"onset": 4.0},
            {"onset": 5.0},
            {"onset": 6.0},
            {"onset": 7.0},
        ],
    )

    df = sentence_onset_task(
        _task_config(sentence_path, word_path, negatives_per_positive=3)
    )

    assert (df["target"] == 1.0).sum() == 2
    assert (df["target"] == 0.0).sum() == 6


def test_sentence_onset_task_reports_unmatched_sentence_onsets_dropped(tmp_path, capsys):
    sentence_path = _write_csv(
        tmp_path,
        "sentences.csv",
        [
            {"sentence_onset": 1.0},
            {"sentence_onset": 3.0},
            {"sentence_onset": 9.0},
        ],
    )
    word_path = _write_csv(
        tmp_path,
        "words.csv",
        [
            {"onset": 0.5},
            {"onset": 1.0},
            {"onset": 2.0},
            {"onset": 3.0},
            {"onset": 4.0},
        ],
    )

    df = sentence_onset_task(
        _task_config(sentence_path, word_path, negatives_per_positive=1)
    )

    output = capsys.readouterr().out
    assert "Unmatched sentence onsets dropped: 1" in output
    assert (df["target"] == 1.0).sum() == 2


def test_sentence_onset_task_samples_with_replacement_when_needed(tmp_path, capsys):
    sentence_path = _write_csv(
        tmp_path,
        "sentences.csv",
        [{"sentence_onset": 1.0}, {"sentence_onset": 3.0}],
    )
    word_path = _write_csv(
        tmp_path,
        "words.csv",
        [{"onset": 1.0}, {"onset": 2.0}, {"onset": 3.0}],
    )

    df = sentence_onset_task(
        _task_config(sentence_path, word_path, negatives_per_positive=2)
    )

    output = capsys.readouterr().out
    negatives = df.loc[df["target"] == 0.0, "start"].to_numpy()
    assert "Replacement sampling needed: True" in output
    assert len(negatives) == 4
    assert set(negatives) == {2.0}


def test_sentence_onset_task_ignores_legacy_negative_margin(tmp_path):
    sentence_path = _write_csv(
        tmp_path,
        "sentences.csv",
        [{"sentence_onset": 1.0}, {"sentence_onset": 3.0}],
    )
    word_path = _write_csv(
        tmp_path,
        "words.csv",
        [
            {"onset": 0.5},
            {"onset": 1.0},
            {"onset": 2.0},
            {"onset": 3.0},
            {"onset": 4.0},
        ],
    )
    base_config = _task_config(sentence_path, word_path, negatives_per_positive=1)
    margin_config = _task_config(sentence_path, word_path, negatives_per_positive=1)
    margin_config.task_specific_config.negative_margin_s = 100.0

    base_df = sentence_onset_task(base_config)
    margin_df = sentence_onset_task(margin_config)

    pd.testing.assert_frame_equal(base_df, margin_df)


def test_sentence_onset_config_accepts_word_path_and_legacy_margin():
    config = dict_to_config(
        {
            "sentence_csv_path": "sentences.csv",
            "word_csv_path": "words.csv",
            "negatives_per_positive": 5,
            "negative_margin_s": 0.75,
        },
        SentenceOnsetConfig,
    )

    assert config.word_csv_path == "words.csv"
    assert config.negative_margin_s == 0.75
