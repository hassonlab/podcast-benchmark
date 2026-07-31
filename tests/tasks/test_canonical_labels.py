import pandas as pd

from core.config import TaskConfig
from tasks.content_noncontent import ContentNonContentConfig, content_noncontent_task
from tasks.gpt_surprise import GptSurpriseConfig, gpt_surprise_task
from tasks.pos_task import PosTaskConfig, pos_task
from tasks.sentence_onset import SentenceOnsetConfig, sentence_onset_task


def _write_labels(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "start": [1.0, 2.0, 3.0, 4.0],
            "end": [1.2, 2.2, 3.2, 4.2],
            "word": ["one", "two", "three", "four"],
            "surprisal": [1.5, 2.5, 3.5, 4.5],
            "is_content": [1, 0, 1, 0],
            "pos_class": [0, 4, 1, 4],
            "sentence_onset": [True, False, True, False],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_four_word_tasks_share_canonical_table(tmp_path):
    path = _write_labels(tmp_path)
    surprise = gpt_surprise_task(
        TaskConfig(task_specific_config=GptSurpriseConfig(labels_path=path))
    )
    content = content_noncontent_task(
        TaskConfig(task_specific_config=ContentNonContentConfig(labels_path=path))
    )
    pos = pos_task(TaskConfig(task_specific_config=PosTaskConfig(labels_path=path)))
    sentence = sentence_onset_task(
        TaskConfig(
            task_specific_config=SentenceOnsetConfig(
                labels_path=path, negatives_per_positive=1
            )
        )
    )

    assert surprise["target"].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert content["target"].tolist() == [1, 0, 1, 0]
    assert pos["target"].tolist() == [0, 4, 1, 4]
    assert sorted(sentence["target"].tolist()) == [0.0, 0.0, 1.0, 1.0]
