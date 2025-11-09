"""Task data getter functions for the podcast benchmark."""

from tasks.content_noncontent import content_noncontent_task
from tasks.gpt_surprise import gpt_surprise_task, gpt_surprise_multiclass_task
from tasks.placeholder_task import placeholder_task
from tasks.pos_task import pos_task
from tasks.sentence_onset import sentence_onset_task
from tasks.volume_level import (
    volume_level_decoding_task,
    volume_level_config_setter,
)
from tasks.word_embedding import word_embedding_decoding_task

__all__ = [
    "content_noncontent_task",
    "gpt_surprise_task",
    "gpt_surprise_multiclass_task",
    "placeholder_task",
    "pos_task",
    "sentence_onset_task",
    "volume_level_decoding_task",
    "volume_level_config_setter",
    "word_embedding_decoding_task",
]
