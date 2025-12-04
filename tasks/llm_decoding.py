from dataclasses import dataclass, field
from typing import Optional, Any

import pandas as pd
import numpy as np

from core.config import BaseTaskConfig, TaskConfig
from core import registry
from language_generation.gpt2_brain import load_gpt2_model_and_tokenizer
from models.shared_config_setters import set_model_spec_fields


@dataclass
class LlmDecodingConfig(BaseTaskConfig):
    """Configuration for llm_decoding_task."""

    input_fields: list[str] = field(
        default_factory=lambda: [
            "all_input_ids",
            "all_attention_mask",
            "target_attention_mask",
        ]
    )
    required_config_setter_names: list[str] = field(
        default_factory=lambda: ["llm_decoding_config_setter"]
    )
    max_context: int = 32
    max_target_tokens: int = 16
    tokenizer: Optional[Any] = None  # Will be set programmatically
    transcript_path: str = "data/stimuli/podcast_transcript.csv"
    prepend_space: bool = True
    model_name: str = "gpt2"
    cache_dir: str = "./model_cache"


@registry.register_config_setter()
def llm_decoding_config_setter(
    experiment_config: TaskConfig, _raws, _task_df
) -> TaskConfig:
    """Config setter for llm_decoding task."""
    if not set_model_spec_fields(
        experiment_config.model_spec,
        {"cache_dir": experiment_config.task_specific_config.cache_dir},
        ["gpt2_brain"],
    ):
        raise ValueError(
            "Could not set cache_dir for gpt2_brain in llm_decoding_config_setter."
        )
    return experiment_config


@registry.register_task_data_getter(config_type=LlmDecodingConfig)
def llm_decoding_task(task_config: TaskConfig):
    """Task for LLM decoding."""
    config: LlmDecodingConfig = task_config.task_specific_config
    _, tokenizer = load_gpt2_model_and_tokenizer(
        cache_dir=task_config.task_specific_config.cache_dir,
    )
    max_context = config.max_context
    max_target_tokens = config.max_target_tokens

    # LLM data
    df_word = pd.read_csv(config.transcript_path)
    stripped_words = df_word.word.str.strip()
    full_transcript = stripped_words.str.cat(sep=" ")

    all_words = full_transcript.split()
    context_windows = []
    # Track bounds of target word for alignment with tokens.
    target_bounds = []
    targets = []
    current_char_pos = 0
    for i, word in enumerate(all_words):
        min_idx = max(0, i - max_context)
        context = " ".join(all_words[min_idx:i])
        if config.prepend_space:
            context = " " + context
        context_windows.append(context)
        targets.append(word)

        word_start = full_transcript.find(word, current_char_pos)
        word_end = word_start + len(word)
        target_bounds.append((word_start, word_end))
        current_char_pos = word_end

    encoding_prev = tokenizer(
        context_windows,
        max_length=max_context,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="np",
        truncation=True,
    )
    encoding_all = tokenizer(
        np.char.add(np.char.add(context_windows, " "), targets).tolist(),
        max_length=max_context + max_target_tokens,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="np",
        truncation=True,
    )

    # Use offsets_mapping to identify which tokens in encoding_all correspond to the target word
    target_input_ids = []
    target_attention_mask = []

    for i, (context, target) in enumerate(zip(context_windows, targets)):
        # The target word starts after context + " " in the concatenated string
        target_start_char = len(context) + 1  # +1 for the space
        target_end_char = target_start_char + len(target)

        offsets = encoding_all["offset_mapping"][i]
        input_ids = encoding_all["input_ids"][i]

        # Find tokens that overlap with the target word span
        target_tokens = []
        target_mask = []
        for token_idx, (start, end) in enumerate(offsets):
            # Check if this token overlaps with the target word
            if start < target_end_char and end > target_start_char:
                target_tokens.append(input_ids[token_idx])
                target_mask.append(1)

        # Pad or truncate to max_target_tokens
        if len(target_tokens) > max_target_tokens:
            target_tokens = target_tokens[:max_target_tokens]
            target_mask = target_mask[:max_target_tokens]
        else:
            # Pad with -100s for ignore_ids in loss computation
            padding_length = max_target_tokens - len(target_tokens)
            target_tokens.extend([-100] * padding_length)
            target_mask.extend([0] * padding_length)

        target_input_ids.append(target_tokens)
        target_attention_mask.append(target_mask)

    return pd.DataFrame(
        {
            "prev_input_ids": list(encoding_prev["input_ids"]),
            "prev_attention_mask": list(encoding_prev["attention_mask"]),
            "all_input_ids": list(encoding_all["input_ids"]),
            "all_attention_mask": list(encoding_all["attention_mask"]),
            "target": target_input_ids,
            "target_attention_mask": target_attention_mask,
            "word": df_word.word,
            "start": df_word.start,
            "end": df_word.end,
        }
    )
