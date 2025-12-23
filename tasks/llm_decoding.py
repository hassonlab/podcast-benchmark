from dataclasses import dataclass, field
from typing import Optional, Any

import pandas as pd
import numpy as np

from core.config import BaseTaskConfig, TaskConfig, ExperimentConfig
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
    transcript_path: str = "data/stimuli/podcast_transcript.csv"
    prepend_space: bool = True
    model_name: str = "gpt2"
    cache_dir: str = "./model_cache"


@dataclass
class LlmEmbeddingPretrainingConfig(BaseTaskConfig):
    """Configuration for LLM embedding pre-training task.

    This task trains an encoder to predict the average of token embeddings
    from GPT-2's embedding layer. The pre-trained encoder can then be loaded
    into the full GPT2Brain model for token prediction fine-tuning.
    """

    # No extra input fields needed - encoder predicts embeddings directly
    input_fields: list[str] = field(default_factory=lambda: [])

    # Context and target configuration (should match llm_decoding_task)
    max_context: int = 32
    max_target_tokens: int = 16
    transcript_path: str = "data/stimuli/podcast_transcript.csv"
    prepend_space: bool = True

    # GPT-2 model configuration
    model_name: str = "gpt2"  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    cache_dir: str = "./model_cache"


@registry.register_config_setter()
def llm_decoding_config_setter(
    experiment_config: ExperimentConfig, _raws, _task_df
) -> ExperimentConfig:
    """Config setter for llm_decoding task."""
    if not set_model_spec_fields(
        experiment_config.model_spec,
        {"cache_dir": experiment_config.task_config.task_specific_config.cache_dir},
        ["gpt2_brain"],
    ):
        raise ValueError(
            "Could not set cache_dir for gpt2_brain in llm_decoding_config_setter."
        )
    return experiment_config


@registry.register_task_data_getter(config_type=LlmDecodingConfig)
def llm_decoding_task(task_config: TaskConfig, tokenizer=None):
    """Task for LLM decoding."""
    config: LlmDecodingConfig = task_config.task_specific_config
    if tokenizer is None:
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


@registry.register_task_data_getter(config_type=LlmEmbeddingPretrainingConfig)
def llm_embedding_pretraining_task(
    task_config: TaskConfig, gpt2_model=None, tokenizer=None
):
    """Task for LLM embedding pre-training.

    Computes average token embeddings from GPT-2's embedding layer as targets
    for encoder pre-training. Each word from the transcript is tokenized, and
    the embeddings of its tokens are averaged to create a single target vector.

    Args:
        task_config: Task configuration
        gpt2_model: Optional pre-loaded GPT-2 model (for reuse/testing)
        tokenizer: Optional pre-loaded tokenizer (for reuse/testing)

    Returns:
        DataFrame with columns:
            - target: Average token embedding [embedding_dim]
            - word: Original word string
            - start: Word start time in seconds
            - end: Word end time in seconds
    """
    import torch

    config: LlmEmbeddingPretrainingConfig = task_config.task_specific_config

    # Load GPT-2 model and tokenizer if not provided
    if gpt2_model is None or tokenizer is None:
        gpt2_model, tokenizer = load_gpt2_model_and_tokenizer(
            cache_dir=config.cache_dir, model_name=config.model_name
        )

    # Get embedding layer from GPT-2
    embedding_layer = gpt2_model.transformer.wte

    # Load transcript data
    df_word = pd.read_csv(config.transcript_path)
    stripped_words = df_word.word.str.strip()
    full_transcript = stripped_words.str.cat(sep=" ")
    all_words = full_transcript.split()

    # Prepare target words list
    targets = []
    for i, word in enumerate(all_words):
        targets.append(word)

    # Tokenize target words to get their token IDs
    target_encoding = tokenizer(
        targets,
        return_tensors="np",
        padding=False,  # No padding - handle variable length
        truncation=True,
        max_length=config.max_target_tokens,
    )

    # Compute average embedding for each target word's tokens
    embedding_targets = []
    with torch.no_grad():
        for input_ids in target_encoding["input_ids"]:
            # Get embeddings for this word's tokens
            token_ids_tensor = torch.tensor(input_ids).unsqueeze(0)  # [1, num_tokens]
            token_embeddings = embedding_layer(
                token_ids_tensor
            )  # [1, num_tokens, embedding_dim]

            # Average over token dimension
            avg_embedding = token_embeddings.mean(dim=1).squeeze(0)  # [embedding_dim]
            embedding_targets.append(avg_embedding.numpy())

    # Stack into array [num_samples, embedding_dim]
    embedding_targets = np.stack(embedding_targets)

    # Return DataFrame compatible with existing training pipeline
    return pd.DataFrame(
        {
            "target": list(embedding_targets),  # Store as list of arrays
            "word": targets,
            "start": df_word.start.values[: len(targets)],
            "end": df_word.end.values[: len(targets)],
        }
    )
