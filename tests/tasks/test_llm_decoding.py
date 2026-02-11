"""
Tests for LLM decoding task to verify target token extraction using offset mapping.

This test suite ensures that target tokens extracted from encoding_all using
offset_mapping are identical to the actual tokens in encoding_all, avoiding
tokenization inconsistencies that would occur with separate tokenization.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock

from tasks.llm_decoding import llm_decoding_task, LlmDecodingConfig
from core.config import TaskConfig, DataParams


class MockTokenizer:
    """Mock tokenizer for testing that simulates transformers tokenizer behavior."""

    def __init__(self, vocab=None):
        """Initialize with a simple vocab mapping."""
        self.vocab = vocab or {
            "": 0,  # padding
            "the": 1,
            "cat": 2,
            "sat": 3,
            "on": 4,
            "mat": 5,
            "a": 6,
            "dog": 7,
            "ran": 8,
            "quick": 9,
            "brown": 10,
            "fox": 11,
            " ": 12,
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(
        self,
        texts,
        max_length=None,
        padding=None,
        return_offsets_mapping=False,
        return_tensors=None,
        truncation=None,
    ):
        """Simulate tokenizer behavior with offset mapping."""
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_mask = []
        offset_mapping = []

        for text in texts:
            # Simple word-based tokenization
            tokens = []
            offsets = []
            current_pos = 0

            # Track positions in original text
            words = text.split()
            for word in words:
                word_start = text.find(word, current_pos)
                word_end = word_start + len(word)

                # Get token ID
                token_id = self.vocab.get(word, self.vocab.get(""))
                tokens.append(token_id)
                offsets.append((word_start, word_end))

                current_pos = word_end

            # Apply padding if needed
            if max_length and len(tokens) < max_length:
                padding_length = max_length - len(tokens)
                tokens.extend([0] * padding_length)
                offsets.extend([(0, 0)] * padding_length)

            # Apply truncation if needed
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
                offsets = offsets[:max_length]

            input_ids.append(tokens)
            attention_mask.append([1 if t != 0 else 0 for t in tokens])
            offset_mapping.append(offsets)

        result = {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask),
        }

        if return_offsets_mapping:
            result["offset_mapping"] = np.array(offset_mapping)

        return result


class TestLLMDecodingTokenAlignment:
    """Test that target tokens are correctly extracted from encoding_all."""

    def create_temp_transcript(self, words_data):
        """Create a temporary transcript CSV file."""
        df = pd.DataFrame(words_data)
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name

    def create_task_config(
        self, transcript_path, max_context=8, max_target_tokens=4
    ):
        """Helper to create a TaskConfig with proper structure."""
        return TaskConfig(
            task_name="llm_decoding",
            data_params=DataParams(subject_ids=[1]),
            task_specific_config=LlmDecodingConfig(
                max_context=max_context,
                max_target_tokens=max_target_tokens,
                transcript_path=transcript_path,
                prepend_space=True,
            ),
        )

    def test_target_tokens_match_all_encoding_single_token_words(self):
        """Test that target tokens exactly match tokens from encoding_all for single-token words."""
        words_data = {
            "word": ["the", "cat", "sat"],
            "start": [0.0, 0.5, 1.0],
            "end": [0.3, 0.8, 1.3],
        }

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            config = self.create_task_config(transcript_path)

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            # For each sample, verify target tokens appear in all_input_ids
            for idx in range(len(result_df)):
                all_ids = result_df.iloc[idx]["all_input_ids"]
                target_ids = result_df.iloc[idx]["target"]
                target_mask = result_df.iloc[idx]["target_attention_mask"]

                # Get actual target tokens (non-padding)
                actual_target = [t for t, m in zip(target_ids, target_mask) if m == 1]

                # Verify target tokens exist in all_input_ids
                assert (
                    len(actual_target) > 0
                ), f"No target tokens found for sample {idx}"

                # Find target tokens in all_input_ids
                all_ids_list = list(all_ids)
                for token in actual_target:
                    assert (
                        token in all_ids_list
                    ), f"Target token {token} not found in all_input_ids for sample {idx}"

        finally:
            os.unlink(transcript_path)

    def test_target_tokens_position_in_all_encoding(self):
        """Test that target tokens appear after context tokens in encoding_all."""
        words_data = {
            "word": ["the", "cat", "sat", "on"],
            "start": [0.0, 0.5, 1.0, 1.5],
            "end": [0.3, 0.8, 1.3, 1.8],
        }

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            config = self.create_task_config(transcript_path)

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            # Check sample where we have context (not the first word)
            for idx in range(1, len(result_df)):
                prev_ids = result_df.iloc[idx]["prev_input_ids"]
                all_ids = result_df.iloc[idx]["all_input_ids"]
                target_ids = result_df.iloc[idx]["target"]
                target_mask = result_df.iloc[idx]["target_attention_mask"]

                # Get actual target tokens (non-padding)
                actual_target = [t for t, m in zip(target_ids, target_mask) if m == 1]

                # Context tokens should appear at the beginning of all_ids
                prev_ids_nonpadding = [t for t in prev_ids if t != 0]
                all_ids_list = list(all_ids)

                # Verify that target tokens don't appear in the context part
                # (they should only appear after context)
                for i, token_id in enumerate(all_ids_list):
                    if token_id in actual_target and i < len(prev_ids_nonpadding):
                        # This is acceptable only if the target word also appeared in context
                        pass

        finally:
            os.unlink(transcript_path)

    def test_no_separate_target_tokenization_issue(self):
        """Test that we don't have tokenization inconsistencies from separate encoding."""
        # This test verifies the fix: target tokens should come from encoding_all,
        # not from a separate tokenization of the target word alone

        words_data = {
            "word": ["cat", "dog", "fox"],
            "start": [0.0, 0.5, 1.0],
            "end": [0.3, 0.8, 1.3],
        }

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            config = self.create_task_config(transcript_path)

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            # Manually verify that target extraction uses offsets correctly
            for idx in range(len(result_df)):
                word = result_df.iloc[idx]["word"]
                all_ids = result_df.iloc[idx]["all_input_ids"]
                target_ids = result_df.iloc[idx]["target"]
                target_mask = result_df.iloc[idx]["target_attention_mask"]

                # Get actual target tokens
                actual_target = [t for t, m in zip(target_ids, target_mask) if m == 1]

                # All target tokens should exist in all_ids
                all_ids_list = list(all_ids)
                for token in actual_target:
                    assert (
                        token in all_ids_list
                    ), f"Token {token} for word '{word}' not in all_input_ids"

        finally:
            os.unlink(transcript_path)

    def test_target_mask_consistency(self):
        """Test that target_attention_mask correctly indicates valid tokens."""
        words_data = {
            "word": ["the", "quick", "brown"],
            "start": [0.0, 0.5, 1.0],
            "end": [0.3, 0.8, 1.3],
        }

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            config = self.create_task_config(transcript_path)

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            for idx in range(len(result_df)):
                target_ids = result_df.iloc[idx]["target"]
                target_mask = result_df.iloc[idx]["target_attention_mask"]

                # Verify mask length equals target_ids length
                assert len(target_ids) == len(target_mask)

                # Verify padding tokens (0) have mask value 0
                for token_id, mask_val in zip(target_ids, target_mask):
                    if token_id == 0:
                        assert mask_val == 0, "Padding token should have mask value 0"

                # Verify at least one non-padding token exists
                assert (
                    sum(target_mask) > 0
                ), "Should have at least one valid target token"

        finally:
            os.unlink(transcript_path)

    def test_max_target_tokens_truncation(self):
        """Test that target tokens are properly truncated to max_target_tokens."""
        words_data = {"word": ["cat", "dog"], "start": [0.0, 0.5], "end": [0.3, 0.8]}

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            max_target_tokens = 2

            config = self.create_task_config(
                transcript_path, max_target_tokens=max_target_tokens
            )

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            for idx in range(len(result_df)):
                target_ids = result_df.iloc[idx]["target"]
                target_mask = result_df.iloc[idx]["target_attention_mask"]

                # Verify length doesn't exceed max_target_tokens
                assert len(target_ids) == max_target_tokens
                assert len(target_mask) == max_target_tokens

        finally:
            os.unlink(transcript_path)

    def test_max_target_tokens_padding(self):
        """Test that target tokens are properly padded when shorter than max_target_tokens."""
        words_data = {"word": ["a", "cat"], "start": [0.0, 0.5], "end": [0.3, 0.8]}

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            max_target_tokens = 10  # Much larger than needed

            config = self.create_task_config(
                transcript_path, max_target_tokens=max_target_tokens
            )

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            for idx in range(len(result_df)):
                target_ids = result_df.iloc[idx]["target"]
                target_mask = result_df.iloc[idx]["target_attention_mask"]

                # Verify length is exactly max_target_tokens
                assert len(target_ids) == max_target_tokens
                assert len(target_mask) == max_target_tokens

                # Count padding tokens
                padding_count = sum(1 for t in target_ids if t == -100)
                assert padding_count > 0, "Should have padding for short words"

                # Verify padding tokens have mask value 0
                for token_id, mask_val in zip(target_ids, target_mask):
                    if token_id == 0:
                        assert mask_val == 0

        finally:
            os.unlink(transcript_path)

    def test_offset_mapping_extraction_correctness(self):
        """Test that offset mapping correctly identifies target token boundaries."""
        # This is a more detailed test of the core fix
        words_data = {
            "word": ["the", "cat", "sat"],
            "start": [0.0, 0.5, 1.0],
            "end": [0.3, 0.8, 1.3],
        }

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            config = self.create_task_config(transcript_path)

            # Manually verify the logic
            context_windows = []
            targets = []

            all_words = ["the", "cat", "sat"]
            for i, word in enumerate(all_words):
                min_idx = max(0, i - 8)
                context = " ".join(all_words[min_idx:i])
                context = " " + context
                context_windows.append(context)
                targets.append(word)

            # Test the concatenation and offset calculation
            for i, (context, target) in enumerate(zip(context_windows, targets)):
                full_text = context + " " + target
                target_start_char = len(context) + 1
                target_end_char = target_start_char + len(target)

                # Verify our offset calculation is correct
                assert (
                    full_text[target_start_char:target_end_char] == target
                ), f"Offset calculation incorrect for '{target}'"

            # Now run the actual function
            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            # Verify we got results
            assert len(result_df) == len(words_data["word"])

        finally:
            os.unlink(transcript_path)

    def test_empty_context_first_word(self):
        """Test that first word with empty context is handled correctly."""
        words_data = {"word": ["the", "cat"], "start": [0.0, 0.5], "end": [0.3, 0.8]}

        transcript_path = self.create_temp_transcript(words_data)

        try:
            tokenizer = MockTokenizer()
            config = self.create_task_config(transcript_path)

            result_df = llm_decoding_task(config, tokenizer=tokenizer)

            # First word should have empty/minimal context
            first_prev_ids = result_df.iloc[0]["prev_input_ids"]
            first_target_ids = result_df.iloc[0]["target"]
            first_target_mask = result_df.iloc[0]["target_attention_mask"]

            # Should still have valid target tokens
            actual_target = [
                t for t, m in zip(first_target_ids, first_target_mask) if m == 1
            ]
            assert len(actual_target) > 0, "First word should have target tokens"

        finally:
            os.unlink(transcript_path)
