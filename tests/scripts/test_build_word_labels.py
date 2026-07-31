import numpy as np
import pandas as pd
import pytest

from scripts.build_word_labels import (
    brain_treebank_word_level,
    build_parser,
    sentence_of,
    surprise_classes,
    tag_by_sentence,
)


def sentences(*texts):
    return pd.DataFrame({"all_sentence": texts})


def test_cli_defaults_to_canonical_output_without_comparison_options():
    parser = build_parser()

    args = parser.parse_args([])

    assert args.out == "processed_data/df_word_onset_with_pos_class.csv"
    assert not hasattr(args, "old")
    assert not hasattr(args, "validate")


def test_sentence_of_aligns_text_and_leaves_omitted_words_unassigned():
    words = np.array(["Intro", "Hello,", "world!", "aside", "Next", "sentence."])

    result = sentence_of(words, sentences("Hello world", "Next sentence"))

    assert result.tolist() == [-1, 0, 0, -1, 1, 1]


def test_sentence_of_ignores_case_apostrophe_and_surrounding_punctuation():
    words = np.array(["THERE'S", "a", "test."])

    result = sentence_of(words, sentences("there’s a test"))

    assert result.tolist() == [0, 0, 0]


def test_sentence_of_rejects_incomplete_sentence_alignment():
    with pytest.raises(ValueError, match="matched 2/3 sentence tokens"):
        sentence_of(np.array(["one", "two"]), sentences("one missing two"))


def test_surprise_classes_use_mean_plus_or_minus_sample_std():
    surprise = pd.Series([0.0, 4.0, 5.0, 6.0, 10.0])

    result = surprise_classes(surprise)

    # mean=5 and sample std=sqrt(13), so only 0 and 10 lie outside mean +/- std.
    assert result.tolist() == [0, 1, 1, 1, 2]


def test_tag_by_sentence_uses_spacy_tags_and_first_token_ownership():
    class Token:
        def __init__(self, idx, tag):
            self.idx = idx
            self.tag_ = tag

    class FakeNLP:
        def pipe(self, texts):
            assert list(texts) == ["Intro", "don't stop"]
            return iter([
                [Token(0, "NN")],
                [Token(0, "VB"), Token(2, "RB"), Token(6, "VB")],
            ])

    words = np.array(["Intro", "don't", "stop"])

    result = tag_by_sentence(words, np.array([-1, 0, 0]), nlp=FakeNLP())

    assert result.tolist() == ["NN", "VB", "VB"]


def test_brain_treebank_word_level_normalizes_timing_and_surprisal():
    transcript = pd.DataFrame(
        {
            "text": ["Hello", "world"],
            "start": [1.0, 1.5],
            "end": [1.4, 1.9],
            "sentence_idx": [7, 7],
            "gpt2_surprisal": [2.0, 3.0],
        }
    )

    result = brain_treebank_word_level(transcript)

    assert result["event_id"].tolist() == [0, 1]
    assert result["sentence_id"].tolist() == [7, 7]
    assert result["surprisal"].tolist() == [2.0, 3.0]
