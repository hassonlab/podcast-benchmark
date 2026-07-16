"""Generate the per-word labels used by the POS, content and GPT-surprisal tasks.

Every column is derived from one source of truth, `data/stimuli/gpt2-xl/transcript.tsv`, the same
file `tasks/word_embedding.py` groups by `word_idx`. Word, onset and label therefore stay on the
same row by construction.

Two things are done better than a naive re-tag:

  * Words are tagged with spaCy **one sentence at a time**, by globally aligning the text in
    `processed_data/all_sentences_podcast.csv` to the transcript. Timestamp spans overlap and
    differ slightly through rounding, so text establishes linguistic sentence membership while
    omitted transcript words are tagged in separate contiguous runs. The transcript has only 2
    sentence-final punctuation marks in 5,136 words, so tagging it as a single blob gives the
    tagger no reliable sentence boundaries and it degrades badly.
  * `surprise` is the word-level surprisal in bits, summed over the word's BPE tokens
    (-sum log2 p), which is the probability of the whole word rather than of its first token.

Usage:
    python -m spacy download en_core_web_sm  # once per environment
    python scripts/build_word_labels.py
    python scripts/build_word_labels.py --out /path/to/word_labels.csv
"""

import argparse
import difflib
import os
import re
import string

import numpy as np
import pandas as pd

# Penn Treebank tag -> the 5 classes `pos_task` documents: Noun/Verb/Adj/Adv/Other.
POS_CLASS = {"NN": 0, "NNS": 0, "NNP": 0, "NNPS": 0,
             "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1,
             "JJ": 2, "JJR": 2, "JJS": 2,
             "RB": 3, "RBR": 3, "RBS": 3}
OTHER = 4

# Content words are nouns, verbs, adjectives and adverbs -- i.e. exactly the tags that get a
# pos_class below OTHER.
def is_content(tag):
    return int(POS_CLASS.get(tag, OTHER) != OTHER)


def surprise_classes(surprise):
    """Classify surprisal as low, typical or high using mean +/- one sample std."""
    mean = surprise.mean()
    std = surprise.std()
    return np.select(
        [surprise < mean - std, surprise > mean + std],
        [0, 2],
        default=1,
    ).astype(int)


def clean(word):
    """Strip surrounding punctuation for the tagger, keep internal apostrophes ("there's")."""
    w = str(word).strip().strip(string.punctuation.replace("'", "") + '"')
    return w if w else str(word).strip()


def word_level(transcript):
    """One row per word: onset from its first token, surprisal summed over all its tokens."""
    t = transcript.sort_values(["word_idx", "start"])
    g = t.groupby("word_idx", sort=True)
    df = g.agg(word=("word", "first"),
               onset=("start", "min"),
               offset=("end", "max"),
               entropy=("entropy", "first")).reset_index()
    # P(word | context) = prod over its BPE tokens  ->  surprisal = -sum log2 p, in bits.
    p = t.true_prob.clip(lower=1e-12)
    df["surprise"] = g.apply(lambda x: float(-np.log2(x.true_prob.clip(lower=1e-12)).sum())).values
    df["true_prob"] = g.apply(lambda x: float(x.true_prob.clip(lower=1e-12).prod())).values
    return df


def normalize_for_alignment(word):
    """Normalize superficial punctuation differences without changing word boundaries."""
    return re.sub(r"[^a-z0-9]+", "", str(word).lower().replace("’", "").replace("'", ""))


def sentence_of(words, sentences):
    """Assign transcript words to sentences by globally aligning their text.

    Sentence timestamps overlap and differ from transcript timestamps by small rounding errors,
    so they cannot uniquely establish linguistic sentence membership. The sentence CSV is a
    subsequence of the word transcript: it omits introductions, interjections and other material
    between annotated sentences. A global sequence alignment identifies those omissions while
    requiring every sentence token to match a transcript word in reading order.

    Returns -1 for transcript words that are not present in the sentence annotations.
    """
    transcript_tokens = [normalize_for_alignment(word) for word in words]
    sentence_tokens, sentence_owners = [], []
    for sentence_idx, sentence in enumerate(sentences.all_sentence):
        for word in str(sentence).split():
            token = normalize_for_alignment(word)
            if not token:
                raise ValueError(
                    f"sentence {sentence_idx} contains a token that is empty after normalization"
                )
            sentence_tokens.append(token)
            sentence_owners.append(sentence_idx)

    matcher = difflib.SequenceMatcher(
        a=transcript_tokens, b=sentence_tokens, autojunk=False
    )
    sid = np.full(len(words), -1, dtype=int)
    matched_sentence_tokens = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            sid[block.a + k] = sentence_owners[block.b + k]
        matched_sentence_tokens += block.size

    if matched_sentence_tokens != len(sentence_tokens):
        raise ValueError(
            "sentence text does not align completely with the transcript: "
            f"matched {matched_sentence_tokens}/{len(sentence_tokens)} sentence tokens"
        )
    return sid


def tag_by_sentence(words, sid, nlp=None, model="en_core_web_sm"):
    """POS-tag each sentence separately with spaCy.

    spaCy can split a transcript word into multiple tokens (`don't` -> `do` + `n't`), so
    tokens and transcript words are not necessarily 1:1. Character offsets retain the word
    that emitted each token, and a word takes the detailed Penn tag of its first token.
    """
    if nlp is None:
        import spacy

        try:
            nlp = spacy.load(model, disable=["ner"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {model!r} is not installed; run "
                f"`python -m spacy download {model}`"
            ) from exc

    tags = np.full(len(words), None, dtype=object)
    # Keep each contiguous run of unannotated transcript words separate. Combining all -1 words
    # would invent tagger context between unrelated parts of the podcast.
    tag_groups = sid.copy()
    outside = np.where(sid < 0)[0]
    if len(outside):
        run = np.cumsum(np.r_[True, np.diff(outside) != 1])
        tag_groups[outside] = -run

    groups = []
    for s in np.unique(tag_groups):
        idx = np.where(tag_groups == s)[0]
        parts = [clean(words[i]) for i in idx]
        starts = np.cumsum([0] + [len(part) + 1 for part in parts[:-1]])
        ends = starts + np.array([len(part) for part in parts])
        groups.append((idx, starts, ends, " ".join(parts)))

    for (idx, starts, ends, _), doc in zip(groups, nlp.pipe(g[-1] for g in groups)):
        seen = set()
        for token in doc:
            local_idx = int(np.searchsorted(ends, token.idx, side="right"))
            if local_idx >= len(idx) or token.idx < starts[local_idx]:
                continue
            owner = idx[local_idx]
            if owner not in seen:  # the word's first token carries its tag
                tags[owner] = token.tag_
                seen.add(owner)
    assert all(t is not None for t in tags), "every word must receive a tag"
    return tags


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="data/stimuli/gpt2-xl/transcript.tsv")
    ap.add_argument("--sentences", default="processed_data/all_sentences_podcast.csv")
    ap.add_argument("--out", default="processed_data/df_word_onset_with_pos_class.csv")
    return ap


def main():
    args = build_parser().parse_args()

    t = pd.read_csv(args.transcript, sep="\t")
    sents = pd.read_csv(args.sentences)
    df = word_level(t)
    print(f"{len(t)} tokens -> {len(df)} words")

    sid = sentence_of(df.word.values, sents)
    print(f"{(sid >= 0).sum()}/{len(df)} transcript words align to all "
          f"{len(sents)} sentence texts")

    df["pos"] = tag_by_sentence(df.word.values, sid)
    df["pos_class"] = [POS_CLASS.get(p, OTHER) for p in df.pos]
    df["is_content"] = [is_content(p) for p in df.pos]
    df["sentence_idx"] = sid

    # Match `gpt_surprise_multiclass_task`: low and high are values more than one
    # standard deviation below or above the mean; the middle class is the typical range.
    df["surprise_class"] = surprise_classes(df.surprise)

    cols = ["word_idx", "word", "onset", "offset", "pos", "is_content", "pos_class",
            "entropy", "true_prob", "surprise", "surprise_class", "sentence_idx"]
    df = df[cols]
    # Rows stay in transcript (word_idx) reading order, which is what the tagger needs. That
    # order is not monotonic in time: the transcript itself steps back at 19 word boundaries
    # where speakers overlap. Each row's onset is still that word's own onset, and the tasks
    # index rows independently, so this is a property of the stimulus, not a defect.
    n_rev = int((np.diff(df.onset.values) < 0).sum())
    print(f"  {n_rev} onset reversals carried over from the transcript (overlapping speech)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    print("  pos_class:", df.pos_class.value_counts().sort_index().to_dict(),
          "(0=noun 1=verb 2=adj 3=adv 4=other)")
    print("  is_content:", df.is_content.value_counts().to_dict())
    print("  surprise (bits): min %.2f  median %.2f  max %.2f" %
          (df.surprise.min(), df.surprise.median(), df.surprise.max()))
    print("  most surprising words:", list(df.nlargest(8, "surprise").word))
    print("  least surprising words:", list(df.nsmallest(8, "surprise").word))

if __name__ == "__main__":
    main()
