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
    ap.add_argument("--dataset", choices=("podcast", "brain_treebank"), default="podcast")
    ap.add_argument("--transcript", default="data/stimuli/gpt2-xl/transcript.tsv")
    ap.add_argument("--sentences", default="processed_data/all_sentences_podcast.csv")
    ap.add_argument("--out", default="processed_data/df_word_onset_with_pos_class.csv")
    return ap


def brain_treebank_word_level(transcript):
    """Normalize timed Brain Treebank transcript rows into canonical events."""
    required = {"text", "start", "end"}
    if not required <= set(transcript):
        raise ValueError(
            f"Brain Treebank transcript is missing {sorted(required - set(transcript))}"
        )
    source = transcript.copy()
    source["start"] = pd.to_numeric(source["start"], errors="coerce")
    source["end"] = pd.to_numeric(source["end"], errors="coerce")
    source = source.loc[source["start"].notna() & source["end"].notna()].copy()
    if "sentence_idx" in source:
        source["sentence_id"] = pd.to_numeric(source["sentence_idx"], errors="coerce")
        missing_sentence = source["sentence_id"].isna()
        if missing_sentence.any():
            if "sentence" in source:
                fallback = pd.factorize(
                    source.loc[missing_sentence, "sentence"].fillna("").astype(str),
                    sort=False,
                )[0]
                next_id = (
                    int(source["sentence_id"].dropna().max()) + 1
                    if source["sentence_id"].notna().any()
                    else 0
                )
                source.loc[missing_sentence, "sentence_id"] = fallback + next_id
            else:
                source.loc[missing_sentence, "sentence_id"] = -1
    elif "sentence" in source:
        source["sentence_id"] = pd.factorize(
            source["sentence"].fillna("").astype(str), sort=False
        )[0]
    else:
        source["sentence_id"] = 0

    source["word"] = source["text"].fillna("").astype(str).str.strip()
    if "sentence" in source:
        for _, group in source.groupby("sentence_id", sort=False):
            sentence_words = str(group["sentence"].iloc[0]).split()
            for position, index in enumerate(group.index):
                if not source.at[index, "word"] and position < len(sentence_words):
                    source.at[index, "word"] = sentence_words[position]
    source = source.loc[source["word"].ne("")].reset_index(drop=True)
    source["sentence_id"] = source["sentence_id"].astype(int)
    source["event_id"] = np.arange(len(source), dtype=int)
    if "surprisal" not in source and "gpt2_surprisal" in source:
        source["surprisal"] = source["gpt2_surprisal"]
    if "surprisal" not in source and "surprise" in source:
        source["surprisal"] = source["surprise"]
    return source


def main():
    args = build_parser().parse_args()

    separator = "\t" if args.transcript.endswith((".tsv", ".tab")) else ","
    t = pd.read_csv(args.transcript, sep=separator, keep_default_na=False)
    if args.dataset == "brain_treebank":
        df = brain_treebank_word_level(t)
        sid = df["sentence_id"].to_numpy(dtype=int)
    else:
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
    df["sentence_id"] = sid
    df["sentence_onset"] = False
    valid_rows = df.index[df["sentence_id"] >= 0]
    if len(valid_rows):
        first_rows = (
            df.loc[valid_rows].groupby("sentence_id", sort=False).head(1).index
        )
        df.loc[first_rows, "sentence_onset"] = True

    if "surprisal" not in df:
        if "surprise" in df:
            df["surprisal"] = df["surprise"]
        else:
            raise ValueError(
                "Transcript does not provide surprisal, gpt2_surprisal, or surprise"
            )
    df["surprisal"] = pd.to_numeric(df["surprisal"], errors="coerce")
    df["surprisal_class"] = surprise_classes(df["surprisal"])
    if "event_id" not in df:
        df["event_id"] = df.get(
            "word_idx", pd.Series(np.arange(len(df)), index=df.index)
        ).to_numpy()
    if "start" not in df:
        df["start"] = df["onset"]
    if "end" not in df:
        df["end"] = df["offset"]

    # Retain the legacy Podcast column while making the canonical name primary.
    if "surprise" in df:
        df["surprise_class"] = surprise_classes(df["surprise"])

    cols = [
        "event_id", "word_idx", "word", "start", "end", "onset", "offset",
        "pos", "is_content", "pos_class", "entropy", "true_prob", "surprise",
        "surprise_class", "surprisal", "surprisal_class", "sentence_idx",
        "sentence_id", "sentence_onset",
    ]
    df = df[[column for column in cols if column in df]]
    # Rows stay in transcript (word_idx) reading order, which is what the tagger needs. That
    # order is not monotonic in time: the transcript itself steps back at 19 word boundaries
    # where speakers overlap. Each row's onset is still that word's own onset, and the tasks
    # index rows independently, so this is a property of the stimulus, not a defect.
    n_rev = int((np.diff(df.start.values) < 0).sum())
    print(f"  {n_rev} onset reversals carried over from the transcript (overlapping speech)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    print("  pos_class:", df.pos_class.value_counts().sort_index().to_dict(),
          "(0=noun 1=verb 2=adj 3=adv 4=other)")
    print("  is_content:", df.is_content.value_counts().to_dict())
    print("  surprisal: min %.2f  median %.2f  max %.2f" %
          (df.surprisal.min(), df.surprisal.median(), df.surprisal.max()))
    print("  most surprising words:", list(df.nlargest(8, "surprisal").word))
    print("  least surprising words:", list(df.nsmallest(8, "surprisal").word))

if __name__ == "__main__":
    main()
