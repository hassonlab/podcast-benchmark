"""Rebuild the per-word label file that `pos_task`, `content_noncontent_task` and
`gpt_surprise_task` read.

The committed `processed_data/df_word_onset_with_pos_class.csv` has no generator script and
three defects (see `word_decoding/POS_CSV_MISALIGNMENT.md`):

  * `pos` carries the previous word's tag on ~8% of rows, and `is_content` / `pos_class` are
    deterministic functions of `pos`, so they inherit the shift exactly.
  * `surprise` *rises* with the word's probability -- it is an anti-surprisal. Its three most
    "surprising" words are `of`, `the`, `you`.
  * `entropy` correlates only r=0.75 with the transcript it supposedly came from.

`word`/`onset` are the only columns that were right, which is why the bug is invisible in the
neural windows and only corrupts the targets.

This script derives every column from one source of truth, `data/stimuli/gpt2-xl/transcript.tsv`,
the same file `tasks/word_embedding.py` groups by `word_idx`. Word, onset and label therefore stay
on the same row by construction rather than by a positional merge.

Two things are done better than a naive re-tag:

  * Words are tagged **one sentence at a time**, using the spans in
    `processed_data/all_sentences_podcast.csv`. The transcript has 2 sentence-final punctuation
    marks in 5,136 words, so tagging it as a single blob gives the tagger no sentence boundaries
    to work with and it degrades badly.
  * `surprise` is the word-level surprisal in bits, summed over the word's BPE tokens
    (-sum log2 p), which is the probability of the whole word rather than of its first token.

Usage:
    python scripts/build_word_labels.py                    # -> processed_data/df_word_onset_with_pos_class_fixed.csv
    python scripts/build_word_labels.py --validate         # also diff against the old file
    python scripts/build_word_labels.py --out processed_data/df_word_onset_with_pos_class.csv
"""

import argparse
import os
import string

import numpy as np
import pandas as pd

# Penn Treebank tag -> the 5 classes `pos_task` documents: Noun/Verb/Adj/Adv/Other.
# Reproduces the mapping implied by the old file, whose pos -> pos_class table was 100% pure.
POS_CLASS = {"NN": 0, "NNS": 0, "NNP": 0, "NNPS": 0,
             "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1,
             "JJ": 2, "JJR": 2, "JJS": 2,
             "RB": 3, "RBR": 3, "RBS": 3}
OTHER = 4

# Content words are nouns, verbs, adjectives and adverbs -- i.e. exactly the tags that get a
# pos_class below OTHER. Matches the old file's pos -> is_content table.
def is_content(tag):
    return int(POS_CLASS.get(tag, OTHER) != OTHER)


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


def sentence_of(onsets, sentences):
    """Assign each word to a sentence by onset time; -1 when it falls in no span."""
    sid = np.full(len(onsets), -1, dtype=int)
    for i, (a, b) in enumerate(zip(sentences.sentence_onset, sentences.sentence_offset)):
        sid[(onsets >= a) & (onsets <= b)] = i
    return sid


def tag_by_sentence(words, sid):
    """POS-tag each sentence separately, so the tagger sees real sentence boundaries.

    The tokenizer splits clitics (`don't` -> `do` + `n't`), so tokens and words are NOT 1:1.
    The original notebook zipped `tagged[k]` onto `words[k]` anyway and shifted 11% of the
    corpus. Here every token remembers which word emitted it (`owner`), and a word takes the
    tag of its own first token -- alignment holds by construction no matter how the tokenizer
    behaves.
    """
    import nltk
    from nltk.tokenize import TreebankWordTokenizer

    tk = TreebankWordTokenizer()
    tags = np.empty(len(words), dtype=object)
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]  # words outside any sentence span are tagged as one run
        toks, owner = [], []
        for i in idx:
            sub = tk.tokenize(clean(words[i])) or [clean(words[i])]
            toks += sub
            owner += [i] * len(sub)
        seen = set()
        for (_, tg), o in zip(nltk.pos_tag(toks), owner):
            if o not in seen:  # the word's first token carries its tag
                tags[o] = tg
                seen.add(o)
    assert all(t is not None for t in tags), "every word must receive a tag"
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="data/stimuli/gpt2-xl/transcript.tsv")
    ap.add_argument("--sentences", default="processed_data/all_sentences_podcast.csv")
    ap.add_argument("--old", default="processed_data/df_word_onset_with_pos_class.csv")
    ap.add_argument("--out", default="processed_data/df_word_onset_with_pos_class_fixed.csv")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    t = pd.read_csv(args.transcript, sep="\t")
    sents = pd.read_csv(args.sentences)
    df = word_level(t)
    print(f"{len(t)} tokens -> {len(df)} words")

    sid = sentence_of(df.onset.values, sents)
    print(f"{(sid >= 0).sum()}/{len(df)} words fall inside one of {len(sents)} sentence spans")

    df["pos"] = tag_by_sentence(df.word.values, sid)
    df["pos_class"] = [POS_CLASS.get(p, OTHER) for p in df.pos]
    df["is_content"] = [is_content(p) for p in df.pos]
    df["sentence_idx"] = sid

    # Terciles of the true surprisal. 0 = least surprising, 2 = most surprising.
    # NOTE: the old file's classes ran the other way, because its `surprise` was inverted.
    df["surprise_class"] = pd.qcut(df.surprise, 3, labels=[0, 1, 2]).astype(int)

    cols = ["word_idx", "word", "onset", "offset", "pos", "is_content", "pos_class",
            "entropy", "true_prob", "surprise", "surprise_class", "sentence_idx"]
    df = df[cols]
    # Rows stay in transcript (word_idx) reading order, which is what the tagger needs. That
    # order is not monotonic in time: the transcript itself steps back at 19 word boundaries
    # where speakers overlap. Each row's onset is still that word's own onset, and the tasks
    # index rows independently, so this is a property of the stimulus, not a defect.
    n_rev = int((np.diff(df.onset.values) < 0).sum())
    print(f"  {n_rev} onset reversals carried over from the transcript (overlapping speech)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    print("  pos_class:", df.pos_class.value_counts().sort_index().to_dict(),
          "(0=noun 1=verb 2=adj 3=adv 4=other)")
    print("  is_content:", df.is_content.value_counts().to_dict())
    print("  surprise (bits): min %.2f  median %.2f  max %.2f" %
          (df.surprise.min(), df.surprise.median(), df.surprise.max()))
    print("  most surprising words:", list(df.nlargest(8, "surprise").word))
    print("  least surprising words:", list(df.nsmallest(8, "surprise").word))

    if args.validate:
        validate(df, pd.read_csv(args.old))


def validate(new, old):
    import difflib
    from scipy.stats import spearmanr

    print("\n=== old vs new ===")
    A = [str(x).strip() for x in old.word]
    B = [str(x).strip() for x in new.word]
    sm = difflib.SequenceMatcher(a=A, b=B, autojunk=False)
    rowmap = {}
    for bl in sm.get_matching_blocks():
        for k in range(bl.size):
            rowmap[bl.a + k] = bl.b + k
    ai = np.array(sorted(rowmap))
    bi = np.array([rowmap[i] for i in ai])
    print(f"{len(ai)}/{len(old)} old rows align to a new row by word sequence")

    o, n = old.iloc[ai], new.iloc[bi]
    print(f"  onset agrees within 5 ms      : {(np.abs(o.onset.values - n.onset.values) < 0.005).mean():.1%}")
    print(f"  pos label changed             : {(o.pos.values != n.pos.values).mean():.1%}")
    print(f"  is_content label changed      : {(o.is_content.values != n.is_content.values).mean():.1%}")
    print(f"  pos_class label changed       : {(o.pos_class.values != n.pos_class.values).mean():.1%}")
    r = spearmanr(o.surprise.values, n.surprise.values).correlation
    print(f"  spearman(old.surprise, new.surprise) = {r:+.3f}   <- negative means the old column was inverted")
    print(f"  surprise_class label changed  : {(o.surprise_class.values != n.surprise_class.values).mean():.1%}")


if __name__ == "__main__":
    main()
