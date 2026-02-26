import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import re
    import numpy as np
    import pandas as pd
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import librosa

    return AutoModelForSpeechSeq2Seq, AutoProcessor, librosa, mo, np, pd, re


@app.cell
def _(AutoModelForSpeechSeq2Seq, AutoProcessor):
    def init_model_processor(gpu=False):
        """Initializes the PSST model and processor with pre-trained weights."""
        processor = AutoProcessor.from_pretrained(
            "foundation_model_weights/models--NathanRoll--psst-medium-en/snapshots/a1112ed5890f654737cfe71eb27bffd8e1ffe363",
            local_files_only=True,
        )
        if gpu:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "foundation_model_weights/models--NathanRoll--psst-medium-en/snapshots/a1112ed5890f654737cfe71eb27bffd8e1ffe363",
                local_files_only=True,
            ).to("cuda:0")
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "foundation_model_weights/models--NathanRoll--psst-medium-en/snapshots/a1112ed5890f654737cfe71eb27bffd8e1ffe363",
                local_files_only=True,
            )
        return model, processor

    return (init_model_processor,)


@app.cell
def _(model, processor, re):
    def generate_transcription(audio, gpu=False):
        """Generate a transcription with IU boundaries from audio using PSST.

        Decodes without skipping all special tokens so <|IU_Boundary|> is
        preserved, then strips the Whisper control tokens via regex.
        """
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

        if gpu:
            input_features = inputs.input_features.cuda()
        else:
            input_features = inputs.input_features

        generated_ids = model.generate(input_features=input_features, max_length=448)

        # skip_special_tokens=False keeps <|IU_Boundary|> in the output
        raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Strip Whisper control tokens (e.g. <|startoftranscript|>, <|en|>,
        # <|transcribe|>, <|notimestamps|>, <|endoftext|>) but keep IU_Boundary
        transcription = re.sub(r"<\|(?!IU_Boundary)[^|]+\|>", "", raw).strip()

        # Fallback: the model sometimes emits !!!!! as a plain-text boundary marker
        transcription = transcription.replace("!!!!!", "<|IU_Boundary|>")

        return transcription

    return (generate_transcription,)


@app.cell
def _(re):
    def normalize_word(word):
        """Lowercase and strip punctuation except apostrophes."""
        return re.sub(r"[^a-z0-9']", "", word.lower())

    def parse_psst_output(text):
        """Parse PSST output into a list of token tuples.

        Returns a list of:
          ('word', word_string)   — a spoken word
          ('boundary', None)      — an intonation unit boundary
        """
        parts = re.split(r"<\|IU_Boundary\|>", text)
        result = []
        for i, part in enumerate(parts):
            for word in part.strip().split():
                if word:
                    result.append(("word", word))
            if i < len(parts) - 1:
                result.append(("boundary", None))
        return result

    return normalize_word, parse_psst_output


@app.cell
def _(normalize_word, np):
    def align_psst_to_transcript(psst_tokens, transcript_df):
        """Align PSST tokens to the reference transcript using DP sequence alignment.

        PSST recognizes words with minor variations vs the reference transcript
        (different punctuation, occasional misrecognition). This function uses a
        classic DP alignment to find the best mapping of PSST words to reference
        words, then uses the reference word timestamps to assign a time to each
        IU boundary.

        The timestamp for a boundary is the midpoint between the `end` of the
        last word before the boundary and the `start` of the first word after it.

        Args:
            psst_tokens: list of ('word', text) or ('boundary', None) tuples
            transcript_df: DataFrame with columns [word, start, end]

        Returns:
            List of dicts with keys: time, prev_word, next_word
        """
        # Separate words from boundaries; record after which word each boundary falls
        psst_words = []
        boundary_after = []  # boundary occurs after psst_words[i]

        word_idx = 0
        for token_type, token_val in psst_tokens:
            if token_type == "word":
                psst_words.append(normalize_word(token_val))
                word_idx += 1
            elif token_type == "boundary":
                boundary_after.append(word_idx - 1)

        if not psst_words:
            return []

        ref_words = [normalize_word(w) for w in transcript_df["word"]]
        n, m = len(psst_words), len(ref_words)

        MATCH = 2.0
        MISMATCH = -1.0
        GAP_REF = -0.3  # skip a reference word (filler / missed by PSST)
        GAP_PSST = -2.0  # skip a PSST word (hallucination)

        dp = np.full((n + 1, m + 1), -np.inf, dtype=np.float32)
        back = np.zeros(
            (n + 1, m + 1), dtype=np.int8
        )  # 0=diag, 1=skip_ref, 2=skip_psst

        # Allow alignment to start at any position in the reference transcript
        dp[0, :] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                s_match = dp[i - 1, j - 1] + (
                    MATCH if psst_words[i - 1] == ref_words[j - 1] else MISMATCH
                )
                s_skip_ref = dp[i, j - 1] + GAP_REF
                s_skip_psst = dp[i - 1, j] + GAP_PSST

                best = max(s_match, s_skip_ref, s_skip_psst)
                dp[i, j] = best
                if best == s_match:
                    back[i, j] = 0
                elif best == s_skip_ref:
                    back[i, j] = 1
                else:
                    back[i, j] = 2

        # Traceback from the best endpoint in the last PSST row
        j = int(np.argmax(dp[n]))
        i = n
        psst_to_ref = {}
        while i > 0 and j > 0:
            b = back[i, j]
            if b == 0:
                psst_to_ref[i - 1] = j - 1
                i -= 1
                j -= 1
            elif b == 1:
                j -= 1
            else:
                i -= 1

        # Convert boundary positions to timestamps
        boundaries = []
        for bp in boundary_after:
            prev_ref = psst_to_ref.get(bp)
            next_ref = psst_to_ref.get(bp + 1)

            if prev_ref is not None and next_ref is not None:
                prev_end = transcript_df.iloc[prev_ref]["end"]
                next_start = transcript_df.iloc[next_ref]["start"]
                # Midpoint of the gap; fall back to prev_end if words are adjacent
                time = (
                    (prev_end + next_start) / 2 if next_start > prev_end else prev_end
                )
                boundaries.append(
                    {
                        "time": time,
                        "prev_word": transcript_df.iloc[prev_ref]["word"],
                        "next_word": transcript_df.iloc[next_ref]["word"],
                    }
                )
            elif prev_ref is not None:
                boundaries.append(
                    {
                        "time": transcript_df.iloc[prev_ref]["end"],
                        "prev_word": transcript_df.iloc[prev_ref]["word"],
                        "next_word": "",
                    }
                )
            elif next_ref is not None:
                boundaries.append(
                    {
                        "time": transcript_df.iloc[next_ref]["start"],
                        "prev_word": "",
                        "next_word": transcript_df.iloc[next_ref]["word"],
                    }
                )

        return boundaries

    return (align_psst_to_transcript,)


@app.cell
def _(align_psst_to_transcript, generate_transcription, parse_psst_output, pd):
    def process_podcast(
        audio, transcript_df, chunk_duration=25.0, sample_rate=16000, gpu=False
    ):
        """Process the full podcast audio and produce IU boundary timestamps.

        Splits the audio into non-overlapping chunks, runs PSST on each chunk to
        detect intonation unit boundaries, then aligns the PSST word sequence to
        the reference transcript to recover the boundary timestamps.

        Args:
            audio: 1-D numpy array of audio at `sample_rate` Hz
            transcript_df: reference word-level transcript with columns [word, start, end]
            chunk_duration: seconds per chunk (Whisper supports up to 30s)
            sample_rate: audio sample rate (must match what the processor expects)
            gpu: whether to run inference on GPU

        Returns:
            DataFrame with columns: time, prev_word, next_word
        """
        chunk_samples = int(chunk_duration * sample_rate)
        all_boundaries = []
        seen_times = set()

        total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

        for chunk_idx, start_sample in enumerate(range(0, len(audio), chunk_samples)):
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk_start_time = start_sample / sample_rate
            chunk_end_time = end_sample / sample_rate

            print(
                f"  Chunk {chunk_idx + 1}/{total_chunks}: {chunk_start_time:.1f}s – {chunk_end_time:.1f}s"
            )

            # Narrow the reference transcript to this time window (small buffer for
            # words that straddle chunk boundaries)
            chunk_transcript = transcript_df[
                (transcript_df["end"] >= chunk_start_time - 0.5)
                & (transcript_df["start"] <= chunk_end_time + 0.5)
            ].reset_index(drop=True)

            if len(chunk_transcript) == 0:
                print("    No reference words in this chunk, skipping.")
                continue

            try:
                raw = generate_transcription(audio[start_sample:end_sample], gpu=gpu)
                print(f"    PSST output: {raw[:120]}{'…' if len(raw) > 120 else ''}")

                tokens = parse_psst_output(raw)
                boundaries = align_psst_to_transcript(tokens, chunk_transcript)
                print(f"    Found {len(boundaries)} IU boundaries")

                for b in boundaries:
                    t = round(b["time"], 4)
                    if t not in seen_times:
                        seen_times.add(t)
                        all_boundaries.append(b)

            except Exception as e:
                print(f"    Error: {e}")

        result = pd.DataFrame(all_boundaries).sort_values("time").reset_index(drop=True)
        return result

    return (process_podcast,)


@app.cell
def _(librosa):
    y, sr = librosa.load("data/stimuli/podcast.wav")
    audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return audio, sr, y


@app.cell
def _(init_model_processor):
    model, processor = init_model_processor(gpu=True)
    return model, processor


@app.cell
def _(pd):
    transcript_df = pd.read_csv("data/stimuli/podcast_transcript.csv")
    transcript_df
    return (transcript_df,)


@app.cell
def _(audio, process_podcast, transcript_df):
    iu_boundaries = process_podcast(audio, transcript_df, chunk_duration=25.0, gpu=True)
    iu_boundaries
    return (iu_boundaries,)


@app.cell
def _(iu_boundaries):
    out_path = "processed_data/iu_boundaries.csv"
    iu_boundaries.to_csv(out_path, index=False)
    print(f"Saved {len(iu_boundaries)} IU boundaries to {out_path}")
    return (out_path,)


if __name__ == "__main__":
    app.run()
