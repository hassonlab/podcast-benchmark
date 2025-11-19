import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    return GPT2LMHeadModel, GPT2TokenizerFast, np, pd, torch


@app.cell
def _(GPT2LMHeadModel, GPT2TokenizerFast):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@app.cell
def _(pd):
    df_sentence = pd.read_csv("processed_data/all_sentences_podcast.csv", index_col="index")
    stripped_sentences = df_sentence.all_sentence.str.strip()
    full_transcript = stripped_sentences.str.cat(sep=" ")
    full_transcript
    return (full_transcript,)


@app.cell
def _():
    # Maybe want a distinction between max words in a context window and max tokens.
    max_window = 32
    max_target_tokens = 10
    return max_target_tokens, max_window


@app.cell
def _(full_transcript, max_window):
    all_words = full_transcript.split()
    context_windows = []
    # Track bounds of target word for alignment with tokens.
    target_bounds = []
    targets = []
    current_char_pos = 0
    for i, word in enumerate(all_words):
        min_idx = max(0, i - max_window)
        context_windows.append(" ".join(all_words[min_idx:i]))
        targets.append(word)
    
        word_start = full_transcript.find(word, current_char_pos)
        word_end = word_start + len(word)
        target_bounds.append((word_start, word_end))
        current_char_pos = word_end
    context_windows, targets, target_bounds
    return context_windows, targets


@app.cell
def _(context_windows, max_target_tokens, max_window, np, targets, tokenizer):
    encoding_prev = tokenizer(context_windows, max_length=max_window, padding="max_length", return_offsets_mapping=True, return_tensors="pt", truncation=True)
    encoding_all = tokenizer(np.char.add(np.char.add(context_windows, " "), targets).tolist(), max_length=max_window + max_target_tokens, padding="max_length", return_offsets_mapping=True, return_tensors="pt", truncation=True)
    encoding_target = tokenizer(targets, max_length=max_target_tokens, padding="max_length", return_offsets_mapping=True, return_tensors="pt", truncation=True)
    return encoding_all, encoding_prev


@app.cell
def _(encoding_prev):
    encoding_prev["input_ids"].shape
    return


@app.cell
def _(encoding_all, model):
    output = model(input_ids=encoding_all["input_ids"][0:1], attention_mask=encoding_all["attention_mask"])
    return (output,)


@app.cell
def _(output):
    output.logits.shape
    return


@app.cell
def _(encoding_all, model, torch):
    predicted_text = model.generate(input_ids=(encoding_all["input_ids"][15][encoding_all["attention_mask"][15].bool()]).unsqueeze(0), 
                                    attention_mask=torch.tensor([[True] * len(encoding_all["input_ids"][15])]),
                                    max_new_tokens=10,
                                    do_sample=True,           # Enable sampling
                                    temperature=0.6,          # Add randomness
                                    top_k=50,                 # Consider top 50 tokens
                                    top_p=0.75,              # Nucleus sampling
                                    repetition_penalty=1.2,   # Penalize repeated tokens
                                   )
    return


@app.function
def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False


@app.cell
def _(model, tokenizer):
    new_tokens = ["<brain/>", "</brain>"]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return (new_tokens,)


@app.cell
def _(model):
    freeze_model(model)
    return


@app.cell
def _(model, new_tokens, tokenizer):
    for new_token in new_tokens:
        new_token_id = tokenizer.convert_tokens_to_ids(f"{new_token}")
        # if 'gpt2' in self.args['model_name']:
        model.transformer.wte.weight[new_token_id].requires_grad = True
        # elif 'llama' in self.args['model_name']: 
        #     self.model.model.embed_tokens.weight[new_token_id].requires_grad = True
        # elif 'huth' in self.args['model_name']:
        #     self.model.transformer.tokens_embed.weight[new_token_id].requires_grad = True
    return


@app.cell
def _(model):
    def words2embedding(input_ids):
            # if self.args['model_name'] in ['llama-7b', 'huth']:
            #     return self.model.get_input_embeddings()(input_ids)
            # else:
            #     if type(input_ids) == list:
            #         re = []
            #         for item in input_ids:
            #             re.append(self.model.transformer.wte(item))
            #         return re
            #     else:
            return model.transformer.wte(input_ids)
    return (words2embedding,)


@app.cell
def _(encoding_all, words2embedding):
    words2embedding(encoding_all["input_ids"][15]).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
