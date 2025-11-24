import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""# LLM Decoding""")
    return


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import mne
    from typing import Optional
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    return (
        DataLoader,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        Optional,
        TensorDataset,
        mne,
        mo,
        nn,
        np,
        optim,
        os,
        pd,
        torch,
    )


@app.cell
def _(GPT2LMHeadModel, GPT2TokenizerFast, tokenizer):
    test_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="gpt2-cache")
    test_model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="gpt2-cache")
    test_model.to("cuda")
    test_model.config.pad_token_id = tokenizer.eos_token_id
    test_tokenizer.pad_token = test_tokenizer.eos_token
    # test_tokenizer.padding_side = 'left'
    return test_model, test_tokenizer


@app.cell
def _(mo):
    mo.md("""## Play with model setup""")
    return


@app.cell
def _(pd):
    # df_word = pd.read_csv(os.path.join(data_params.data_root, "stimuli/podcast_transcript.csv"))
    df_word = pd.read_csv("processed_data/punctuated_transcript.csv")
    stripped_words = df_word.word.str.strip()
    full_transcript = stripped_words.str.cat(sep=" ")
    full_transcript
    return df_word, full_transcript


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
def _(
    context_windows,
    max_target_tokens,
    max_window,
    np,
    targets,
    test_tokenizer,
):
    encoding_prev = test_tokenizer(context_windows, max_length=max_window, padding="max_length", return_offsets_mapping=True, return_tensors="pt", truncation=True)
    encoding_all = test_tokenizer(np.char.add(np.char.add(context_windows, " "), targets).tolist(), max_length=max_window + max_target_tokens, padding="max_length", return_offsets_mapping=True, return_tensors="pt", truncation=True)
    encoding_target = test_tokenizer(targets, max_length=max_target_tokens, padding="max_length", return_offsets_mapping=True, return_tensors="pt", truncation=True)
    return encoding_all, encoding_prev, encoding_target


@app.cell
def _(encoding_all, encoding_prev, encoding_target):
    prev_input_ids, all_input_ids, target_input_ids = encoding_prev["input_ids"].to("cuda"), encoding_all["input_ids"].to("cuda"), encoding_target["input_ids"].to("cuda")
    prev_attention_mask, all_attention_mask, target_attention_mask = encoding_prev["attention_mask"].to("cuda"), encoding_all["attention_mask"].to("cuda"), encoding_target["input_ids"].to("cuda")
    return (
        all_attention_mask,
        all_input_ids,
        prev_attention_mask,
        prev_input_ids,
    )


@app.cell
def _(prev_attention_mask, prev_input_ids):
    prev_input_ids[0:4], prev_attention_mask[0:4]
    return


@app.cell
def _(all_attention_mask, all_input_ids, test_model):
    output = test_model(input_ids=all_input_ids[0:1], attention_mask=all_attention_mask[0:1])
    return (output,)


@app.cell
def _(output):
    output.logits.shape
    return


@app.cell
def _(torch):
    def pad2left(content_prev, content_prev_mask):
        padding_counts = (content_prev_mask == 1).sum(dim=1)
        # initialize new tensors for fill
        front_padded_input_embeds = torch.zeros_like(content_prev)
        front_padded_mask = torch.zeros_like(content_prev_mask)

        for i in range(content_prev.size(0)):  # go through each sample
            # calculate the number of positions we need to move
            shift = padding_counts[i].item()
            # fill the input_embeds and the mask
            front_padded_input_embeds[i, content_prev.size(1) - shift:] = content_prev[i, :shift]
            front_padded_input_embeds[i, :content_prev.size(1) - shift] = content_prev[i, shift:]
            front_padded_mask[i, content_prev.size(1) - shift:] = content_prev_mask[i, :shift]
        return front_padded_input_embeds, front_padded_mask
    return (pad2left,)


@app.cell
def _(
    pad2left,
    prev_attention_mask,
    prev_input_ids,
    test_model,
    test_tokenizer,
):
    left_padded_prev_input_ids, left_padded_prev_attention_masks = pad2left(prev_input_ids[0:32], prev_attention_mask[0:32])
    predicted_text = test_model.generate(input_ids=left_padded_prev_input_ids,
                                    attention_mask=left_padded_prev_attention_masks,
                                    max_new_tokens=10,
                                    pad_token_id=test_tokenizer.eos_token_id,
                                    do_sample=True,           # Enable sampling
                                    temperature=0.6,          # Add randomness
                                    top_k=50,                 # Consider top 50 tokens
                                    top_p=0.75,              # Nucleus sampling
                                    repetition_penalty=1.2,   # Penalize repeated tokens
                                   )
    return (predicted_text,)


@app.cell
def _(
    full_transcript,
    predicted_text,
    prev_attention_mask,
    prev_input_ids,
    test_tokenizer,
):
    def _():
        for i, seq in enumerate(predicted_text):
            # Get only the non-padding part
            mask = prev_attention_mask[i]
            original_length = mask.sum().item()

            original = test_tokenizer.decode(prev_input_ids[i][mask.bool()], skip_special_tokens=True)
            generated = test_tokenizer.decode(seq, skip_special_tokens=True)

            print(f"Sequence {i}:")
            print(f"  Original: {original}")
            print(f"  Generated: {generated}")
            print(f"  Groundtruth: {' '.join(full_transcript.split()[:len(generated.split())])}")
        return print()


    _()
    return


@app.cell
def _(mo):
    mo.md("""## Setup Brain Data""")
    return


@app.function
def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False


@app.cell
def _(GPT2LMHeadModel, GPT2TokenizerFast):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="gpt2-cache")
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="gpt2-cache")
    model.to("cuda")
    model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'

    new_tokens = ["<brain/>", "</brain>"]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return model, new_tokens, tokenizer


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
def _(all_input_ids, words2embedding):
    words2embedding(all_input_ids[15])
    return


@app.cell
def _():
    from utils import data_utils
    from utils import decoding_utils
    from core.config import DataParams
    return DataParams, data_utils


@app.cell
def _(DataParams):
    data_params = DataParams(
      data_root = "data",
      word_column = "word",
      # electrode_file_path = "processed_data/all_subject_sig.csv",
      subject_ids = [9],
      channel_reg_ex = "^G([1-9]|[1-5][0-9]|6[0-4])$",
      window_width = 1.0,
      preprocessing_fn_name = "window_average_neural_data",
      # neural_conv_decoder specific config
      preprocessor_params = {
        "num_average_samples": 12
      }
    )
    return (data_params,)


@app.cell
def _(data_params, data_utils):
    # TODO: 
    # 1. Add brain data
    # 2. Add brain encoder
    # 3. Implement training loop
    if data_params.electrode_file_path:
        subject_id_map = data_utils.read_subject_mapping(
            "data/participants.tsv", delimiter="\t"
        )
        subject_electrode_map = data_utils.read_electrode_file(
            data_params.electrode_file_path,
            subject_mapping=subject_id_map,
        )
        data_params.subject_ids = list(subject_electrode_map.keys())

    raws = data_utils.load_raws(data_params)
    return (raws,)


@app.cell
def _(data_params, max_target_tokens, max_window, np, os, pd, tokenizer):
    # Want to make aligned start and word columns in a dataframe
    def get_llm_decoding_data():
        # LLM data
        df_word = pd.read_csv(os.path.join(data_params.data_root, "stimuli/podcast_transcript.csv"))
        stripped_words = df_word.word.str.strip()
        full_transcript = stripped_words.str.cat(sep=" ")

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

        encoding_prev = tokenizer(context_windows, max_length=max_window, padding="max_length", return_offsets_mapping=True, return_tensors="np", truncation=True)
        encoding_all = tokenizer(np.char.add(np.char.add(context_windows, " "), targets).tolist(), max_length=max_window + max_target_tokens, padding="max_length", return_offsets_mapping=True, return_tensors="np", truncation=True)
        encoding_target = tokenizer(targets, max_length=max_target_tokens, padding="max_length", return_offsets_mapping=True, return_tensors="np", truncation=True)

        return pd.DataFrame({
            "prev_input_ids": list(encoding_prev["input_ids"]),
            "prev_attention_mask": list(encoding_prev["attention_mask"]),
            "all_input_ids": list(encoding_all["input_ids"]),
            "all_attention_mask": list(encoding_all["attention_mask"]),
            "target_input_ids": list(encoding_target["input_ids"]),
            "target_attention_mask": list(encoding_target["attention_mask"]),
            "word": df_word.word,
            "start": df_word.start,
            "end": df_word.end,
        })

        # prev_input_ids, all_input_ids, target_input_ids = encoding_prev["input_ids"].to("cuda"), encoding_all["input_ids"].to("cuda"), encoding_target["input_ids"].to("cuda")
        # prev_attention_mask, all_attention_mask, target_attention_mask = encoding_prev["attention_mask"].to("cuda"), encoding_all["attention_mask"].to("cuda"), encoding_target["input_ids"].to("cuda")
    return (get_llm_decoding_data,)


@app.cell
def _(get_llm_decoding_data):
    llm_decoding_data_df = get_llm_decoding_data()
    return (llm_decoding_data_df,)


@app.cell
def _(mo):
    mo.md("""## Training a model""")
    return


@app.cell
def _():
    from models.shared_preprocessors import window_average_neural_data
    from models.neural_conv_decoder.decoder_model import EnsemblePitomModel
    return EnsemblePitomModel, window_average_neural_data


@app.cell
def _(Optional, mne, np, pd):
    def _apply_preprocessing(data, preprocessing_fns, preprocessor_params):
        """Apply a list of preprocessing functions to data.

        Args:
            data: numpy array to preprocess
            preprocessing_fns: list of preprocessing functions to apply in order
            preprocessor_params: parameters to pass to preprocessing functions (dict or list of dicts)

        Returns:
            Preprocessed data array
        """
        if not preprocessing_fns:
            return data

        for i, preprocessing_fn in enumerate(preprocessing_fns):
            if preprocessor_params and isinstance(preprocessor_params, list):
                params = preprocessor_params[i] if i < len(preprocessor_params) else None
            else:
                params = preprocessor_params
            data = preprocessing_fn(data, params)

        return data

    def get_data(
        lag,
        raws: list[mne.io.Raw],
        task_df: pd.DataFrame,
        window_width: float,
        preprocessing_fns=None,
        preprocessor_params: dict = None,
        word_column: Optional[str] = None,
    ):
        """Gather data for every row in task_df from raw.

        Args:
            lag: the lag relative to each word onset to gather data around
            raws: list of mne.Raw object holding electrode data
            task_df: dataframe containing columns start, target, and optionally word_column
            window_width: the width of the window which is gathered around each word onset + lag
            preprocessing_fns: functions to apply to epoch data in order of calling.
                Should have contract:
                    preprocessing_fn(data: np.array of shape [num_words, num_electrodes, timesteps],
                                    preprocessor_params)  -> array of shape [num_words, ...]
            word_column: If provided, will return the column of words specified here.
        """
        datas = []
        for raw in raws:
            # Calculate time bounds for filtering
            tmin = lag / 1000 - window_width / 2
            tmax = lag / 1000 + window_width / 2 - 2e-3
            data_duration = raw.times[-1]  # End time of the data

            # Filter out events where the time window falls outside data bounds
            valid_mask = (task_df.start + tmin >= 0) & (
                task_df.start + tmax <= data_duration
            )
            task_df_valid = task_df[valid_mask].reset_index(drop=True)

            if len(task_df_valid) == 0:
                # No valid events for this raw, skip
                continue

            events = np.zeros((len(task_df_valid), 3), dtype=int)
            events[:, 0] = (task_df_valid.start * raw.info["sfreq"]).astype(int)

            epochs = mne.Epochs(
                raw,
                events,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                proj=False,
                event_id=None,
                preload=True,
                on_missing="ignore",
                event_repeated="merge",
                verbose="ERROR",
            )

            data = epochs.get_data(copy=False)

            selected_all_input_ids = task_df_valid.all_input_ids
            selected_all_mask = task_df_valid.all_attention_mask
            selected_target_input_ids = task_df_valid.target_input_ids
            selected_target_mask = task_df_valid.target_attention_mask

            # TODO: Clean this up so we don't need to pass around this potentially None variable.
            if word_column:
                selected_words = task_df_valid[word_column].to_numpy()[epochs.selection]
            else:
                selected_words = None

            # Make sure the number of samples match
            assert data.shape[0] == selected_all_input_ids.shape[0], "Sample counts don't match"
            if selected_words is not None:
                assert data.shape[0] == selected_words.shape[0], "Words don't match"

            datas.append(data)

        if len(datas) == 0:
            raise ValueError("No valid events found within data time bounds")

        datas = np.concatenate(datas, axis=1)

        datas = _apply_preprocessing(datas, preprocessing_fns, preprocessor_params)

        return datas, (np.vstack(selected_all_input_ids), np.vstack(selected_all_mask), np.vstack(selected_target_input_ids), np.vstack(selected_target_mask)), selected_words
    return (get_data,)


@app.cell
def _(
    data_params,
    get_data,
    llm_decoding_data_df,
    raws,
    torch,
    window_average_neural_data,
):
    neural_data, (lag_all_input_ids, lag_all_attention_masks, lag_target_input_ids, lag_target_attention_masks), _ = get_data(0, raws, llm_decoding_data_df, window_width=data_params.window_width, preprocessing_fns=[window_average_neural_data], preprocessor_params=data_params.preprocessor_params)
    neural_data = torch.FloatTensor(neural_data)
    lag_all_input_ids = torch.LongTensor(lag_all_input_ids)
    lag_all_attention_masks = torch.BoolTensor(lag_all_attention_masks)
    lag_target_attention_masks = torch.BoolTensor(lag_target_attention_masks)
    return (
        lag_all_attention_masks,
        lag_all_input_ids,
        lag_target_attention_masks,
        lag_target_input_ids,
        neural_data,
    )


@app.cell
def _(neural_data):
    neural_data.shape
    return


@app.cell
def _(EnsemblePitomModel, neural_data):
    encoder_model = EnsemblePitomModel(conv_filters=128,
                                      dropout=0.2,
                                      num_models=1,
                                      output_dim=768,
                                      input_channels=neural_data.shape[1]
                                    ).to("cuda")
    encoder_models = [encoder_model]
    return encoder_model, encoder_models


@app.cell
def _(mo):
    mo.md("""### Get Data Through Model""")
    return


@app.cell
def _(encoder_model, lag_all_input_ids, neural_data, words2embedding):
    neural_embedding = encoder_model(neural_data[0:2].to("cuda")).unsqueeze(1)
    input_embeddings = words2embedding(lag_all_input_ids[0:2].to("cuda"))
    input_embeddings.shape
    return input_embeddings, neural_embedding


@app.cell
def _(
    input_embeddings,
    lag_all_attention_masks,
    neural_embedding,
    tokenizer,
    torch,
    words2embedding,
):
    brain_sep_embeddings = words2embedding(tokenizer(["<brain/>", "</brain>"], return_tensors="pt")["input_ids"].to("cuda"))
    # Expand to match batching
    first_sep_embedding = brain_sep_embeddings[0].unsqueeze(0).expand(neural_embedding.shape[0], 1, brain_sep_embeddings.shape[-1])
    last_sep_embedding = brain_sep_embeddings[1].unsqueeze(0).expand(neural_embedding.shape[0], 1, brain_sep_embeddings.shape[-1])
    brain_prompt = torch.concat([first_sep_embedding, neural_embedding, last_sep_embedding], dim=1)
    prompt = torch.concat([brain_prompt, input_embeddings], dim=1)
    prompt_attention_mask = torch.concat([torch.ones(brain_prompt.shape[:-1]).to("cuda"), lag_all_attention_masks[0:2].to("cuda")], dim=1)
    prompt.shape, prompt_attention_mask.shape
    return brain_sep_embeddings, prompt, prompt_attention_mask


@app.cell
def _(model, prompt, prompt_attention_mask):
    brain_pred_output = model(inputs_embeds=prompt, attention_mask=prompt_attention_mask)
    brain_pred_output.logits.shape
    return


@app.cell
def _(mo):
    mo.md("""### Pre-Train Encoder Model""")
    return


@app.cell
def _():
    from utils.fold_utils import get_sequential_folds
    from metrics.classification_metrics import perplexity
    from metrics.embedding_metrics import compute_nll_contextual, cosine_similarity
    return (
        compute_nll_contextual,
        cosine_similarity,
        get_sequential_folds,
        perplexity,
    )


@app.cell
def _(
    DataLoader,
    TensorDataset,
    compute_nll_contextual,
    cosine_similarity,
    encoder_model,
    encoder_models,
    get_sequential_folds,
    lag_all_input_ids,
    model,
    neural_data,
    nn,
    optim,
    torch,
    words2embedding,
):
    # For each neural data, generate embeddings, then compare to mean of corresponding text embeddings

    # Assuming you have:
    # - model: your neural network
    # - train_loader: DataLoader for training data
    # - val_loader: DataLoader for validation data (optional)

    # Input dataset for pretraining is brain data. Target is average of 
    X = neural_data
    # Can also make this lazy load embeddings if this is too big. Or can 
    Y = words2embedding(lag_all_input_ids.to("cuda")).mean(dim=1).unsqueeze(1).tile(1, len(encoder_models), 1)

    folds = get_sequential_folds(X, num_folds=5)
    (tr_idx, val_idx, test_idx) = folds[0]
    train_dataset = TensorDataset(X[tr_idx], Y[tr_idx])
    val_dataset = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parameters = []
    parameters += model.parameters()
    if type(encoder_models) == list:
        for k in range(len(encoder_models)):
            parameters += encoder_models[k].parameters()
    criterion = nn.MSELoss()  # or your loss function
    pretrain_optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=1e-4, weight_decay=0)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, targ) in enumerate(train_loader):
            # Move data to device
            data, targ = data.to(device), targ.to(device)

            # Forward pass
            outputs = encoder_model(data).unsqueeze(1)
            loss = criterion(outputs, targ)

            # Backward pass and optimization
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

            train_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase (optional)
        model.eval()
        val_loss = 0.0
        val_cosine_sim = 0.0
        val_nll_embedding = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targ in val_loader:
                data, targ = data.to(device), targ.to(device)
                outputs = encoder_model(data).unsqueeze(1)
                loss = criterion(outputs, targ)
                cosine_sim = cosine_similarity(outputs, targ)
                nll_embedding = compute_nll_contextual(outputs.squeeze(), targ.squeeze())
                val_loss += loss.item()
                val_cosine_sim += cosine_sim.item()
                val_nll_embedding += nll_embedding.item()


        avg_val_loss = val_loss / len(val_loader)
        avg_val_cosine = val_cosine_sim / len(val_loader)
        avg_val_nll_embedding = val_nll_embedding / len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Cosine: {avg_val_cosine:.4f}, '
              f'Val NLL: {avg_val_nll_embedding:.4f}'
             )
    return (X,)


@app.cell
def _(mo):
    mo.md("""### Finetune Encoder and positional embeddings""")
    return


@app.cell
def _(
    brain_sep_embeddings,
    nn,
    pad2left,
    test_tokenizer,
    tokenizer,
    torch,
    words2embedding,
):
    class BrainPromptModel(nn.Module):
        def __init__(self, lm_model, encoder_model):
            super().__init__()
            self.lm_model = lm_model
            self.encoder_model = encoder_model
            brain_sep_embeddings = words2embedding(tokenizer(["<brain/>", "</brain>"], return_tensors="pt")["input_ids"].to("cuda"))
            self.first_sep_embedding = brain_sep_embeddings[0].unsqueeze(0)
            self.last_sep_embedding = brain_sep_embeddings[1].unsqueeze(0)

        def convert_to_embeddings(self, neural_data, input_ids, attention_mask):
            neural_embedding = self.encoder_model(neural_data).unsqueeze(1)
            input_embeddings = words2embedding(input_ids)
            first_sep_embedding = self.first_sep_embedding.expand(neural_embedding.shape[0], 1, brain_sep_embeddings.shape[-1])
            last_sep_embedding = self.last_sep_embedding.expand(neural_embedding.shape[0], 1, brain_sep_embeddings.shape[-1])
            brain_prompt = torch.concat([first_sep_embedding, neural_embedding, last_sep_embedding], dim=1)
            prompt = torch.concat([brain_prompt, input_embeddings], dim=1)
            prompt_attention_mask = torch.concat([torch.ones(brain_prompt.shape[:-1]).to("cuda"), attention_mask], dim=1)
            return prompt, prompt_attention_mask

        def forward(self, neural_data, input_ids, attention_mask):
            prompt, prompt_attention_mask = self.convert_to_embeddings(neural_data, input_ids, attention_mask)
            return self.lm_model(inputs_embeds=prompt, attention_mask=prompt_attention_mask), prompt_attention_mask

        def generate(self, neural_data, input_ids, attention_mask):
            prompt, prompt_attention_mask = self.convert_to_embeddings(neural_data, input_ids, attention_mask)
            prompt, prompt_attention_mask = pad2left(prompt, prompt_attention_mask)
            return self.lm_model.generate(inputs_embeds=prompt, attention_mask=prompt_attention_mask, 
                                    max_new_tokens=10,
                                    pad_token_id=test_tokenizer.eos_token_id,
                                    do_sample=True,           # Enable sampling
                                    temperature=0.6,          # Add randomness
                                    top_k=50,                 # Consider top 50 tokens
                                    top_p=0.75,              # Nucleus sampling
                                    repetition_penalty=1.2,   # Penalize repeated tokens
                                         )
    return (BrainPromptModel,)


@app.cell
def _(BrainPromptModel, encoder_model, model):
    brain_prompt_model = BrainPromptModel(model.to("cuda"), encoder_model.to("cuda"))
    # brain_prompt_model(neural_data[0:2].to("cuda"), lag_all_input_ids[0:2].to("cuda"), lag_all_attention_masks[0:2].to("cuda"))
    return (brain_prompt_model,)


@app.cell
def _(
    brain_prompt_model,
    lag_all_attention_masks,
    lag_all_input_ids,
    neural_data,
):
    brain_predicted_text = brain_prompt_model.generate(neural_data[0:32].to("cuda"), lag_all_input_ids[0:32, ].to("cuda"), lag_all_attention_masks[0:32].to("cuda"))
    return (brain_predicted_text,)


@app.cell
def _():
    # torch.save(brain_prompt_model, "trained_prompt_model.pt")
    return


@app.cell
def _(
    brain_prompt_model,
    get_target_preds,
    lag_all_attention_masks,
    lag_all_input_ids,
    lag_target_attention_masks,
    lag_target_input_ids,
    neural_data,
    torch,
):
    bot_idx, max_idx = 0, 5
    forward_neur = neural_data[bot_idx:max_idx].to("cuda")
    forward_ids = lag_all_input_ids[bot_idx:max_idx].to("cuda")
    forward_mask = lag_all_attention_masks[bot_idx:max_idx].to("cuda")
    forward_preds, forward_att_mask = brain_prompt_model(forward_neur, forward_ids, forward_mask)
    pred_probs = torch.softmax(get_target_preds(forward_preds, forward_att_mask[bot_idx:max_idx], lag_target_attention_masks[bot_idx:max_idx]), dim=1)[lag_target_input_ids[bot_idx:max_idx][lag_target_attention_masks[bot_idx:max_idx]]].to("cpu")
    pred_probs
    return


@app.cell
def _(
    brain_predicted_text,
    full_transcript,
    prev_attention_mask,
    prev_input_ids,
    tokenizer,
):
    def _():
        for i, seq in enumerate(brain_predicted_text):
            # Get only the non-padding part
            mask = prev_attention_mask[i]
            original_length = mask.sum().item()

            generated = tokenizer.decode(seq, skip_special_tokens=True)
            original = tokenizer.decode(prev_input_ids[i][mask.bool()], skip_special_tokens=True)

            print(f"Sequence {i}:")
            print(f"  Original: {original}")
            print(f"  Generated: {generated}")
            print(f"  Groundtruth: {' '.join(full_transcript.split()[:len(generated.split())])}")
        return print()


    _()
    return


@app.cell
def _():
    from tqdm.auto import tqdm 
    return (tqdm,)


@app.cell
def _(
    DataLoader,
    X,
    brain_prompt_model,
    get_sequential_folds,
    lag_all_attention_masks,
    lag_all_input_ids,
    lag_target_attention_masks,
    lag_target_input_ids,
    neural_data,
    nn,
    optim,
    perplexity,
    torch,
    tqdm,
):
    class MultiInputDataset(torch.utils.data.Dataset):
        def __init__(self, neural_data, input_ids, attention_mask, target_ids, target_attention_mask):
            self.neural_data = neural_data
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.target_ids = target_ids
            self.target_attention_mask = target_attention_mask

        def __len__(self):
            return self.target_attention_mask.shape[0]

        def __getitem__(self, idx):
            # Return multiple inputs as a tuple
            return self.neural_data[idx], self.input_ids[idx], self.attention_mask[idx], self.target_ids[idx], self.target_attention_mask[idx]


    def get_target_preds(outputs, all_attention_mask, target_attention_mask):
        logits = outputs.logits[:, :-1, :] # b * seq_all-1 * logits
        all_attention_mask = all_attention_mask[:,1:]

        labels_mask = torch.zeros_like(all_attention_mask)
        target_mask_sum = torch.sum(target_attention_mask, dim=1).int()            
        all_mask_sum = torch.sum(all_attention_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][all_mask_sum[batch_id]-target_mask_sum[batch_id]:all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to("cuda")
        logits = logits[labels_mask==1]
        return logits 


    def _(neural_data):
        # Assuming you have:
        # - model: your neural network
        # - train_loader: DataLoader for training data
        # - val_loader: DataLoader for validation data (optional)
        folds = get_sequential_folds(X, num_folds=5)
        (tr_idx, val_idx, test_idx) = folds[0]
        train_dataset = MultiInputDataset(neural_data[tr_idx], lag_all_input_ids[tr_idx], lag_all_attention_masks[tr_idx], lag_target_input_ids[tr_idx], lag_target_attention_masks[tr_idx])
        val_dataset = MultiInputDataset(neural_data[val_idx], lag_all_input_ids[val_idx], lag_all_attention_masks[val_idx], lag_target_input_ids[val_idx], lag_target_attention_masks[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # brain_prompt_model = brain_prompt_model.to(device)
        criterion = nn.CrossEntropyLoss()  # or your loss function
        parameters = []
        parameters += brain_prompt_model.parameters()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=1e-4, weight_decay=0.01)

        # Training loop
        num_epochs = 30

        for epoch in range(num_epochs):
            # Training phase
            brain_prompt_model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

            for batch_idx, (curr_neur, input_ids, attention_mask, target_input_ids, targ_attention_mask) in enumerate(train_pbar):
                # Move data to device
                curr_neur, input_ids, attention_mask, target_input_ids, targ_attention_mask = curr_neur.to(device), input_ids.to(device), attention_mask.to(device), target_input_ids.to(device), targ_attention_mask.to(device)

                # Forward pass
                outputs, brain_prompt_mask = brain_prompt_model(curr_neur, input_ids, attention_mask)
                target_preds = get_target_preds(outputs, brain_prompt_mask, targ_attention_mask)
                target_labels = target_input_ids[targ_attention_mask]
                loss = criterion(target_preds, target_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
                })

            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase (optional)
            brain_prompt_model.eval()
            val_loss = 0.0
            val_perplexity = 0.0
            correct = 0
            total = 0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

            with torch.no_grad():
                for curr_neur, input_ids, attention_mask, target_input_ids, targ_attention_mask in val_pbar:
                    curr_neur, input_ids, attention_mask, target_input_ids, targ_attention_mask = curr_neur.to(device), input_ids.to(device), attention_mask.to(device), target_input_ids.to(device), targ_attention_mask.to(device)
                    outputs, brain_prompt_mask = brain_prompt_model(curr_neur, input_ids, attention_mask)
                    target_preds = get_target_preds(outputs, brain_prompt_mask, targ_attention_mask)
                    target_labels = target_input_ids[targ_attention_mask]
                    loss = criterion(target_preds, target_labels)
                    val_perplexity += perplexity(torch.softmax(target_preds, dim=1).cpu().numpy(), target_labels.cpu().numpy())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_perplexity = val_perplexity / len(val_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, '
                      f'Val Perplexity: {avg_val_perplexity:.4f}'
                 )


    _(neural_data)
    return (get_target_preds,)


@app.cell
def _():
    # def _():
    #     folds = get_sequential_folds(X, num_folds=5)
    #     (tr_idx, val_idx, test_idx) = folds[0]
    #     train_dataset = MultiInputDataset(neural_data[tr_idx], lag_all_input_ids[tr_idx], lag_all_attention_masks[tr_idx], lag_target_input_ids[tr_idx], lag_target_attention_masks[tr_idx])
    #     val_dataset = MultiInputDataset(neural_data[val_idx], lag_all_input_ids[val_idx], lag_all_attention_masks[val_idx], lag_target_input_ids[val_idx], lag_target_attention_masks[val_idx])

    #     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #     val_loader = DataLoader(val_dataset, batch_size=32)

    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # brain_prompt_model = brain_prompt_model.to(device)
    #     criterion = nn.CrossEntropyLoss()  # or your loss function

    #     # Validation phase (optional)
    #     brain_prompt_model.eval()
    #     val_loss = 0.0
    #     val_perplexity = 0.0
    #     correct = 0
    #     total = 0

    #     val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

    #     with torch.no_grad():
    #         for curr_neur, input_ids, attention_mask, target_input_ids, targ_attention_mask in val_pbar:
    #             curr_neur, input_ids, attention_mask, target_input_ids, targ_attention_mask = curr_neur.to(device), input_ids.to(device), attention_mask.to(device), target_input_ids.to(device), targ_attention_mask.to(device)
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    #             target_preds = get_target_preds(outputs, attention_mask, targ_attention_mask)
    #             target_labels = target_input_ids[targ_attention_mask]
    #             loss = criterion(target_preds, target_labels)
    #             val_perplexity += perplexity(torch.softmax(target_preds, dim=1).cpu().numpy(), target_labels.cpu().numpy())
    #             val_loss += loss.item()

    #     avg_val_loss = val_loss / len(val_loader)
    #     avg_val_perplexity = val_perplexity / len(val_loader)
    #     return print(f'Epoch [{epoch+1}/{num_epochs}], '
    #               f'Train Loss: {avg_train_loss:.4f}, '
    #               f'Val Loss: {avg_val_loss:.4f}, '
    #               f'Val Perplexity: {avg_val_perplexity:.4f}'
    #          )


    # _()
    return


@app.cell
def _(mo):
    mo.md("""## Cleaned word transcript file generation""")
    return


@app.cell
def _(df_word, pd):
    words = df_word.word.to_string(header=False, index=False).split("\n")
    words_transcript = " ".join([word.strip() for word in words])

    df_sentences = pd.read_csv("processed_data/all_sentences_podcast.csv")
    stripped_sentences = df_sentences.all_sentence.str.strip()
    sentence_transcript = stripped_sentences.str.cat(sep=" ")
    sentence_transcript
    return sentence_transcript, words_transcript


@app.cell
def _(sentence_transcript, words_transcript):
    import re
    import difflib

    def tokenize_with_punctuation(text):
        """
        Tokenize text into words while preserving punctuation info.
        Returns list of tuples: (word, trailing_punctuation, original_token)
        """
        # Split on whitespace while preserving the tokens
        tokens = text.split()
        result = []

        for token in tokens:
            # Separate word from trailing punctuation
            match = re.match(r"^(.*?)([.,!?;:'\"-]*)$", token)
            if match:
                word = match.group(1)
                punct = match.group(2)
                result.append((word, punct, token))

        return result

    def normalize_word(word):
        """Normalize word for comparison (lowercase, remove punctuation)"""
        return re.sub(r'[^\w]', '', word.lower())

    def align_transcripts(tokens1, tokens2):
        """
        Align two token lists and create mapping.
        Returns list of (token2_index, token1_index or None) pairs
        """
        words1 = [normalize_word(t[0]) for t in tokens1]
        words2 = [normalize_word(t[0]) for t in tokens2]

        # Use difflib to find matching sequences
        matcher = difflib.SequenceMatcher(None, words2, words1)

        # Create alignment mapping
        alignment = [None] * len(tokens2)

        for match in matcher.get_matching_blocks():
            i, j, size = match
            for k in range(size):
                if i + k < len(tokens2):
                    alignment[i + k] = j + k

        return alignment

    def merge_transcripts(text1, text2):
        """
        Merge transcripts: keep words from text2, import punctuation from text1
        """
        tokens1 = tokenize_with_punctuation(text1)
        tokens2 = tokenize_with_punctuation(text2)
        print(tokens1)

        print(f"Transcript 1: {len(tokens1)} tokens")
        print(f"Transcript 2: {len(tokens2)} tokens")

        # Align the transcripts
        alignment = align_transcripts(tokens1, tokens2)
        print(alignment)

        # Build merged result
        merged_tokens = []

        for idx2, (word2, punct2, orig2) in enumerate(tokens2):
            idx1 = alignment[idx2]

            if idx1 is not None:
                # We have an alignment - use punctuation from transcript 1
                word1, punct1, orig1 = tokens1[idx1]

                # Keep the word from transcript 2, but use punctuation from transcript 1
                # Also consider capitalization from transcript 1
                if word1 and word1[0].isupper() and word2 and word2[0].islower():
                    # Capitalize to match transcript 1
                    merged_word = word2[0].upper() + word2[1:] if len(word2) > 1 else word2.upper()
                else:
                    merged_word = word2

                merged_tokens.append(merged_word + punct1)
            else:
                # No alignment - keep original from transcript 2
                merged_tokens.append(orig2)

        return ' '.join(merged_tokens)

    print("Reading transcript files...")

    print("\nMerging transcripts...")
    merged = merge_transcripts(sentence_transcript, words_transcript)

    print(f"Total length: {len(merged)} characters")
    return merged, re


@app.cell
def _(data_params, merged, os, pd):
    len(merged.split()), len(pd.read_csv(os.path.join(data_params.data_root, "stimuli/podcast_transcript.csv")).word)
    return


@app.cell
def _(data_params, merged, np, os, pd, re):
    (np.array([re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text) for text in merged.split()]) == np.array([re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text) for text in pd.read_csv(os.path.join(data_params.data_root, "stimuli/podcast_transcript.csv")).word])).all()
    return


@app.cell
def _(df_word, merged, pd):
    punctuated_word_df = pd.DataFrame({"word": merged.split(), "start": df_word.start, "end": df_word.end})
    punctuated_word_df.to_csv("processed_data/punctuated_transcript.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
