import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from core import registry


def load_gpt2_model_and_tokenizer(
    cache_dir: str,
    model_name: str = "gpt2",
):
    """Load GPT-2 model and tokenizer."""
    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


class GPT2Brain(nn.Module):
    """
    GPT2-based language model with brain data integration.

    Wraps a pretrained GPT2 model (any variant) with a neural encoder to generate
    text conditioned on brain activity. Brain embeddings are wrapped with special
    tokens (<brain/>, </brain>) and prepended to the input sequence.

    Supports two encoder output formats:
    - [batch_size, embedding_dim]: Single embedding per sample
    - [batch_size, num_tokens, embedding_dim]: Multiple embeddings per sample
    """

    def __init__(
        self,
        lm_model,
        tokenizer,
        encoder_model,
        freeze_lm=True,
        encoder_forward_kwargs={},
        no_brain_encoder=False,
        no_brain_token_injection=False,
    ):
        """
        Initialize GPT2Brain model.

        Args:
            lm_model: Pretrained GPT2LMHeadModel (any variant: gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            tokenizer: GPT2TokenizerFast tokenizer
            encoder_model: Neural encoder that transforms brain data.
                          Input: [batch_size, num_channels, num_timepoints]
                          Output: [batch_size, embedding_dim] or [batch_size, num_tokens, embedding_dim]
            freeze_lm: Whether to freeze the language model weights (default: True)
            encoder_forward_kwargs: Additional kwargs to pass to encoder_model during forward
            no_brain_encoder: If True, bypass the encoder and do not provide brain embeddings.
            no_brain_token_injection: If True, do not inject brain separator tokens.
        """
        super().__init__()

        self.no_brain_encoder = no_brain_encoder
        self.no_brain_token_injection = no_brain_token_injection

        self.lm_model = lm_model
        self.tokenizer = tokenizer
        if not self.no_brain_encoder:
            self.encoder_model = encoder_model

        self.encoder_forward_kwargs = encoder_forward_kwargs

        # Add brain separator tokens to tokenizer
        if not self.no_brain_token_injection:
            brain_tokens = ["<brain/>", "</brain>"]
            self.tokenizer.add_tokens(brain_tokens)

            # Resize model embeddings to accommodate new tokens
            self.lm_model.resize_token_embeddings(len(self.tokenizer))

            self.brain_token_ids = [
                self.tokenizer.convert_tokens_to_ids("<brain/>"),
                self.tokenizer.convert_tokens_to_ids("</brain>"),
            ]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model.config.pad_token_id = self.tokenizer.eos_token_id

        # Freeze language model if requested
        if freeze_lm:
            for param in self.lm_model.parameters():
                param.requires_grad = False

            # Enable gradients for the entire embedding layer
            # (we can't set requires_grad on a view/subset of embeddings)
            if not self.no_brain_token_injection:
                self.lm_model.transformer.wte.weight.requires_grad = True

                # Register backward hook to zero gradients for all embeddings
                # except brain token embeddings
                def zero_non_brain_gradients(grad):
                    """Zero out gradients for all embeddings except brain tokens."""
                    # Create a mask that's 1 for brain tokens, 0 for everything else
                    mask = torch.zeros_like(grad)
                    for token_id in self.brain_token_ids:
                        mask[token_id] = 1.0
                    # Zero gradients for non-brain tokens
                    return grad * mask

                self.lm_model.transformer.wte.weight.register_hook(
                    zero_non_brain_gradients
                )

    def _get_brain_separator_embeddings(self, batch_size, device):
        """Get embeddings for brain separator tokens."""
        brain_token_ids = torch.tensor(self.brain_token_ids, device=device).unsqueeze(0)
        brain_token_ids = brain_token_ids.expand(batch_size, 2)  # [batch, 2]

        brain_sep_embeddings = self.lm_model.transformer.wte(
            brain_token_ids
        )  # [batch, 2, hidden_size]

        return (
            brain_sep_embeddings[:, 0:1, :],
            brain_sep_embeddings[:, 1:2, :],
        )  # first_sep, last_sep

    def _get_target_predictions(
        self, outputs, all_attention_mask, target_attention_mask
    ):
        """
        Extract predictions for target tokens only, maintaining batch structure.

        Similar to playground.py's get_target_preds, but returns padded predictions
        [batch, max_target_tokens, vocab_size] instead of flattened.

        Args:
            outputs: Model output with logits [batch, seq_len, vocab_size]
            all_attention_mask: Attention mask for full sequence [batch, seq_len]
            target_attention_mask: Attention mask for target tokens [batch, max_target_tokens]

        Returns:
            target_logits: Padded logits for target tokens [batch, max_target_tokens, vocab_size]
        """
        device = outputs.logits.device
        # Shift logits and mask to align predictions with next token
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        all_attention_mask = all_attention_mask[:, 1:]  # [batch, seq_len-1]

        batch_size = logits.shape[0]
        max_target_tokens = target_attention_mask.shape[1]
        vocab_size = logits.shape[2]

        # Create output tensor for target predictions
        target_logits = torch.zeros(
            batch_size, max_target_tokens, vocab_size, device=device
        )

        # Extract target tokens for each sample in the batch
        target_mask_sum = torch.sum(target_attention_mask, dim=1).int()
        all_mask_sum = torch.sum(all_attention_mask, dim=1).int()

        for batch_id in range(batch_size):
            num_target_tokens = target_mask_sum[batch_id].item()
            if num_target_tokens == 0:
                continue

            # Target tokens are at the end of the valid sequence
            start_idx = all_mask_sum[batch_id] - num_target_tokens
            end_idx = all_mask_sum[batch_id]

            # Extract target predictions and place in output
            target_logits[batch_id, :num_target_tokens] = logits[
                batch_id, start_idx:end_idx
            ]

        return target_logits

    def _convert_to_embeddings(self, neural_data, input_ids, attention_mask):
        """
        Convert neural data and input_ids to combined embedding sequence.

        Args:
            neural_data: [batch_size, num_channels, num_timepoints]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            prompt_embeddings: [batch_size, brain_prompt_len + seq_len, hidden_size]
            prompt_attention_mask: [batch_size, brain_prompt_len + seq_len]
        """
        device = input_ids.device
        batch_size = neural_data.shape[0]

        all_tokens = []
        num_neural_tokens = 0

        if not self.no_brain_encoder:
            neural_embedding = self.encoder_model(
                neural_data, **self.encoder_forward_kwargs
            )  # [batch, embed_dim] or [batch, num_tokens, embed_dim]

            if neural_embedding.ndim == 2:
                # Single embedding: [batch, embed_dim] -> [batch, 1, embed_dim]
                neural_embedding = neural_embedding.unsqueeze(1)

            num_neural_tokens = neural_embedding.shape[1]
            all_tokens.append(neural_embedding)

        if not self.no_brain_token_injection:
            first_sep_embedding, last_sep_embedding = (
                self._get_brain_separator_embeddings(batch_size, device)
            )
            all_tokens = [first_sep_embedding] + all_tokens + [last_sep_embedding]

        input_embeddings = self.lm_model.transformer.wte(
            input_ids
        )  # [batch, seq_len, hidden_size]
        all_tokens.append(input_embeddings)

        # Concatenate: [<brain/>, neural_embeds, </brain>, input_embeds]
        prompt_embeddings = torch.cat(
            all_tokens,
            dim=1,
        )

        # Create attention mask for brain prompt (always attended to)
        brain_prompt_len = num_neural_tokens + (
            0 if self.no_brain_token_injection else 2
        )  # neural_tokens + optional separators

        if brain_prompt_len > 0:
            brain_attention_mask = torch.ones(
                batch_size, brain_prompt_len, device=device
            )
            prompt_attention_mask = torch.cat(
                [
                    brain_attention_mask,  # [batch, brain_prompt_len]
                    attention_mask.to(device),  # [batch, seq_len]
                ],
                dim=1,
            )
        else:
            # No brain data at all, just use input attention mask
            prompt_attention_mask = attention_mask.to(device)

        return prompt_embeddings, prompt_attention_mask

    def forward(
        self,
        neural_data,
        all_input_ids,
        all_attention_mask,
        target_attention_mask=None,
        return_all_preds=False,
    ):
        """
        Forward pass through the model.

        Args:
            neural_data: [batch_size, num_channels, num_timepoints]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            target_attention_mask: [batch_size, max_target_tokens] - Required if return_all_preds=False
            return_all_preds: If True, returns predictions for all tokens (default).
                            If False, returns only predictions for target tokens.

        Returns:
            If return_all_preds=True:
                output: Model output with logits [batch_size, brain_prompt_len + seq_len, vocab_size]
                updated_attention_mask: [batch_size, brain_prompt_len + seq_len]
            If return_all_preds=False:
                target_logits: Predictions for target tokens [batch_size, max_target_tokens, vocab_size]
        """
        if not return_all_preds and target_attention_mask is None:
            raise ValueError(
                "target_attention_mask must be provided when return_all_preds=False"
            )

        prompt_embeddings, prompt_attention_mask = self._convert_to_embeddings(
            neural_data, all_input_ids, all_attention_mask
        )

        output = self.lm_model(
            inputs_embeds=prompt_embeddings, attention_mask=prompt_attention_mask
        )

        if return_all_preds:
            return output, prompt_attention_mask
        else:
            target_logits = self._get_target_predictions(
                output, prompt_attention_mask, target_attention_mask
            )
            return target_logits

    def generate(
        self,
        neural_data,
        input_ids,
        attention_mask,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.6,
        top_k=50,
        top_p=0.75,
        repetition_penalty=1.2,
        **kwargs,
    ):
        """
        Generate text autoregressively from brain data and context.

        NOTE: Returns only text token IDs (input_ids + newly generated tokens).
              Does NOT include brain embedding positions since they have no token IDs.

        Args:
            neural_data: [batch_size, num_channels, num_timepoints]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling (vs greedy decoding)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            generated_ids: [batch_size, seq_len + num_new_tokens] - text token IDs only
        """
        device = input_ids.device
        prompt_embeddings, prompt_attention_mask = self._convert_to_embeddings(
            neural_data, input_ids, attention_mask
        )

        # For generation, we need left padding instead of right padding
        # This ensures the model generates after the prompt rather than in padding
        prompt_embeddings, prompt_attention_mask = self._pad_left(
            prompt_embeddings, prompt_attention_mask
        )

        # Generate with embeddings as input
        # IMPORTANT: When using inputs_embeds, model.generate() only returns
        # newly generated token IDs, NOT token IDs for the embedding positions
        generated_tokens = self.lm_model.generate(
            inputs_embeds=prompt_embeddings,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        # Concatenate original input_ids with newly generated tokens
        # generated_tokens contains only the new tokens generated by the model
        full_sequence = torch.cat([input_ids.to(device), generated_tokens], dim=1)

        return full_sequence

    def _pad_left(self, embeddings, attention_mask):
        """
        Convert right-padded sequences to left-padded for generation.

        Args:
            embeddings: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]

        Returns:
            left_padded_embeddings: [batch, seq_len, hidden_size]
            left_padded_mask: [batch, seq_len]
        """
        batch_size, seq_len, hidden_size = embeddings.shape

        # Count non-padding positions for each sample
        content_lengths = attention_mask.sum(dim=1).long()  # [batch]

        left_padded_embeddings = torch.zeros_like(embeddings)
        left_padded_mask = torch.zeros_like(attention_mask)

        for i in range(batch_size):
            content_len = content_lengths[i].item()
            # Move content to the right (end of sequence)
            left_padded_embeddings[i, -content_len:] = embeddings[i, :content_len]
            left_padded_mask[i, -content_len:] = 1

        return left_padded_embeddings, left_padded_mask

    def save_checkpoint(self, path):
        """
        Save minimal checkpoint with only trainable parameters.

        Saves only:
        - Encoder model weights
        - Brain token embeddings (<brain/>, </brain>)
        - Brain token IDs

        Does NOT save the full GPT2 model (which can be loaded separately).

        Args:
            path: Filepath to save checkpoint
        """
        checkpoint = {
            "encoder_model": (
                self.encoder_model.state_dict() if not self.no_brain_encoder else None
            ),
            "brain_token_embeddings": (
                self.lm_model.transformer.wte.weight[self.brain_token_ids]
                .detach()
                .cpu()
                if not self.no_brain_token_injection
                else None
            ),
            "brain_token_ids": (
                self.brain_token_ids if not self.no_brain_token_injection else None
            ),
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load checkpoint and restore trainable parameters.

        Restores:
        - Encoder model weights
        - Brain token embeddings to correct positions in GPT2 embedding layer

        Args:
            path: Filepath to load checkpoint from
        """
        checkpoint = torch.load(path, map_location="cpu")

        if not self.no_brain_encoder:
            self.encoder_model.load_state_dict(checkpoint["encoder_model"])

        if not self.no_brain_token_injection:
            brain_token_embeddings = checkpoint["brain_token_embeddings"]
            with torch.no_grad():
                for i, token_id in enumerate(self.brain_token_ids):
                    self.lm_model.transformer.wte.weight[token_id] = (
                        brain_token_embeddings[i]
                    )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        lm_model,
        tokenizer,
        encoder_model,
        freeze_lm=True,
    ):
        """
        Alternative constructor to create GPT2Brain from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            lm_model: Pretrained GPT2LMHeadModel
            tokenizer: GPT2TokenizerFast tokenizer
            encoder_model: Neural encoder model
            freeze_lm: Whether to freeze language model weights

        Returns:
            GPT2Brain instance with loaded weights
        """
        model = cls(
            lm_model=lm_model,
            tokenizer=tokenizer,
            encoder_model=encoder_model,
            freeze_lm=freeze_lm,
        )
        model.load_checkpoint(checkpoint_path)

        return model


@registry.register_model_constructor("gpt2_brain")
def gpt2_brain_model_constructor(model_params):
    """Construct GPT2Brain model from model_spec."""
    lm_model, tokenizer = load_gpt2_model_and_tokenizer(
        cache_dir=model_params.get("cache_dir", None)
    )
    encoder_model = model_params.get("encoder_model", None)
    freeze_lm = model_params.get("freeze_lm", True)

    model = GPT2Brain(
        lm_model=lm_model,
        tokenizer=tokenizer,
        encoder_model=encoder_model,
        freeze_lm=freeze_lm,
        encoder_forward_kwargs=model_params.get("encoder_forward_kwargs", {}),
        no_brain_encoder=model_params.get("no_brain_encoder", False),
        no_brain_token_injection=model_params.get("no_brain_token_injection", False),
    )

    return model
