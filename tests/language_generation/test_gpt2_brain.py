import pytest
import torch
import torch.nn as nn
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch


class MockEncoderSingleEmbed(nn.Module):
    """Mock encoder that produces single embedding: [batch, channels, time] -> [batch, embedding_dim]"""

    def __init__(self, input_channels=64, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_channels, output_dim)

    def forward(self, x):
        # x: [batch, channels, time]
        # Average over time dimension and project
        x_avg = x.mean(dim=2)  # [batch, channels]
        return self.linear(x_avg)  # [batch, output_dim]


class MockEncoderMultiEmbed(nn.Module):
    """Mock encoder that produces multiple embeddings: [batch, channels, time] -> [batch, num_tokens, embedding_dim]"""

    def __init__(self, input_channels=64, output_dim=256, num_tokens=3):
        super().__init__()
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.linear = nn.Linear(input_channels, output_dim * num_tokens)

    def forward(self, x):
        # x: [batch, channels, time]
        x_avg = x.mean(dim=2)  # [batch, channels]
        out = self.linear(x_avg)  # [batch, output_dim * num_tokens]
        batch_size = x.shape[0]
        return out.view(
            batch_size, self.num_tokens, self.output_dim
        )  # [batch, num_tokens, output_dim]


class MockTransformer(nn.Module):
    """Mock transformer for proper parameter handling"""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)


class MockGPT2LMHeadModel(nn.Module):
    """Mock GPT2 model for testing without downloading weights"""

    def __init__(self, vocab_size=1000, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Create mock transformer structure with real nn.Module
        self.transformer = MockTransformer(vocab_size, hidden_size)

        # Simple output layer
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Config
        self.config = MagicMock()
        self.config.vocab_size = vocab_size
        self.config.n_embd = hidden_size
        self.config.pad_token_id = None

    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.transformer.wte(input_ids)

        logits = self.lm_head(hidden_states)

        output = MagicMock()
        output.logits = logits
        return output

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        max_new_tokens=10,
        pad_token_id=None,
        **kwargs,
    ):
        """Mock generate method - returns only newly generated tokens when using inputs_embeds"""
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            # When using inputs_embeds, HuggingFace generate() returns ONLY the newly generated tokens
            # It does NOT return token IDs for the embedding positions or original input
            return torch.randint(0, self.vocab_size, (batch_size, max_new_tokens))
        else:
            batch_size = input_ids.shape[0]
            input_text_len = input_ids.shape[1]
            # When using input_ids, it returns the full sequence (input + new tokens)
            return torch.randint(
                0, self.vocab_size, (batch_size, input_text_len + max_new_tokens)
            )

    def resize_token_embeddings(self, new_vocab_size):
        """Mock resize embeddings"""
        old_embeddings = self.transformer.wte
        new_embeddings = nn.Embedding(new_vocab_size, self.hidden_size)

        # Copy old weights
        new_embeddings.weight.data[: old_embeddings.num_embeddings] = (
            old_embeddings.weight.data
        )

        self.transformer.wte = new_embeddings

        # Also resize lm_head to match new vocab size
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.hidden_size, new_vocab_size, bias=False)
        new_lm_head.weight.data[: old_lm_head.out_features] = old_lm_head.weight.data
        self.lm_head = new_lm_head

        self.vocab_size = new_vocab_size
        self.config.vocab_size = new_vocab_size


class MockGPT2TokenizerFast:
    """Mock tokenizer for testing"""

    def __init__(self):
        self.vocab = {f"token_{i}": i for i in range(1000)}
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 999
        self.pad_token = None
        self.pad_token_id = None

    def add_tokens(self, tokens):
        """Add new tokens to vocabulary"""
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        """Convert token to ID"""
        return self.vocab.get(token, 0)

    def get_vocab(self):
        """Return vocabulary"""
        return self.vocab

    def __call__(self, text, **kwargs):
        """Mock tokenizer call"""
        if isinstance(text, str):
            text = [text]

        # Return mock token IDs
        input_ids = torch.randint(0, len(self.vocab), (len(text), 10))
        return {"input_ids": input_ids}

    def __len__(self):
        return len(self.vocab)


@pytest.fixture
def mock_gpt2_model():
    """Fixture to create mock GPT2 model"""
    return MockGPT2LMHeadModel()


@pytest.fixture
def mock_tokenizer():
    """Fixture to create mock tokenizer"""
    return MockGPT2TokenizerFast()


@pytest.fixture
def mock_encoder_single():
    """Fixture providing a mock encoder with single embedding output"""
    return MockEncoderSingleEmbed(input_channels=64, output_dim=256)


@pytest.fixture
def mock_encoder_multi():
    """Fixture providing a mock encoder with multiple embedding outputs"""
    return MockEncoderMultiEmbed(input_channels=64, output_dim=256, num_tokens=3)


@pytest.fixture
def sample_batch():
    """Fixture providing sample batch data"""
    batch_size = 2
    num_channels = 64
    num_timepoints = 100
    seq_len = 10

    return {
        "neural_data": torch.randn(batch_size, num_channels, num_timepoints),
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }


class TestGPT2BrainConstruction:
    """Test GPT2Brain class instantiation"""

    def test_instantiation_with_encoder(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer
    ):
        """Test 1: GPT2Brain class can be instantiated with encoder and tokenizer"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        assert model is not None
        assert hasattr(model, "encoder_model")
        assert hasattr(model, "lm_model")
        assert hasattr(model, "tokenizer")
        assert model.encoder_model is mock_encoder_single
        assert model.lm_model is mock_gpt2_model
        assert model.tokenizer is mock_tokenizer

    def test_brain_tokens_added(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer
    ):
        """Test 4: Brain separator token embeddings are created correctly"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        # Check that brain tokens were added to tokenizer
        assert "<brain/>" in model.tokenizer.get_vocab()
        assert "</brain>" in model.tokenizer.get_vocab()

        # Check that we stored the token IDs
        assert hasattr(model, "brain_token_ids")
        assert len(model.brain_token_ids) == 2

    def test_model_frozen(self, mock_encoder_single, mock_gpt2_model, mock_tokenizer):
        """Test that GPT2 weights are frozen except brain tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        # Most parameters should be frozen
        frozen_params = sum(
            1 for p in model.lm_model.parameters() if not p.requires_grad
        )
        trainable_params = sum(
            1 for p in model.lm_model.parameters() if p.requires_grad
        )

        # Should have frozen most params
        assert frozen_params > 0

        # Embedding weight should have requires_grad=True
        embedding_weight = model.lm_model.transformer.wte.weight
        assert embedding_weight.requires_grad

        # Brain token embeddings should be trainable
        brain_token_id_1, brain_token_id_2 = model.brain_token_ids
        assert embedding_weight[brain_token_id_1].requires_grad
        assert embedding_weight[brain_token_id_2].requires_grad

    def test_gradient_masking(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer
    ):
        """Test that gradients are correctly zeroed for non-brain token embeddings"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        # Create dummy input with brain tokens included to ensure they have gradients
        batch_size = 2
        neural_data = torch.randn(batch_size, 64, 100)

        # Include brain tokens in the input_ids to ensure they're used
        brain_token_ids_tensor = torch.tensor(
            [model.brain_token_ids], dtype=torch.long
        ).repeat(batch_size, 1)
        regular_tokens = torch.randint(0, 1000, (batch_size, 8))
        input_ids = torch.cat([brain_token_ids_tensor, regular_tokens], dim=1)
        attention_mask = torch.ones(batch_size, input_ids.shape[1])

        # Forward pass
        output, _ = model(neural_data, input_ids, attention_mask, return_all_preds=True)

        # Create a dummy loss and compute gradients
        loss = output.logits.sum()
        loss.backward()

        # Get the embedding gradients
        embedding_grad = model.lm_model.transformer.wte.weight.grad
        assert embedding_grad is not None

        # Check that only brain token embeddings have non-zero gradients
        brain_token_id_1, brain_token_id_2 = model.brain_token_ids

        # Check brain token gradients are non-zero
        brain_grad_1 = embedding_grad[brain_token_id_1]
        brain_grad_2 = embedding_grad[brain_token_id_2]

        assert not torch.allclose(
            brain_grad_1, torch.zeros_like(brain_grad_1)
        ), "Brain token 1 should have non-zero gradient"
        assert not torch.allclose(
            brain_grad_2, torch.zeros_like(brain_grad_2)
        ), "Brain token 2 should have non-zero gradient"

        # Check all other embeddings have zero gradients
        for token_id in range(embedding_grad.shape[0]):
            if token_id not in model.brain_token_ids:
                assert torch.allclose(
                    embedding_grad[token_id],
                    torch.zeros_like(embedding_grad[token_id]),
                ), f"Token {token_id} should have zero gradient but has non-zero values"


class TestGPT2BrainForwardSingleEmbed:
    """Test GPT2Brain forward pass with single embedding encoder"""

    def test_forward_pass_shapes_single_embed(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test 2a: Forward pass with single embedding [batch, embed_dim] produces correct shapes"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            return_all_preds=True,
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        vocab_size = len(model.tokenizer)

        # Output logits should have shape [batch, seq_len + 3, vocab_size]
        # 3 = <brain/> + neural_embed (1 token) + </brain>
        expected_seq_len = seq_len + 3
        assert output.logits.shape == (batch_size, expected_seq_len, vocab_size)

        # Attention mask should also be updated
        assert updated_attention_mask.shape == (batch_size, expected_seq_len)

    def test_attention_mask_handling_single_embed(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test 3a: Attention mask handling with single embedding"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        # Create attention mask with some padding (zeros at end)
        batch_size, seq_len = sample_batch["input_ids"].shape
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -3:] = 0  # Mask last 3 tokens

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=attention_mask,
            return_all_preds=True,
        )

        # First 3 positions should be attended to (brain prompt: <brain/>, embed, </brain>)
        assert (updated_attention_mask[:, :3] == 1).all()

        # Original attention pattern should be preserved after brain prompt
        assert torch.allclose(updated_attention_mask[:, 3:], attention_mask)


class TestGPT2BrainForwardMultiEmbed:
    """Test GPT2Brain forward pass with multiple embedding encoder"""

    def test_forward_pass_shapes_multi_embed(
        self, mock_encoder_multi, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test 2b: Forward pass with multiple embeddings [batch, num_tokens, embed_dim] produces correct shapes"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_multi,
            freeze_lm=True,
        )

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            return_all_preds=True,
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        vocab_size = len(model.tokenizer)
        num_neural_tokens = 3  # From mock_encoder_multi

        # Output logits should have shape [batch, seq_len + num_neural_tokens + 2, vocab_size]
        # Structure: <brain/> + num_neural_tokens + </brain> + input_ids
        expected_seq_len = seq_len + num_neural_tokens + 2
        assert output.logits.shape == (batch_size, expected_seq_len, vocab_size)

        # Attention mask should also be updated
        assert updated_attention_mask.shape == (batch_size, expected_seq_len)

    def test_attention_mask_handling_multi_embed(
        self, mock_encoder_multi, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test 3b: Attention mask handling with multiple embeddings"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_multi,
            freeze_lm=True,
        )

        # Create attention mask with some padding
        batch_size, seq_len = sample_batch["input_ids"].shape
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=attention_mask,
            return_all_preds=True,
        )

        num_neural_tokens = 3  # From mock_encoder_multi
        brain_prompt_len = num_neural_tokens + 2  # <brain/> + tokens + </brain>

        # First brain_prompt_len positions should be attended to
        assert (updated_attention_mask[:, :brain_prompt_len] == 1).all()

        # Original attention pattern should be preserved after brain prompt
        assert torch.allclose(
            updated_attention_mask[:, brain_prompt_len:], attention_mask
        )


class TestGPT2BrainGeneration:
    """Test GPT2Brain text generation"""

    def test_generate_produces_sequences_single_embed(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test 5a: Generate method produces text sequences with single embedding - returns only text tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )
        model.eval()

        with torch.no_grad():
            generated_ids = model.generate(
                neural_data=sample_batch["neural_data"],
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
                max_new_tokens=5,
                do_sample=False,  # Use greedy for deterministic test
            )

        batch_size, original_seq_len = sample_batch["input_ids"].shape

        # Generated sequences should contain original input + new tokens
        # Brain embeddings are NOT included in the output (no token IDs for them)
        assert generated_ids.shape[0] == batch_size
        assert (
            generated_ids.shape[1] == original_seq_len + 5
        )  # original + max_new_tokens

    def test_generate_produces_sequences_multi_embed(
        self, mock_encoder_multi, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test 5b: Generate method produces text sequences with multiple embeddings - returns only text tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_multi,
            freeze_lm=True,
        )
        model.eval()

        with torch.no_grad():
            generated_ids = model.generate(
                neural_data=sample_batch["neural_data"],
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
            )

        batch_size, original_seq_len = sample_batch["input_ids"].shape

        # Generated sequences should contain original input + new tokens
        # Brain embeddings are NOT included in the output
        assert generated_ids.shape[0] == batch_size
        assert (
            generated_ids.shape[1] == original_seq_len + 5
        )  # original + max_new_tokens

    def test_generate_with_custom_params(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test generate with custom generation parameters"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )
        model.eval()

        with torch.no_grad():
            generated_ids = model.generate(
                neural_data=sample_batch["neural_data"],
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
                max_new_tokens=10,
                do_sample=True,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
            )

        # Should produce valid token sequences
        assert generated_ids.shape[0] == sample_batch["neural_data"].shape[0]
        assert generated_ids.dtype == torch.long


class TestGPT2BrainCheckpointing:
    """Test GPT2Brain checkpoint save/load functionality"""

    def test_save_checkpoint(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, tmp_path
    ):
        """Test 6: Save checkpoint stores only encoder + brain token embeddings"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        model.save_checkpoint(str(checkpoint_path))

        # Check that checkpoint file was created
        assert checkpoint_path.exists()

        # Load checkpoint and verify contents
        checkpoint = torch.load(checkpoint_path)

        assert "encoder_model" in checkpoint
        assert "brain_token_embeddings" in checkpoint
        assert "brain_token_ids" in checkpoint

        # Brain token embeddings should be shape [2, embedding_dim]
        assert checkpoint["brain_token_embeddings"].shape[0] == 2

        # Check that file size is reasonable (< 100MB, should be much smaller)
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        assert file_size_mb < 100  # Should be much smaller than full GPT2

    def test_load_checkpoint(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, tmp_path
    ):
        """Test 7: Load checkpoint restores encoder + brain token embeddings correctly"""
        from language_generation.gpt2_brain import GPT2Brain

        # Create and save a model
        model1 = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        model1.save_checkpoint(str(checkpoint_path))

        # Store original values
        original_encoder_state = model1.encoder_model.state_dict()
        original_brain_embeds = model1.lm_model.transformer.wte.weight[
            model1.brain_token_ids
        ].clone()

        # Create new model and load checkpoint
        new_encoder = MockEncoderSingleEmbed(input_channels=64, output_dim=256)
        new_gpt2_model = MockGPT2LMHeadModel()
        new_tokenizer = MockGPT2TokenizerFast()

        model2 = GPT2Brain(
            lm_model=new_gpt2_model,
            tokenizer=new_tokenizer,
            encoder_model=new_encoder,
            freeze_lm=True,
        )
        model2.load_checkpoint(str(checkpoint_path))

        # Verify encoder weights were restored
        loaded_encoder_state = model2.encoder_model.state_dict()
        for key in original_encoder_state:
            assert torch.allclose(
                original_encoder_state[key], loaded_encoder_state[key]
            )

        # Verify brain token embeddings were restored
        loaded_brain_embeds = model2.lm_model.transformer.wte.weight[
            model2.brain_token_ids
        ]
        assert torch.allclose(original_brain_embeds, loaded_brain_embeds)

    def test_from_checkpoint_classmethod(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, tmp_path
    ):
        """Test alternative constructor from_checkpoint"""
        from language_generation.gpt2_brain import GPT2Brain

        # Create and save a model
        model1 = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        model1.save_checkpoint(str(checkpoint_path))

        # Create model from checkpoint
        new_encoder = MockEncoderSingleEmbed(input_channels=64, output_dim=256)
        new_gpt2_model = MockGPT2LMHeadModel()
        new_tokenizer = MockGPT2TokenizerFast()

        model2 = GPT2Brain.from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            lm_model=new_gpt2_model,
            tokenizer=new_tokenizer,
            encoder_model=new_encoder,
            freeze_lm=True,
        )

        assert model2 is not None
        assert hasattr(model2, "encoder_model")
        assert hasattr(model2, "lm_model")


class TestGPT2BrainTargetPredictions:
    """Test target prediction extraction functionality (return_all_preds=False parameter)"""

    def test_forward_with_return_all_preds_false_requires_target_mask(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test that return_all_preds=False requires target_attention_mask"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        # Should raise error without target_attention_mask
        with pytest.raises(ValueError, match="target_attention_mask must be provided"):
            model.forward(
                neural_data=sample_batch["neural_data"],
                all_input_ids=sample_batch["input_ids"],
                all_attention_mask=sample_batch["attention_mask"],
                return_all_preds=False,
            )

    def test_forward_returns_target_preds_padded(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test that return_all_preds=False returns padded target predictions"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        batch_size = sample_batch["neural_data"].shape[0]
        max_target_tokens = 3
        target_attention_mask = torch.zeros(batch_size, max_target_tokens)
        target_attention_mask[0, :2] = 1  # First sample: 2 target tokens
        target_attention_mask[1, :1] = 1  # Second sample: 1 target token

        target_preds = model.forward(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            target_attention_mask=target_attention_mask,
            return_all_preds=False,
        )

        vocab_size = len(model.tokenizer)

        # Should return padded predictions maintaining batch structure
        assert target_preds.shape == (batch_size, max_target_tokens, vocab_size)
        assert target_preds.ndim == 3

    def test_empty_target_mask(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test handling of empty target mask (no target tokens)"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        batch_size = sample_batch["neural_data"].shape[0]
        max_target_tokens = 3
        target_attention_mask = torch.zeros(batch_size, max_target_tokens)

        target_preds = model.forward(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            target_attention_mask=target_attention_mask,
            return_all_preds=False,
        )

        vocab_size = len(model.tokenizer)
        # Should still return padded structure even with no valid targets
        assert target_preds.shape == (batch_size, max_target_tokens, vocab_size)

    def test_full_target_mask(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test with all target tokens valid (no padding)"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
        )

        batch_size = sample_batch["neural_data"].shape[0]
        max_target_tokens = 4
        target_attention_mask = torch.ones(batch_size, max_target_tokens)

        target_preds = model.forward(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            target_attention_mask=target_attention_mask,
            return_all_preds=False,
        )

        vocab_size = len(model.tokenizer)
        assert target_preds.shape == (batch_size, max_target_tokens, vocab_size)


class TestGPT2BrainNoBrainEncoder:
    """Test GPT2Brain with no_brain_encoder flag"""

    def test_instantiation_no_brain_encoder(
        self, mock_gpt2_model, mock_tokenizer
    ):
        """Test that model can be instantiated without encoder when no_brain_encoder=True"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
        )

        assert model is not None
        assert not hasattr(model, "encoder_model")
        assert model.no_brain_encoder is True

    def test_brain_tokens_added_with_no_brain_encoder(
        self, mock_gpt2_model, mock_tokenizer
    ):
        """Test that brain tokens are still added when no_brain_encoder=True"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
        )

        # Brain tokens should still be added by default
        assert "<brain/>" in model.tokenizer.get_vocab()
        assert "</brain>" in model.tokenizer.get_vocab()
        assert hasattr(model, "brain_token_ids")

    def test_forward_no_brain_encoder_single_embed(
        self, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test forward pass with no_brain_encoder produces embeddings with only separator tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
        )

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            return_all_preds=True,
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        vocab_size = len(model.tokenizer)

        # Output should have shape [batch, seq_len + 2, vocab_size]
        # Only 2 separator tokens, no neural embeddings
        expected_seq_len = seq_len + 2  # <brain/> + </brain>
        assert output.logits.shape == (batch_size, expected_seq_len, vocab_size)
        assert updated_attention_mask.shape == (batch_size, expected_seq_len)

    def test_embeddings_structure_no_brain_encoder(
        self, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test that embeddings contain separator tokens but no neural embeddings"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
        )

        prompt_embeddings, prompt_attention_mask = model._convert_to_embeddings(
            sample_batch["neural_data"],
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        hidden_size = model.lm_model.hidden_size

        # Should have: <brain/> + </brain> + input_ids
        expected_len = 2 + seq_len
        assert prompt_embeddings.shape == (batch_size, expected_len, hidden_size)
        assert prompt_attention_mask.shape == (batch_size, expected_len)

        # First 2 positions should be brain separator tokens (all ones in attention)
        assert (prompt_attention_mask[:, :2] == 1).all()


class TestGPT2BrainNoBrainTokenInjection:
    """Test GPT2Brain with no_brain_token_injection flag"""

    def test_instantiation_no_brain_token_injection(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer
    ):
        """Test that model can be instantiated with no_brain_token_injection=True"""
        from language_generation.gpt2_brain import GPT2Brain

        original_vocab_size = len(mock_tokenizer)

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
            no_brain_token_injection=True,
        )

        assert model is not None
        assert model.no_brain_token_injection is True
        assert not hasattr(model, "brain_token_ids")

        # Tokenizer vocab should not have grown (no brain tokens added)
        assert len(model.tokenizer) == original_vocab_size

    def test_no_brain_tokens_added(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer
    ):
        """Test that brain tokens are not added when no_brain_token_injection=True"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
            no_brain_token_injection=True,
        )

        # Brain tokens should NOT be in vocabulary
        assert "<brain/>" not in model.tokenizer.get_vocab()
        assert "</brain>" not in model.tokenizer.get_vocab()

    def test_forward_no_brain_token_injection_single_embed(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test forward pass with no_brain_token_injection produces embeddings without separator tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
            no_brain_token_injection=True,
        )

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            return_all_preds=True,
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        vocab_size = len(model.tokenizer)

        # Output should have shape [batch, seq_len + 1, vocab_size]
        # 1 neural embedding, no separator tokens
        expected_seq_len = seq_len + 1  # neural_embed only
        assert output.logits.shape == (batch_size, expected_seq_len, vocab_size)
        assert updated_attention_mask.shape == (batch_size, expected_seq_len)

    def test_forward_no_brain_token_injection_multi_embed(
        self, mock_encoder_multi, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test forward pass with no_brain_token_injection and multiple embeddings"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_multi,
            freeze_lm=True,
            no_brain_token_injection=True,
        )

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            return_all_preds=True,
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        vocab_size = len(model.tokenizer)
        num_neural_tokens = 3  # From mock_encoder_multi

        # Output should have shape [batch, seq_len + num_neural_tokens, vocab_size]
        # Only neural embeddings, no separator tokens
        expected_seq_len = seq_len + num_neural_tokens
        assert output.logits.shape == (batch_size, expected_seq_len, vocab_size)
        assert updated_attention_mask.shape == (batch_size, expected_seq_len)

    def test_embeddings_structure_no_brain_token_injection(
        self, mock_encoder_single, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test that embeddings contain neural embeddings but no separator tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=mock_encoder_single,
            freeze_lm=True,
            no_brain_token_injection=True,
        )

        prompt_embeddings, prompt_attention_mask = model._convert_to_embeddings(
            sample_batch["neural_data"],
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        hidden_size = model.lm_model.hidden_size

        # Should have: neural_embed + input_ids (no separators)
        expected_len = 1 + seq_len
        assert prompt_embeddings.shape == (batch_size, expected_len, hidden_size)
        assert prompt_attention_mask.shape == (batch_size, expected_len)


class TestGPT2BrainCombinedFlags:
    """Test GPT2Brain with both no_brain_encoder and no_brain_token_injection flags"""

    def test_both_flags_enabled(
        self, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test model with both no_brain_encoder and no_brain_token_injection enabled"""
        from language_generation.gpt2_brain import GPT2Brain

        original_vocab_size = len(mock_tokenizer)

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
            no_brain_token_injection=True,
        )

        assert model is not None
        assert model.no_brain_encoder is True
        assert model.no_brain_token_injection is True
        assert not hasattr(model, "encoder_model")
        assert not hasattr(model, "brain_token_ids")

        # Tokenizer vocab should not have grown
        assert len(model.tokenizer) == original_vocab_size

    def test_forward_both_flags_no_brain_data(
        self, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test forward pass with both flags produces embeddings with only input tokens"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
            no_brain_token_injection=True,
        )

        output, updated_attention_mask = model(
            neural_data=sample_batch["neural_data"],
            all_input_ids=sample_batch["input_ids"],
            all_attention_mask=sample_batch["attention_mask"],
            return_all_preds=True,
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        vocab_size = len(model.tokenizer)

        # Output should have shape [batch, seq_len, vocab_size]
        # No brain embeddings or separator tokens, just input tokens
        assert output.logits.shape == (batch_size, seq_len, vocab_size)
        assert updated_attention_mask.shape == (batch_size, seq_len)

    def test_embeddings_both_flags_vanilla_behavior(
        self, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test that embeddings with both flags behave like vanilla GPT2"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
            no_brain_token_injection=True,
        )

        prompt_embeddings, prompt_attention_mask = model._convert_to_embeddings(
            sample_batch["neural_data"],
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
        )

        batch_size, seq_len = sample_batch["input_ids"].shape
        hidden_size = model.lm_model.hidden_size

        # Should have only input_ids embeddings (no brain data at all)
        assert prompt_embeddings.shape == (batch_size, seq_len, hidden_size)
        assert prompt_attention_mask.shape == (batch_size, seq_len)

        # Attention mask should match original input attention mask
        assert torch.equal(prompt_attention_mask, sample_batch["attention_mask"])

    def test_generate_both_flags(
        self, mock_gpt2_model, mock_tokenizer, sample_batch
    ):
        """Test generation with both flags works like vanilla GPT2"""
        from language_generation.gpt2_brain import GPT2Brain

        model = GPT2Brain(
            lm_model=mock_gpt2_model,
            tokenizer=mock_tokenizer,
            encoder_model=None,
            freeze_lm=True,
            no_brain_encoder=True,
            no_brain_token_injection=True,
        )
        model.eval()

        with torch.no_grad():
            generated_ids = model.generate(
                neural_data=sample_batch["neural_data"],
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
            )

        batch_size, original_seq_len = sample_batch["input_ids"].shape

        # Generated sequences should contain original input + new tokens
        # No brain data involved, behaves like vanilla GPT2
        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] == original_seq_len + 5
