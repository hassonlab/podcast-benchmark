"""
Integration code for the Example Foundation Model

This module demonstrates TWO ways to use a foundation model in the benchmark:

1. **Feature Extraction (Frozen)**: Load a pretrained model, freeze it, and use it to
   extract embeddings during preprocessing. Then train a simple decoder on top.

2. **Finetuning**: Include the foundation model as part of your decoder architecture,
   and continue training it (fully or partially) on your downstream task.

Both patterns are registered with the framework's registry system.
"""

import os
import sys
import types
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from core import registry
from .simple_transformer import (
    load_pretrained_model as load_simple_pretrained_model,
    SimpleTransformer,
)


# =============================================================================
# PATTERN 1: FEATURE EXTRACTION (FROZEN MODEL)
# =============================================================================

@registry.register_data_preprocessor("brainbert_feature_extraction")
def extract_foundation_features(data, preprocessor_params):
    """
    Extract frozen features from the foundation model during preprocessing.

    This pattern:
    1. Loads the pretrained foundation model
    2. Freezes all parameters
    3. Runs the data through the model to extract embeddings
    4. Returns embeddings which are then used to train a simple decoder

    Args:
        data: Neural data of shape [num_samples, num_channels, num_timepoints]
        preprocessor_params: Dictionary with:
            - model_dir: Path to pretrained model directory
            - batch_size: Batch size for processing (default: 32)

    Returns:
        embeddings: Numpy array of shape [num_samples, model_dim]
    """
    model_dir = preprocessor_params["model_dir"]
    batch_size = preprocessor_params.get("batch_size", 32)

    # Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_simple_pretrained_model(model_dir, device=device)
    model.eval()
    model.freeze()

    print(f"Loaded foundation model from {model_dir}")
    print(f"Model has {model.get_num_params():,} parameters (all frozen)")

    # Extract embeddings in batches
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Extracting features"):
            batch = torch.tensor(
                data[i : i + batch_size],
                dtype=torch.float32,
                device=device
            )
            batch_embeddings = model(batch, return_sequence=False)
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Extracted embeddings of shape: {embeddings.shape}")

    return embeddings


# Simple MLP decoder to use on top of frozen features
class MLPDecoder(nn.Module):
    """
    Simple MLP decoder for use with frozen foundation features.

    This is used with PATTERN 1 (feature extraction).
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        output_activation: str = "linear",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # Build layers
        prev_dim = input_dim
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_dim, size))
            if use_layer_norm and size != layer_sizes[-1]:
                self.layer_norms.append(nn.LayerNorm(size))
            prev_dim = size

        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        self.output_activation = output_activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Don't apply activation/dropout to final layer
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        
        # Apply output activation if specified
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)
        elif self.output_activation == "softmax":
            x = F.softmax(x, dim=-1)
        # "linear" means no activation (default)
        
        # Squeeze the output to match the label shape [batch_size] instead of [batch_size, 1]
        # This matches the behavior of neural_conv_decoder
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        return x


@registry.register_data_preprocessor("stft_preprocessing")
def stft_preprocessing(data, preprocessor_params):
    from models.popt.preprocessors.stft import STFTPreprocessor

    stft_preprocessor = STFTPreprocessor(**preprocessor_params)
    stft_preprocessor.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datas_torch = torch.from_numpy(data).to(device)
    stft_preprocessor = stft_preprocessor.to(device)

    with torch.no_grad():
        stft_output = stft_preprocessor(datas_torch)

    datas = stft_output.cpu().numpy()
    print(
        f"Applied STFT preprocessing ({'GPU' if device.type == 'cuda' else 'CPU'}): "
        f"input shape {datas_torch.shape} -> output shape {datas.shape}"
    )
    return datas


def _resolve_training_losses(model_params):
    losses = model_params.get("_training_losses")
    if losses:
        return list(losses)
    loss_name = model_params.get("_loss_name")
    if loss_name:
        return [loss_name]
    return []


def _resolve_output_activation(model_params, output_dim):
    explicit = model_params.get("output_activation")
    if explicit is not None:
        return explicit

    losses = set(_resolve_training_losses(model_params))
    if "bce" in losses:
        return "sigmoid"
    if "soft_bce" in losses:
        return "sigmoid"
    if output_dim > 1 and "softmax_output" in losses:
        return "softmax"
    return "linear"


def _append_preprocessor(data_params, fn_name, params):
    existing_names = data_params.preprocessing_fn_name
    existing_params = data_params.preprocessor_params

    if existing_names is None:
        data_params.preprocessing_fn_name = [fn_name]
        data_params.preprocessor_params = [params]
        return

    if not isinstance(existing_names, list):
        existing_names = [existing_names]
    if existing_params is None:
        existing_params = [None] * len(existing_names)
    elif not isinstance(existing_params, list):
        existing_params = [existing_params]

    if fn_name in existing_names:
        idx = existing_names.index(fn_name)
        while len(existing_params) <= idx:
            existing_params.append(None)
        existing_params[idx] = params
    else:
        existing_names.append(fn_name)
        existing_params.append(params)

    data_params.preprocessing_fn_name = existing_names
    data_params.preprocessor_params = existing_params


def _default_output_dim_for_task(task_name, task_specific_config):
    if task_name in (
        "word_embedding_decoding_task",
        "whisper_embedding_decoding_task",
        "whisper_embedding",
    ):
        return getattr(task_specific_config, "embedding_pca_dim", None) or 50
    if task_name in (
        "gpt_surprise_task",
        "sentence_onset_task",
        "content_noncontent_task",
        "volume_level_decoding_task",
    ):
        return 1
    if task_name == "gpt_surprise_multiclass_task":
        return 3
    if task_name == "pos_task":
        return 5
    return None


def _find_first_model_spec_by_constructor(model_spec, constructor_name):
    if model_spec.constructor_name == constructor_name:
        return model_spec
    for sub_model_spec in model_spec.sub_models.values():
        found = _find_first_model_spec_by_constructor(sub_model_spec, constructor_name)
        if found is not None:
            return found
    return None


def _find_nested_encoder_model_spec(model_spec, encoder_constructor_name):
    encoder_spec = model_spec.sub_models.get("encoder_model")
    while encoder_spec is not None and encoder_spec.constructor_name == "caching_model":
        encoder_spec = encoder_spec.sub_models.get("inner_model")
    if (
        encoder_spec is not None
        and encoder_spec.constructor_name == encoder_constructor_name
    ):
        return encoder_spec
    return None


def _setup_brainbert_path():
    brainbert_root = os.path.dirname(os.path.abspath(__file__))
    brainbert_wrapper = os.path.join(brainbert_root, "BrainBERT")
    if brainbert_wrapper not in sys.path:
        sys.path.insert(0, brainbert_wrapper)
    return brainbert_wrapper


def _dict_to_cfg(d):
    cfg = types.SimpleNamespace()
    for k, v in d.items():
        setattr(cfg, k, v)
    return cfg


def _load_config_yaml(model_dir):
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"BrainBERT config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _config_dict_to_upstream_cfg(config_dict):
    return _dict_to_cfg(
        {
            "name": "masked_tf_model",
            "hidden_dim": config_dict.get("model_dim", 768),
            "layer_dim_feedforward": config_dict.get("dim_feedforward", 3072),
            "layer_activation": config_dict.get("layer_activation", "gelu"),
            "nhead": config_dict.get("num_heads", 12),
            "encoder_num_layers": config_dict.get("num_layers", 6),
            "input_dim": config_dict.get("input_channels", 40),
        }
    )


def _model_params_to_upstream_cfg(model_params):
    return _dict_to_cfg(
        {
            "name": "masked_tf_model",
            "hidden_dim": model_params.get("model_dim", 768),
            "layer_dim_feedforward": model_params.get("dim_feedforward", 3072),
            "layer_activation": model_params.get("layer_activation", "gelu"),
            "nhead": model_params.get("num_heads", 12),
            "encoder_num_layers": model_params.get("num_layers", 6),
            "input_dim": model_params.get("input_channels", 40),
        }
    )


def _find_checkpoint_path(model_dir):
    for name in ("stft_large_pretrained.pth", "checkpoint.pth"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"BrainBERT checkpoint not found in {model_dir}. "
        "Expected stft_large_pretrained.pth or checkpoint.pth"
    )


def _resolve_checkpoint_and_config_dir(model_params):
    foundation_dir = model_params.get("foundation_dir") or model_params.get(
        "checkpoint_path"
    )
    model_dir = model_params.get("model_dir")

    if foundation_dir and os.path.isfile(foundation_dir):
        foundation_dir = os.path.abspath(foundation_dir)
        return foundation_dir, os.path.dirname(foundation_dir)
    if model_dir and os.path.isdir(model_dir):
        return _find_checkpoint_path(model_dir), model_dir
    return None, None


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    return ckpt


def _remap_state_dict_to_reference(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("spec_prediction_head."):
            new_state[k] = v
        elif k.startswith("transformer_encoder."):
            new_state["transformer." + k[len("transformer_encoder.") :]] = v
        elif k.startswith("input_projection."):
            new_state["input_encoding.in_proj." + k[len("input_projection.") :]] = v
        elif k == "pos_encoder.pe" or k.startswith("pos_encoder.pe"):
            v = v.clone()
            if v.dim() == 3 and v.shape[1] == 1:
                v = v.transpose(0, 1)
            new_state["input_encoding.positional_encoding.pe"] = v
        elif k.startswith("layer_norm."):
            new_state["input_encoding.layer_norm." + k[len("layer_norm.") :]] = v
        else:
            new_state[k] = v
    return new_state


def _get_upstream_cfg_from_checkpoint(ckpt):
    if not isinstance(ckpt, dict) or "model_cfg" not in ckpt:
        return None
    cfg = ckpt["model_cfg"]
    if hasattr(cfg, "name"):
        if getattr(cfg, "name") == "debug_model":
            cfg.name = "masked_tf_model"
        return cfg
    return _dict_to_cfg(cfg) if isinstance(cfg, dict) else None


def load_reference_pretrained_model(foundation_dir_or_model_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _setup_brainbert_path()

    if foundation_dir_or_model_dir and os.path.isfile(foundation_dir_or_model_dir):
        ckpt_path = foundation_dir_or_model_dir
        config_dir = os.path.dirname(ckpt_path)
    elif foundation_dir_or_model_dir and os.path.isdir(foundation_dir_or_model_dir):
        ckpt_path = _find_checkpoint_path(foundation_dir_or_model_dir)
        config_dir = foundation_dir_or_model_dir
    else:
        raise FileNotFoundError(
            "BrainBERT load_reference_pretrained_model requires foundation_dir or model_dir."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    upstream_cfg = _get_upstream_cfg_from_checkpoint(ckpt)
    if upstream_cfg is None:
        upstream_cfg = _config_dict_to_upstream_cfg(_load_config_yaml(config_dir))

    original_modules = {}
    for name in list(sys.modules.keys()):
        if name in ("models", "utils") or name.startswith("models.") or name.startswith(
            "utils."
        ):
            original_modules[name] = sys.modules[name]
            del sys.modules[name]

    try:
        from models import build_model

        upstream = build_model(upstream_cfg)
        states = _extract_state_dict(ckpt)
        try:
            upstream.load_state_dict(states, strict=True)
        except Exception:
            upstream.load_state_dict(_remap_state_dict_to_reference(states), strict=False)
        upstream.to(device)
        return upstream
    finally:
        for name, mod in original_modules.items():
            sys.modules[name] = mod


load_pretrained_model = load_reference_pretrained_model


class ReferenceBrainBERTEncoder(nn.Module):
    def __init__(
        self,
        upstream,
        frozen_upstream=False,
        num_electrodes=None,
        hidden_dim=768,
    ):
        super().__init__()
        self.upstream = upstream
        self.frozen_upstream = frozen_upstream
        self.num_electrodes = num_electrodes
        self.hidden_dim = hidden_dim

    def forward(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(
                "BrainBERT finetuning expects STFT input with shape [batch, channels, time, freq]."
            )

        batch_size, num_channels, time_steps, freq_channels = x.shape
        inputs = x.contiguous().view(batch_size * num_channels, time_steps, freq_channels)
        pad_mask = None

        if self.frozen_upstream:
            self.upstream.eval()
            with torch.no_grad():
                features = self.upstream(inputs, pad_mask, intermediate_rep=True)
        else:
            features = self.upstream(inputs, pad_mask, intermediate_rep=True)

        if features.shape[0] == batch_size * num_channels:
            seq_len = features.shape[1]
            middle = seq_len // 2
            start = max(0, middle - 5)
            end = min(seq_len, middle + 5)
            if end <= start:
                pooled = features.mean(dim=1)
            else:
                pooled = features[:, start:end, :].mean(dim=1)
        else:
            pooled = features.mean(dim=0)

        pooled = pooled.view(batch_size, num_channels, -1)
        if self.num_electrodes is not None and self.hidden_dim is not None:
            return pooled.reshape(batch_size, -1)
        return pooled.mean(dim=1)


class ReferenceBrainBERTHead(nn.Module):
    def __init__(self, head_model=None):
        super().__init__()
        self.head_model = head_model if head_model is not None else nn.Identity()

    def forward(self, x, **kwargs):
        return self.head_model(x)


class ReferenceBrainBERTDecoder(nn.Module):
    def __init__(
        self,
        encoder_model,
        projector,
        output_dim=1,
        output_activation="linear",
        finetune_model=None,
    ):
        super().__init__()
        self.encoder_model = encoder_model
        self.projector = projector
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.finetune_model = finetune_model

    def forward(self, x, **kwargs):
        latents = self.encoder_model(x, **kwargs)
        out = self.projector(latents)

        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.output_activation == "softmax":
            out = F.softmax(out, dim=-1)

        if self.output_dim == 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


@registry.register_model_constructor("brainbert_mlp")
def create_mlp_decoder(model_params):
    """
    Create MLP decoder for frozen foundation features.

    This is used with PATTERN 1 (feature extraction).

    Expected model_params:
        - input_dim: Dimension of foundation model embeddings
        - layer_sizes: List of layer sizes for MLP
        - dropout: Dropout probability (optional)
        - use_layer_norm: Whether to use layer normalization (optional)
    """
    return MLPDecoder(
        input_dim=model_params["input_dim"],
        layer_sizes=model_params["layer_sizes"],
        dropout=model_params.get("dropout", 0.0),
        use_layer_norm=model_params.get("use_layer_norm", False),
    )


# =============================================================================
# PATTERN 2: FINETUNING (TRAINABLE MODEL)
# =============================================================================

class BrainBERTDecoder(nn.Module):
    """
    Decoder that includes the BrainBERT model as a trainable submodule.

    This pattern:
    1. Loads the BrainBERT model with pretrained weights
    2. Optionally freezes some layers (partial finetuning)
    3. Includes the BrainBERT model in the decoder architecture
    4. During training, gradients flow through unfrozen parts

    This is used with PATTERN 2 (finetuning).
    """

    def __init__(
        self,
        model_dir: str,
        output_dim: int,
        mlp_layer_sizes: list[int],
        freeze_foundation: bool = False,
        num_frozen_layers: int = 0,
        dropout: float = 0.0,
        input_channels: int = None,
        output_activation: str = "linear",
    ):
        """
        Args:
            model_dir: Path to pretrained foundation model directory
            output_dim: Output dimension for final predictions
            mlp_layer_sizes: Layer sizes for decoder head MLP
            freeze_foundation: If True, freeze entire foundation model
            num_frozen_layers: Number of foundation layers to freeze (0 = none)
            dropout: Dropout probability
            input_channels: Number of input channels (if different from pretrained model)
            output_activation: Activation function for output layer ("linear", "sigmoid", "tanh", "softmax")
        """
        super().__init__()

        # Load pretrained foundation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.foundation_model = load_pretrained_model(model_dir, device=device)
        
        # Fix: If input_channels is provided and different from pretrained model,
        # reinitialize the input_projection layer
        if input_channels is not None and input_channels != self.foundation_model.input_channels:
            print(f"Reinitializing input_projection: {self.foundation_model.input_channels} -> {input_channels}")
            model_dim = self.foundation_model.model_dim
            self.foundation_model.input_projection = nn.Linear(input_channels, model_dim)
            self.foundation_model.input_channels = input_channels

        # Handle freezing
        if freeze_foundation:
            self.foundation_model.freeze()
            print("Foundation model completely frozen")
        elif num_frozen_layers > 0:
            self.foundation_model.freeze_layers(num_frozen_layers)
            print(f"Froze first {num_frozen_layers} layers of foundation model")
        else:
            print("Foundation model fully trainable")

        # Decoder head (MLP on top of foundation features)
        foundation_dim = self.foundation_model.model_dim
        self.decoder_head = MLPDecoder(
            input_dim=foundation_dim,
            layer_sizes=mlp_layer_sizes + [output_dim],
            dropout=dropout,
            use_layer_norm=True,
            output_activation=output_activation,
        )

    def forward(self, x, **kwargs):
        """
        Forward pass through foundation model and decoder head.

        Args:
            x: Input tensor 
                - [batch_size, num_channels, seq_len] for raw signals
                - [batch_size, num_channels, time_stft, freq_channels] for STFT preprocessed data
            **kwargs: Additional keyword arguments (e.g., preserve_ensemble for word embedding tasks)

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Check if input is already STFT preprocessed (4D tensor)
        if len(x.shape) == 4:
            # STFT preprocessed: [batch, channels, time_stft, freq_channels]
            batch_size, num_channels, time_stft, freq_channels = x.shape
            
            # Reshape: [batch, channels, time_stft, freq_channels] → [batch*channels, time_stft, freq_channels]
            x_stft = x.contiguous().view(batch_size * num_channels, time_stft, freq_channels)
            
            # Convert to BrainBERT input shape: [batch*channels, time_stft, freq_channels] → [batch*channels, freq_channels, time_stft]
            # BrainBERT expects [batch, channels, time] where channels=freq_channels (40)
            x_brainbert = x_stft.transpose(1, 2)  # [batch*channels, freq_channels, time_stft]
            
            # Get features from foundation model
            features_seq = self.foundation_model(x_brainbert, return_sequence=True)  # [batch*channels, time_stft, model_dim]
            
            # Aggregate over time dimension (mean pooling)
            features = features_seq.mean(dim=1)  # [batch*channels, model_dim]
            
            # Reshape back: [batch*channels, model_dim] → [batch, channels, model_dim]
            features = features.view(batch_size, num_channels, -1)
            
            # Aggregate over channels (mean pooling) to get per-sample features
            features = features.mean(dim=1)  # [batch, model_dim]
        else:
            # Raw signal: [batch_size, num_channels, seq_len]
            # Get features from foundation model
            features = self.foundation_model(x, return_sequence=False)

        # Pass through decoder head
        output = self.decoder_head(features)

        return output


@registry.register_model_constructor("brainbert_finetune")
def create_finetuning_decoder(model_params):
    """
    Create decoder with foundation model for finetuning.

    This is used with PATTERN 2 (finetuning).

    Expected model_params:
        - model_dir: Path to pretrained model directory
        - output_dim: Output dimension
        - mlp_layer_sizes: Layer sizes for decoder head
        - freeze_foundation: Whether to freeze entire foundation (optional)
        - num_frozen_layers: Number of layers to freeze (optional)
        - dropout: Dropout probability (optional)
        - input_channels: Number of input channels (optional, will be set by config setter)
        - output_activation: Output activation function (optional, auto-determined if not provided)
    """
    output_dim = model_params.get("output_dim", 1)
    frozen_upstream = model_params.get("frozen_upstream", False) or model_params.get(
        "freeze_foundation", False
    )
    mlp_layer_sizes = model_params.get("mlp_layer_sizes", [])
    dropout = model_params.get("dropout", 0.0)
    num_electrodes = model_params.get("num_electrodes")

    ckpt_path, config_dir = _resolve_checkpoint_and_config_dir(model_params)
    random_init = ckpt_path is None
    _setup_brainbert_path()

    if random_init:
        if not any(
            model_params.get(k) is not None for k in ("model_dim", "num_layers", "num_heads")
        ):
            raise ValueError(
                "BrainBERT random init requires model_dim, num_layers, and num_heads in model_params."
            )
        upstream_cfg = _model_params_to_upstream_cfg(model_params)
        ckpt = None
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        upstream_cfg = _get_upstream_cfg_from_checkpoint(ckpt)
        if upstream_cfg is None:
            if any(
                model_params.get(k) is not None for k in ("model_dim", "num_layers", "num_heads")
            ):
                upstream_cfg = _model_params_to_upstream_cfg(model_params)
            elif config_dir:
                upstream_cfg = _config_dict_to_upstream_cfg(_load_config_yaml(config_dir))
            else:
                raise ValueError(
                    "BrainBERT checkpoint has no model_cfg and no config.yaml/model_params were provided."
                )

    original_modules = {}
    for name in list(sys.modules.keys()):
        if name in ("models", "utils") or name.startswith("models.") or name.startswith(
            "utils."
        ):
            original_modules[name] = sys.modules[name]
            del sys.modules[name]

    try:
        from models import build_model

        upstream = build_model(upstream_cfg)
        if ckpt is not None:
            states = _extract_state_dict(ckpt)
            try:
                upstream.load_state_dict(states, strict=True)
            except Exception:
                upstream.load_state_dict(_remap_state_dict_to_reference(states), strict=False)

        hidden_dim = getattr(upstream_cfg, "hidden_dim", 768)
        encoder_model = model_params.get("encoder_model")
        if encoder_model is None:
            encoder_model = ReferenceBrainBERTEncoder(
                upstream=upstream,
                frozen_upstream=frozen_upstream,
                num_electrodes=num_electrodes,
                hidden_dim=hidden_dim,
            )

        if not num_electrodes:
            finetune_cfg = _dict_to_cfg(
                {
                    "name": "finetune_model",
                    "frozen_upstream": frozen_upstream,
                    "hidden_dim": hidden_dim,
                }
            )
            finetune_model = build_model(finetune_cfg, upstream)
            if mlp_layer_sizes:
                head_model = MLPDecoder(
                    input_dim=hidden_dim,
                    layer_sizes=mlp_layer_sizes + [output_dim],
                    dropout=dropout,
                    use_layer_norm=True,
                    output_activation="linear",
                )
            else:
                head_model = nn.Linear(hidden_dim, output_dim)
            finetune_model.linear_out = head_model
        else:
            finetune_model = None
            input_dim = num_electrodes * hidden_dim
            if mlp_layer_sizes:
                layers = []
                curr_dim = input_dim
                for h_dim in mlp_layer_sizes:
                    layers.append(nn.Linear(curr_dim, h_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    curr_dim = h_dim
                layers.append(nn.Linear(curr_dim, 1 if output_dim == 1 else output_dim))
                head_model = nn.Sequential(*layers)
            else:
                head_model = nn.Linear(input_dim, output_dim)
                nn.init.normal_(head_model.weight, mean=0.0, std=0.001)
                nn.init.zeros_(head_model.bias)

        decoder = ReferenceBrainBERTDecoder(
            encoder_model=encoder_model,
            projector=ReferenceBrainBERTHead(head_model),
            output_dim=output_dim,
            output_activation=_resolve_output_activation(model_params, output_dim),
            finetune_model=finetune_model,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder.to(device)
        return decoder
    finally:
        for name, mod in original_modules.items():
            sys.modules[name] = mod


@registry.register_model_constructor("brainbert_encoder")
def create_brainbert_encoder(model_params):
    decoder = create_finetuning_decoder(dict(model_params))
    return decoder.encoder_model


# =============================================================================
# CONFIG SETTERS
# =============================================================================

@registry.register_config_setter("brainbert_feature_extraction")
def set_feature_extraction_config(experiment_config, raws, _df_word):
    """
    Config setter for feature extraction pattern.

    Sets the input_dim for the MLP based on the foundation model's dimension.
    """
    from .config import load_config

    model_params = experiment_config.model_spec.params
    data_params = experiment_config.task_config.data_params

    model_dir = model_params["model_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    foundation_config = load_config(config_path)

    # Set input_dim to match foundation model output
    model_params["input_dim"] = foundation_config.model_dim

    # Set preprocessor params
    if not data_params.preprocessor_params:
        data_params.preprocessor_params = {}
    data_params.preprocessor_params["model_dir"] = model_dir

    return experiment_config


@registry.register_config_setter("brainbert_finetune")
def set_finetuning_config(experiment_config, raws, _df_word):
    """
    Config setter for finetuning pattern.

    Sets the output_dim and loads foundation model config.
    BrainBERT expects STFT features (input_channels=40), so STFT preprocessing is enabled.
    """
    from models.shared_config_setters import set_input_channels

    experiment_config = set_input_channels(
        experiment_config, raws, _df_word, ["brainbert_finetune"]
    )

    target_spec = _find_first_model_spec_by_constructor(
        experiment_config.model_spec, "brainbert_finetune"
    )
    if target_spec is None:
        raise ValueError("Could not find brainbert_finetune model spec.")

    model_params = target_spec.params
    data_params = experiment_config.task_config.data_params
    task_name = experiment_config.task_config.task_name
    task_specific_config = experiment_config.task_config.task_specific_config

    original_channels = model_params.get("input_channels")
    if original_channels is not None:
        model_params["num_electrodes"] = original_channels

    sample_rate = (
        data_params.target_sr
        or model_params.get("sample_rate")
        or int(raws[0].info["sfreq"])
        if raws
        else 512
    )
    stft_config = dict(
        model_params.get("stft_config")
        or data_params.stft_config
        or {
            "freq_channel_cutoff": 40,
            "nperseg": 400,
            "noverlap": 350,
            "normalizing": "zscore",
        }
    )
    stft_config.setdefault("fs", int(sample_rate))

    _append_preprocessor(data_params, "stft_preprocessing", stft_config)
    data_params.use_stft_preprocessing = True
    data_params.stft_config = stft_config

    foundation_dir = model_params.get("foundation_dir") or model_params.get("checkpoint_path")
    model_dir = model_params.get("model_dir")
    config_dir = None
    if foundation_dir and os.path.isfile(foundation_dir):
        config_dir = os.path.dirname(foundation_dir)
    elif model_dir and os.path.isdir(model_dir):
        config_dir = model_dir

    if data_params.window_width is None or data_params.window_width <= 0:
        window_width = model_params.get("window_width")
        if window_width is None and config_dir:
            try:
                window_width = _load_config_yaml(config_dir).get("window_width")
            except FileNotFoundError:
                window_width = None
        data_params.window_width = window_width or 1.0

    if model_params.get("output_dim") is None:
        output_dim = _default_output_dim_for_task(task_name, task_specific_config)
        if output_dim is not None:
            model_params["output_dim"] = output_dim

    losses = experiment_config.training_params.losses or []
    if not losses and experiment_config.training_params.loss_name:
        losses = [experiment_config.training_params.loss_name]
    model_params["_training_losses"] = losses
    model_params["_loss_name"] = experiment_config.training_params.loss_name
    model_params["output_activation"] = _resolve_output_activation(
        model_params, model_params.get("output_dim", 1)
    )

    model_params["input_channels"] = stft_config.get("freq_channel_cutoff", 40)
    model_params["sample_rate"] = int(sample_rate)
    if model_params.get("output_dim") is not None:
        model_params["embedding_dim"] = model_params["output_dim"]

    encoder_spec = _find_nested_encoder_model_spec(target_spec, "brainbert_encoder")
    if encoder_spec is not None:
        encoder_spec.params.update(model_params)

    return experiment_config
