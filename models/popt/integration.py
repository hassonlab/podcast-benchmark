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


@registry.register_data_preprocessor("population_transformer_feature_extraction")
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
                data[i : i + batch_size], dtype=torch.float32, device=device
            )
            batch_embeddings = model(batch, return_sequence=False)
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Extracted embeddings of shape: {embeddings.shape}")

    return embeddings


@registry.register_config_setter("set_sample_rate_for_stft")
def set_sample_rate_for_stft(experiment_config, raws, _df_word):
    """
    Config setter to set sample_rate in data_params for STFT preprocessing.

    This is used to ensure that the STFT preprocessor uses the correct sampling rate.
    """
    # Assume all raws have the same sampling rate
    sample_rate = int(raws[0].info["sfreq"])
    data_params = experiment_config.task_config.data_params
    if not data_params.preprocessor_params:
        # Defaults from popT paper.
        data_params.preprocessor_params = {
            "fs": sample_rate,
            "freq_channel_cutoff": 40,
            "nperseg": 400,
            "noverlap": 350,
            "normalizing": "zscore",
        }
    data_params.preprocessor_params["sample_rate"] = sample_rate
    print(f"Set sample_rate for STFT preprocessing to {sample_rate} Hz")
    return experiment_config


@registry.register_data_preprocessor("stft_preprocessing")
def stft_preprocessing(data, preprocessor_params):
    # Import STFT preprocessor
    from models.popt.preprocessors.stft import STFTPreprocessor

    preprocessor_params = dict(preprocessor_params)
    chunk_size = int(
        preprocessor_params.pop("stft_chunk_size", None)
        or preprocessor_params.pop("batch_size", None)
        or 4
    )

    # Initialize STFT preprocessor
    stft_preprocessor = STFTPreprocessor(**preprocessor_params)
    stft_preprocessor.eval()

    # Apply STFT to all data: [batch, channels, time] → [batch, channels, time_stft, freq_channels]
    # Check if GPU is available for STFT acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU STFT over the entire dataset can easily blow up memory because preprocessing
    # runs before DataLoader minibatching. Process sample chunks instead.
    if data.ndim == 2:
        datas_torch = torch.from_numpy(data).to(device)
        stft_preprocessor = stft_preprocessor.to(device)
        with torch.no_grad():
            stft_output = stft_preprocessor(datas_torch)
        datas = stft_output.cpu().numpy()
        print(
            f"Applied STFT preprocessing ({'GPU' if device.type == 'cuda' else 'CPU'}): "
            f"input shape {datas_torch.shape} → output shape {datas.shape}"
        )
        return datas

    stft_preprocessor = stft_preprocessor.to(device)
    stft_chunks = []
    for start_idx in range(0, len(data), chunk_size):
        chunk = torch.from_numpy(data[start_idx : start_idx + chunk_size]).to(device)
        with torch.no_grad():
            chunk_out = stft_preprocessor(chunk)
        stft_chunks.append(chunk_out.cpu())
        del chunk, chunk_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    datas = torch.cat(stft_chunks, dim=0).numpy()

    print(
        f"Applied STFT preprocessing in chunks ({'GPU' if device.type == 'cuda' else 'CPU'}, chunk_size={chunk_size}): "
        f"input shape {data.shape} → output shape {datas.shape}"
    )

    return datas


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


def _resolve_training_losses(model_params):
    losses = model_params.get("_training_losses")
    if losses:
        return list(losses)
    loss_name = model_params.get("_loss_name")
    if loss_name:
        return [loss_name]
    return []


def _resolve_output_activation(model_params):
    explicit = model_params.get("output_activation")
    if explicit is not None:
        return explicit

    losses = set(_resolve_training_losses(model_params))
    if "bce" in losses or "soft_bce" in losses:
        return "sigmoid"
    if "softmax_output" in losses:
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


def _setup_popt_path():
    popt_root = os.path.dirname(os.path.abspath(__file__))
    popt_wrapper = os.path.join(popt_root, "PopulationTransformer")
    if popt_wrapper not in sys.path:
        sys.path.insert(0, popt_wrapper)
    return popt_wrapper


def _dict_to_cfg(d):
    cfg = types.SimpleNamespace()
    for k, v in d.items():
        setattr(cfg, k, v)
    return cfg


def _load_config_yaml(model_dir):
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"POPT config not found: {config_path}")
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
            "hidden_dim": model_params.get("popt_model_dim")
            or model_params.get("model_dim", 768),
            "layer_dim_feedforward": model_params.get("popt_dim_feedforward")
            or model_params.get("dim_feedforward", 3072),
            "layer_activation": model_params.get("layer_activation", "gelu"),
            "nhead": model_params.get("popt_num_heads")
            or model_params.get("num_heads", 12),
            "encoder_num_layers": model_params.get("popt_num_layers")
            or model_params.get("num_layers", 6),
            "input_dim": model_params.get("input_channels", 40),
        }
    )


def _find_checkpoint_path(model_dir):
    for name in (
        "pretrained_popt_brainbert_stft.pth",
        "checkpoint.pth",
        "stft_large_pretrained.pth",
    ):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"POPT checkpoint not found in {model_dir}. "
        "Expected pretrained_popt_brainbert_stft.pth, checkpoint.pth, or stft_large_pretrained.pth"
    )


def _resolve_checkpoint_and_config_dir(model_params):
    foundation_dir = (
        model_params.get("popt_foundation_dir")
        or model_params.get("foundation_dir")
        or model_params.get("checkpoint_path")
    )
    model_dir = model_params.get("popt_model_dir") or model_params.get("model_dir")

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


def _cfg_lookup(cfg, *names):
    """Read the first available key/attribute from heterogeneous cfg objects."""
    if cfg is None:
        return None
    for name in names:
        try:
            if isinstance(cfg, dict) and name in cfg:
                value = cfg.get(name)
            elif hasattr(cfg, name):
                value = getattr(cfg, name)
            elif hasattr(cfg, "get"):
                value = cfg.get(name)
            else:
                continue
        except Exception:
            continue
        if value is not None:
            return value
    return None


def _normalize_upstream_cfg(
    upstream_cfg,
    *,
    model_params,
    config_dir,
    use_brainbert,
    use_lip_coords,
):
    """
    Canonicalize checkpoint/config/model_params into the schema expected by
    PopulationTransformer's build_model():
      name, hidden_dim, layer_dim_feedforward, layer_activation,
      nhead, encoder_num_layers, input_dim, position_encoding(optional).

    The pretrained POPT checkpoint ships an OmegaConf model_cfg using keys like
    ``pt_model_custom``, ``n_head`` and ``n_layers``. The benchmark port also
    rewrites cfg objects when BrainBERT is enabled. Normalize everything here so
    downstream builder code sees a stable schema.
    """
    config_dict = {}
    if config_dir:
        try:
            config_dict = _load_config_yaml(config_dir) or {}
        except FileNotFoundError:
            config_dict = {}

    normalized = {
        # Only masked_tf_model is registered in the embedded PopulationTransformer code.
        "name": "masked_tf_model",
        "hidden_dim": _cfg_lookup(upstream_cfg, "hidden_dim", "model_dim")
        or config_dict.get("model_dim")
        or model_params.get("popt_model_dim")
        or model_params.get("model_dim")
        or 768,
        "layer_dim_feedforward": _cfg_lookup(
            upstream_cfg, "layer_dim_feedforward", "dim_feedforward"
        )
        or config_dict.get("dim_feedforward")
        or model_params.get("popt_dim_feedforward")
        or model_params.get("dim_feedforward")
        or 3072,
        "layer_activation": _cfg_lookup(upstream_cfg, "layer_activation")
        or config_dict.get("layer_activation")
        or model_params.get("layer_activation")
        or "gelu",
        "nhead": _cfg_lookup(upstream_cfg, "nhead", "n_head", "num_heads")
        or config_dict.get("num_heads")
        or model_params.get("popt_num_heads")
        or model_params.get("num_heads")
        or 12,
        "encoder_num_layers": _cfg_lookup(
            upstream_cfg, "encoder_num_layers", "n_layers", "num_layers"
        )
        or config_dict.get("num_layers")
        or model_params.get("popt_num_layers")
        or model_params.get("num_layers")
        or 6,
        "input_dim": BRAINBERT_OUTPUT_DIM
        if use_brainbert
        else _cfg_lookup(upstream_cfg, "input_dim", "input_channels")
        or config_dict.get("input_channels")
        or model_params.get("input_channels")
        or 40,
    }

    position_encoding = "multi_subj_position_encoding" if use_lip_coords else None
    if position_encoding is not None:
        normalized["position_encoding"] = position_encoding

    target_dim = _cfg_lookup(upstream_cfg, "target_dim")
    if target_dim is not None:
        normalized["target_dim"] = target_dim

    return _dict_to_cfg(normalized)


def _prepare_upstream_state_dict(state_dict, *, drop_position_encoding=False):
    prepared = dict(state_dict)
    if drop_position_encoding:
        for key in list(prepared.keys()):
            if "positional_encoding.pe" in key:
                prepared.pop(key, None)
    return prepared


def load_reference_pretrained_model(foundation_dir_or_model_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _setup_popt_path()
    if foundation_dir_or_model_dir and os.path.isfile(foundation_dir_or_model_dir):
        ckpt_path = foundation_dir_or_model_dir
        config_dir = os.path.dirname(ckpt_path)
    elif foundation_dir_or_model_dir and os.path.isdir(foundation_dir_or_model_dir):
        ckpt_path = _find_checkpoint_path(foundation_dir_or_model_dir)
        config_dir = foundation_dir_or_model_dir
    else:
        raise FileNotFoundError(
            "POPT load_reference_pretrained_model requires foundation_dir or model_dir."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    upstream_cfg = _get_upstream_cfg_from_checkpoint(ckpt)
    if upstream_cfg is None:
        upstream_cfg = _config_dict_to_upstream_cfg(_load_config_yaml(config_dir))
    upstream_cfg = _normalize_upstream_cfg(
        upstream_cfg,
        model_params={},
        config_dir=config_dir,
        use_brainbert=False,
        use_lip_coords=_cfg_lookup(upstream_cfg, "position_encoding")
        == "multi_subj_position_encoding",
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


BRAINBERT_OUTPUT_DIM = 768
BRAINBERT_MIDDLE_WINDOW = 5


def _load_brainbert_upstream(brainbert_foundation_dir):
    from models.brainbert import integration as brainbert_integration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = brainbert_integration.load_reference_pretrained_model(
        brainbert_foundation_dir, device=device
    )
    upstream.eval()
    for p in upstream.parameters():
        p.requires_grad = False
    return upstream


class ReferencePOPTDecoder(nn.Module):
    def __init__(
        self,
        upstream,
        output_dim=1,
        hidden_dim=768,
        mlp_layer_sizes=None,
        dropout=0.0,
        input_dim=40,
        brainbert_upstream=None,
        use_lip_coords=False,
        brainbert_electrode_sequence=True,
        output_activation="linear",
    ):
        super().__init__()
        self.upstream = upstream
        self.brainbert_upstream = brainbert_upstream
        self.use_lip_coords = use_lip_coords
        self.brainbert_electrode_sequence = brainbert_electrode_sequence
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.cls_dim = BRAINBERT_OUTPUT_DIM if brainbert_upstream is not None else input_dim
        self.classifier_norm = nn.LayerNorm(hidden_dim)
        self.head = MLPDecoder(
            input_dim=hidden_dim,
            layer_sizes=(mlp_layer_sizes or []) + [output_dim],
            dropout=dropout,
            use_layer_norm=True,
            output_activation=output_activation,
        )

    def _make_cls_token(self, batch_size, device, dtype):
        return torch.ones(batch_size, 1, self.cls_dim, device=device, dtype=dtype)

    def _get_positions(self, lip_coords, batch_size, num_channels, device):
        if not self.use_lip_coords:
            return None
        if lip_coords is None:
            raise ValueError("lip_coords is required when use_lip_coords=True")
        lip_coords = lip_coords.to(device=device, dtype=torch.long)
        if lip_coords.ndim != 3 or lip_coords.shape[1] != num_channels:
            raise ValueError(
                f"Expected lip_coords shape [batch, {num_channels}, 3], got {tuple(lip_coords.shape)}"
            )
        seq_ids = torch.zeros(
            batch_size, num_channels, dtype=torch.long, device=device
        )
        return lip_coords, seq_ids

    def forward(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(
                "PopT finetuning expects STFT input with shape [batch, channels, time, freq]."
            )

        lip_coords = kwargs.get("lip_coords")
        batch_size, num_channels, time_steps, freq_channels = x.shape
        inputs = x.contiguous().view(batch_size * num_channels, time_steps, freq_channels)
        pad_mask = None

        if self.brainbert_upstream is not None:
            self.brainbert_upstream.eval()
            with torch.no_grad():
                features = self.brainbert_upstream(inputs, pad_mask, intermediate_rep=True)

            if self.brainbert_electrode_sequence:
                middle = features.shape[1] // 2
                start = max(0, middle - BRAINBERT_MIDDLE_WINDOW)
                end = min(features.shape[1], middle + BRAINBERT_MIDDLE_WINDOW)
                pooled = features[:, start:end, :].mean(dim=1)
                seq = pooled.view(batch_size, num_channels, -1)
                cls = self._make_cls_token(batch_size, seq.device, seq.dtype)
                seq = torch.cat([cls, seq], dim=1)
                positions = self._get_positions(
                    lip_coords, batch_size, num_channels, seq.device
                )
                encoded = self.upstream(
                    seq, pad_mask, intermediate_rep=True, positions=positions
                )
                cls_repr = encoded[:, 0, :]
            else:
                if self.use_lip_coords:
                    raise ValueError(
                        "use_lip_coords requires brainbert_electrode_sequence=True for PopT."
                    )
                cls = self._make_cls_token(
                    batch_size * num_channels, features.device, features.dtype
                )
                seq = torch.cat([cls, features], dim=1)
                encoded = self.upstream(seq, pad_mask, intermediate_rep=True)
                cls_repr = encoded[:, 0, :].view(batch_size, num_channels, -1).mean(dim=1)
        else:
            if self.use_lip_coords:
                raise ValueError("use_lip_coords requires use_brainbert=True in this port.")
            cls = self._make_cls_token(
                batch_size * num_channels, inputs.device, inputs.dtype
            )
            seq = torch.cat([cls, inputs], dim=1)
            encoded = self.upstream(seq, pad_mask, intermediate_rep=True)
            cls_repr = encoded[:, 0, :].view(batch_size, num_channels, -1).mean(dim=1)

        cls_repr = self.classifier_norm(cls_repr)
        if kwargs.get('return_feature_emb_instead_of_projection', False):
            #* used for feature caching
            assert self.output_activation not in ['sigmoid','softmax', 'tanh'], "Output activation not impelmented since it needs to do the finetune model then that thing, which we currently don't implement in the decoding_utils.py"
            return cls_repr
        return self.head(cls_repr)


@registry.register_model_constructor("population_transformer_mlp")
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


class PopulationTransformerDecoder(nn.Module):
    """
    Decoder that includes the PopulationTransformer model as a trainable submodule.

    This pattern:
    1. Loads the PopulationTransformer model with pretrained weights
    2. Optionally freezes some layers (partial finetuning)
    3. Includes the PopulationTransformer model in the decoder architecture
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
        use_lip_coords: bool = False,
        max_coord_value: int = 5000,
        use_brainbert: bool = True,
        brainbert_model_dir: str = None,
        stft_config: dict = None,
        sample_rate: int = 2048,
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
            use_lip_coords: If True, use LIP coordinates for positional encoding
            max_coord_value: Maximum coordinate value for PE table
            use_brainbert: If True, use BrainBERT to process time sequences (default: True, matching original PopT)
            brainbert_model_dir: Path to BrainBERT pretrained model directory (default: models/brainbert/pretrained_model)
            stft_config: STFT preprocessing configuration dict. If None, uses default values matching original PopT.
            sample_rate: Sampling rate (Hz) for STFT preprocessing. Default: 2048
        """
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BrainBERT model if needed
        self.use_brainbert = use_brainbert
        if use_brainbert:
            if brainbert_model_dir is None:
                # Default BrainBERT path
                brainbert_model_dir = "models/brainbert/pretrained_model"

            try:
                from models.brainbert.simple_transformer import (
                    load_pretrained_model as load_brainbert,
                )

                self.brainbert = load_brainbert(brainbert_model_dir, device=device)
                self.brainbert.eval()
                # BrainBERT is pretrained, so freeze it
                for param in self.brainbert.parameters():
                    param.requires_grad = False
                print(f"Loaded BrainBERT from {brainbert_model_dir}")

                # Initialize STFT preprocessor (matching original PopT)
                if stft_config is None:
                    # Default STFT configuration matching original PopT
                    stft_config = {
                        "fs": sample_rate,
                        "freq_channel_cutoff": 40,
                        "nperseg": 400,
                        "noverlap": 350,
                        "normalizing": "zscore",
                    }

                from .preprocessors.stft import STFTPreprocessor

                self.stft_preprocessor = STFTPreprocessor(**stft_config)
                self.stft_preprocessor.eval()
                # STFT is deterministic, no gradients needed
                for param in self.stft_preprocessor.parameters():
                    param.requires_grad = False
                print(
                    f"Initialized STFT preprocessor with fs={sample_rate}, freq_channels=40"
                )

            except Exception as e:
                print(
                    f"Warning: Failed to load BrainBERT from {brainbert_model_dir}: {e}"
                )
                print("Falling back to direct input processing (use_brainbert=False)")
                self.use_brainbert = False
                self.brainbert = None
                self.stft_preprocessor = None
        else:
            self.brainbert = None
            self.stft_preprocessor = None

        # Load pretrained foundation model
        self.foundation_model = load_pretrained_model(model_dir, device=device)

        # If use_lip_coords is True, we need to replace the positional encoder
        if use_lip_coords:
            from .simple_transformer import MultiSubjBrainPositionalEncoding

            self.foundation_model.use_lip_coords = True
            self.foundation_model.pos_encoder = MultiSubjBrainPositionalEncoding(
                self.foundation_model.model_dim, dropout, max_coord_value
            )
            # Move to device
            self.foundation_model.pos_encoder.to(device)
        else:
            self.foundation_model.use_lip_coords = False

        self.use_lip_coords = use_lip_coords

        # Handle input_projection based on whether BrainBERT is used
        if self.use_brainbert:
            # BrainBERT outputs 768-dimensional embeddings per channel
            expected_input_channels = 768
            if self.foundation_model.input_channels != expected_input_channels:
                print(
                    f"Reinitializing input_projection for BrainBERT output: "
                    f"{self.foundation_model.input_channels} -> {expected_input_channels}"
                )
                model_dim = self.foundation_model.model_dim
                self.foundation_model.input_projection = nn.Linear(
                    expected_input_channels, model_dim
                )
                self.foundation_model.input_channels = expected_input_channels
        else:
            # Direct input processing: use actual channel count
            if (
                input_channels is not None
                and input_channels != self.foundation_model.input_channels
            ):
                print(
                    f"Reinitializing input_projection: {self.foundation_model.input_channels} -> {input_channels}"
                )
                model_dim = self.foundation_model.model_dim
                self.foundation_model.input_projection = nn.Linear(
                    input_channels, model_dim
                )
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
        Forward pass through STFT → BrainBERT (if enabled) → PopT foundation model.

        Matches original DIVER_CLIP PopT implementation:
        - Supports data_info_list (original approach)
        - Also supports lip_coords directly (backward compatibility)

        Args:
            x: Input tensor [batch_size, num_channels, seq_len] - raw neural signals
            data_info_list: List of dicts with 'LIP_id' key (original DIVER_CLIP approach)
            lip_coords: Optional LIP coordinates [batch_size, num_channels, 3] LongTensor (backward compatibility)
            **kwargs: Additional keyword arguments (e.g., preserve_ensemble for word embedding tasks)

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Extract LIP_id from data_info_list
        data_info_list = kwargs.get("data_info_list", None)
        lip_coords = kwargs.get("lip_coords", None)

        # Extract LIP_id from data_info_list (matching original PopT's datainfo_to_poptpostion_tensor() approach)
        if data_info_list is not None:
            # Reference original PopT.py's datainfo_to_poptpostion_tensor() approach
            coord_tensor = []
            for i in range(len(data_info_list)):
                coord_tensor.append(data_info_list[i]["LIP_id"])
            lip_coords = torch.stack(
                coord_tensor, dim=0
            )  # [batch_size, num_channels, 3]

        # Check if input is already STFT preprocessed (4D tensor: [batch, channels, time_stft, freq_channels])
        is_stft_preprocessed = len(x.shape) == 4

        # Process through BrainBERT if enabled
        if self.use_brainbert:
            if self.brainbert is None:
                raise ValueError("BrainBERT model not loaded but use_brainbert=True")

            if is_stft_preprocessed:
                # Input is already STFT preprocessed: [batch, channels, time_stft, freq_channels]
                batch_size, num_channels, time_stft, freq_channels = x.shape

                # Reshape: [batch, channels, time_stft, freq_channels] → [batch*channels, time_stft, freq_channels]
                x_stft = x.contiguous().view(
                    batch_size * num_channels, time_stft, freq_channels
                )

                # Convert to BrainBERT input shape: [batch*channels, time_stft, freq_channels] → [batch*channels, freq_channels, time_stft]
                # BrainBERT expects [batch, channels, time] where channels=freq_channels (40)
                x_brainbert = x_stft.transpose(
                    1, 2
                )  # [batch*channels, freq_channels, time_stft]

                with torch.no_grad():
                    # BrainBERT inference (frozen, no gradients) - all channels at once
                    # Return sequence to keep time dimension for middle selection (matching original PopT)
                    all_emb_seq = self.brainbert(
                        x_brainbert, return_sequence=True
                    )  # [batch*channels, time_stft, 768]

                    # Time dimension processing: select middle portion and pool (matching original PopT)
                    time_steps_stft = all_emb_seq.shape[1]
                    middle = int(time_steps_stft / 2)
                    # Select middle 10 time steps: [batch*channels, 10, 768]
                    all_emb_selected = all_emb_seq[:, middle - 5 : middle + 5, :]

                    # Mean pooling over time dimension: [batch*channels, 10, 768] → [batch*channels, 768]
                    all_emb = all_emb_selected.mean(dim=1)  # [batch*channels, 768]

                # Reshape back to [batch, channels, 768]
                x_emb = all_emb.view(batch_size, num_channels, 768)
            else:
                # Input is raw signal: [batch, channels, time]
                # STFT preprocessing should have been done in data loading, but fallback to model-level STFT
                if self.stft_preprocessor is None:
                    raise ValueError(
                        "Raw signal input detected but STFT preprocessor not initialized. "
                        "Either enable stft_preprocessing or initialize STFT preprocessor."
                    )

                batch_size, num_channels, time_steps = x.shape

                # Process all channels at once (matching original PopT implementation)
                x_reshaped = x.contiguous().view(
                    batch_size * num_channels, time_steps
                )  # [batch*channels, time]

                with torch.no_grad():
                    # STFT preprocessing: [batch*channels, time] → [batch*channels, time_stft, 40]
                    all_spec = self.stft_preprocessor(
                        x_reshaped
                    )  # [batch*channels, time_stft, 40]

                    # Move STFT output to same device as input (STFT returns CPU tensor)
                    all_spec = all_spec.to(x.device)  # [batch*channels, time_stft, 40]

                    # Convert to BrainBERT input shape: [batch*channels, time_stft, 40] → [batch*channels, 40, time_stft]
                    all_spec_brainbert = all_spec.transpose(
                        1, 2
                    )  # [batch*channels, 40, time_stft]

                    # BrainBERT inference (frozen, no gradients) - all channels at once
                    all_emb_seq = self.brainbert(
                        all_spec_brainbert, return_sequence=True
                    )  # [batch*channels, time_stft, 768]

                    # Time dimension processing: select middle portion and pool (matching original PopT)
                    time_steps_stft = all_emb_seq.shape[1]
                    middle = int(time_steps_stft / 2)
                    all_emb_selected = all_emb_seq[:, middle - 5 : middle + 5, :]
                    all_emb = all_emb_selected.mean(dim=1)  # [batch*channels, 768]

                # Reshape back to [batch, channels, 768]
                x_emb = all_emb.view(batch_size, num_channels, 768)

            # Pass BrainBERT embeddings to PopT foundation model
            if self.use_lip_coords:
                if lip_coords is None:
                    raise ValueError("lip_coords is required when use_lip_coords=True")
                features = self.foundation_model(
                    x_emb, return_sequence=False, lip_coords=lip_coords
                )
            else:
                features = self.foundation_model(x_emb, return_sequence=False)
        else:
            # Direct processing: pass raw signals to PopT (original behavior)
            if self.use_lip_coords:
                if lip_coords is None:
                    raise ValueError("lip_coords is required when use_lip_coords=True")
                features = self.foundation_model(
                    x, return_sequence=False, lip_coords=lip_coords
                )
            else:
                features = self.foundation_model(x, return_sequence=False)

        # Pass through decoder head
        output = self.decoder_head(features)

        return output


@registry.register_model_data_getter("popt_lip_coords")
def get_popt_lip_coords(task_df, raws, model_params):
    """
    Add lip_coords column with LIP coordinates for PopT model.

    The lip_coords tensor contains electrode coordinates for the PopT positional encoding.
    Each sample gets a copy of the same [num_channels, 3] LongTensor since all samples
    share the same electrode configuration.

    Note: This getter is optional for PopT. Only use it when use_lip_coords=True.
    Specify model_data_getter: popt_lip_coords in config to enable.

    Args:
        task_df: DataFrame containing task-specific data
        raws: List of MNE Raw objects
        model_params: Dictionary of model parameters from ModelSpec

    Returns:
        Tuple of (enriched_df, list of added column names)
    """
    from utils.data_utils import extract_subject_id_from_raw, get_lip_coordinates

    data_root = model_params.get("data_root", "data")
    coord_blocks = []
    total_channels = 0
    for raw in raws:
        subject_id = extract_subject_id_from_raw(raw)
        channel_names = raw.ch_names
        total_channels += len(channel_names)

        lip_df = get_lip_coordinates(subject_id, data_root)
        print(f"PopT: Loaded LIP coordinates for subject {subject_id}")

        channel_lip_map = {
            row["name"]: [int(row["x"]), int(row["y"]), int(row["z"])]
            for _, row in lip_df.iterrows()
        }
        block = torch.LongTensor(
            [channel_lip_map.get(ch, [0, 0, 0]) for ch in channel_names]
        )
        coord_blocks.append(block)

    if not coord_blocks:
        raise ValueError("PopT: No raws available to construct lip_coords")

    lip_coords = torch.cat(coord_blocks, dim=0)
    if lip_coords.shape[0] != total_channels:
        raise ValueError(
            f"PopT: Expected {total_channels} LIP coordinates, got {lip_coords.shape[0]}"
        )

    # Repeat for all samples - will be stacked during batching
    num_samples = len(task_df)
    task_df = task_df.copy()  # Avoid modifying original
    task_df["lip_coords"] = [lip_coords.clone() for _ in range(num_samples)]

    return task_df, ["lip_coords"]


@registry.register_model_constructor("popt_finetune")
@registry.register_model_constructor("population_transformer_finetune")
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
        - use_lip_coords: Whether to use LIP coordinates for positional encoding (optional, default: False)
        - max_coord_value: Maximum coordinate value for PE table (optional, default: 5000)
        - use_brainbert: Whether to use BrainBERT for time sequence processing (optional, default: True)
        - brainbert_model_dir: Path to BrainBERT pretrained model directory (optional)
    """
    output_dim = model_params.get("output_dim", 1)
    frozen_upstream = model_params.get("frozen_upstream", False) or model_params.get(
        "freeze_foundation", False
    )
    mlp_layer_sizes = model_params.get("mlp_layer_sizes", [])
    dropout = model_params.get("dropout", 0.0)
    use_brainbert = model_params.get("use_brainbert", True)
    use_lip_coords = model_params.get("use_lip_coords", False)
    brainbert_electrode_sequence = model_params.get(
        "brainbert_electrode_sequence", True
    )
    brainbert_foundation_dir = model_params.get("brainbert_foundation_dir") or model_params.get(
        "brainbert_model_dir"
    ) or "models/brainbert/pretrained_model"
    brainbert_upstream = None
    if use_brainbert:
        brainbert_upstream = _load_brainbert_upstream(brainbert_foundation_dir)

    ckpt_path, config_dir = _resolve_checkpoint_and_config_dir(model_params)
    random_init = ckpt_path is None
    _setup_popt_path()

    if random_init:
        if not any(
            model_params.get(k) is not None
            for k in ("popt_model_dim", "model_dim", "popt_num_layers", "num_layers")
        ):
            raise ValueError(
                "POPT random init requires popt_model_dim/model_dim and popt_num_layers/num_layers in model_params."
            )
        upstream_cfg = _model_params_to_upstream_cfg(model_params)
        ckpt = None
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        upstream_cfg = _get_upstream_cfg_from_checkpoint(ckpt)
        if upstream_cfg is None:
            if any(
                model_params.get(k) is not None
                for k in ("popt_model_dim", "model_dim", "popt_num_layers", "num_layers")
            ):
                upstream_cfg = _model_params_to_upstream_cfg(model_params)
            elif config_dir:
                upstream_cfg = _config_dict_to_upstream_cfg(_load_config_yaml(config_dir))
            else:
                raise ValueError(
                    "POPT checkpoint has no model_cfg and no config.yaml/model_params were provided."
                )

    upstream_cfg = _normalize_upstream_cfg(
        upstream_cfg,
        model_params=model_params,
        config_dir=config_dir,
        use_brainbert=use_brainbert,
        use_lip_coords=use_lip_coords,
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
            drop_position_encoding = not use_lip_coords
            states = _prepare_upstream_state_dict(
                _extract_state_dict(ckpt),
                drop_position_encoding=drop_position_encoding,
            )
            try:
                upstream.load_state_dict(
                    states, strict=not drop_position_encoding
                )
            except Exception:
                remapped = _prepare_upstream_state_dict(
                    _remap_state_dict_to_reference(states),
                    drop_position_encoding=drop_position_encoding,
                )
                upstream.load_state_dict(remapped, strict=False)

        if frozen_upstream:
            for p in upstream.parameters():
                p.requires_grad = False

        decoder = ReferencePOPTDecoder(
            upstream=upstream,
            output_dim=output_dim,
            hidden_dim=getattr(upstream_cfg, "hidden_dim", 768),
            mlp_layer_sizes=mlp_layer_sizes,
            dropout=dropout,
            input_dim=getattr(upstream_cfg, "input_dim", 40),
            brainbert_upstream=brainbert_upstream,
            use_lip_coords=use_lip_coords,
            brainbert_electrode_sequence=brainbert_electrode_sequence,
            output_activation=_resolve_output_activation(model_params),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder.to(device)
        return decoder
    finally:
        for name, mod in original_modules.items():
            sys.modules[name] = mod


# =============================================================================
# CONFIG SETTERS
# =============================================================================


@registry.register_config_setter("population_transformer_feature_extraction")
def set_feature_extraction_config(experiment_config, raws, _df_word):
    """
    Config setter for feature extraction pattern.

    Sets the input_dim for the MLP based on the foundation model's dimension.
    """
    from .config import load_config

    model_params = experiment_config.model_spec.params
    model_dir = model_params["model_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    foundation_config = load_config(config_path)

    # Set input_dim to match foundation model output
    model_params["input_dim"] = foundation_config.model_dim

    # Set preprocessor params
    data_params = experiment_config.task_config.data_params
    if not data_params.preprocessor_params:
        data_params.preprocessor_params = {}
    data_params.preprocessor_params["model_dir"] = model_dir

    return experiment_config


def _find_first_model_spec_by_constructors(model_spec, constructor_names):
    if model_spec.constructor_name in constructor_names:
        return model_spec
    for sub_model_spec in model_spec.sub_models.values():
        found = _find_first_model_spec_by_constructors(
            sub_model_spec, constructor_names
        )
        if found is not None:
            return found
    return None


@registry.register_config_setter("popt_finetune")
@registry.register_config_setter("population_transformer_finetune")
def set_finetuning_config(experiment_config, raws, _df_word):
    """
    Config setter for finetuning pattern.

    Sets the output_dim and loads foundation model config.
    Automatically sets use_brainbert based on use_lip_coords:
    - use_lip_coords=True -> use_brainbert=True (original PopT with LIP)
    - use_lip_coords=False -> use_brainbert=False (direct input processing)
    """
    from models.shared_config_setters import set_input_channels

    experiment_config = set_input_channels(
        experiment_config,
        raws,
        _df_word,
        ["popt_finetune", "population_transformer_finetune"],
    )

    target_spec = _find_first_model_spec_by_constructors(
        experiment_config.model_spec,
        {"popt_finetune", "population_transformer_finetune"},
    )
    if target_spec is None:
        raise ValueError("Could not find PopT model spec.")

    model_params = target_spec.params
    data_params = experiment_config.task_config.data_params
    task_name = experiment_config.task_config.task_name
    task_specific_config = experiment_config.task_config.task_specific_config

    original_channels = model_params.get("input_channels")
    if original_channels is not None:
        model_params["num_electrodes"] = original_channels

    use_lip_coords = model_params.get("use_lip_coords", False) or getattr(
        data_params, "use_lip_coords", False
    )
    model_params["use_lip_coords"] = use_lip_coords
    model_params["use_brainbert"] = model_params.get("use_brainbert", True)

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
    stft_config.setdefault("batch_size", experiment_config.training_params.batch_size or 4)

    _append_preprocessor(data_params, "stft_preprocessing", stft_config)
    data_params.use_stft_preprocessing = True
    data_params.stft_config = stft_config
    data_params.use_lip_coords = use_lip_coords

    if experiment_config.model_spec.constructor_name == "gpt2_brain" and use_lip_coords:
        experiment_config.model_spec.model_data_getter = "popt_lip_coords"
        experiment_config.model_spec.params["data_root"] = data_params.data_root
    elif use_lip_coords and not experiment_config.model_spec.model_data_getter:
        experiment_config.model_spec.model_data_getter = "popt_lip_coords"

    foundation_dir = (
        model_params.get("popt_foundation_dir")
        or model_params.get("foundation_dir")
        or model_params.get("checkpoint_path")
    )
    model_dir = model_params.get("popt_model_dir") or model_params.get("model_dir")
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
    model_params["output_activation"] = _resolve_output_activation(model_params)
    model_params["input_channels"] = stft_config.get("freq_channel_cutoff", 40)
    model_params["sample_rate"] = int(sample_rate)
    if use_lip_coords:
        model_params.setdefault("popt_position_encoding", "multi_subj_position_encoding")
    if model_params.get("output_dim") is not None:
        model_params["embedding_dim"] = model_params["output_dim"]

    return experiment_config
