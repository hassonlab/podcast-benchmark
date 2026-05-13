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
import torch
import torch.nn as nn
from torch.nn import functional as F

from core import registry
from models.shared_decoders import MLPProbeDecoder as MLPDecoder
from models.stft_config import configure_explicit_stft_preprocessor

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
        "input_dim": (
            BRAINBERT_OUTPUT_DIM
            if use_brainbert
            else _cfg_lookup(upstream_cfg, "input_dim", "input_channels")
            or config_dict.get("input_channels")
            or model_params.get("input_channels")
            or 40
        ),
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
        if (
            name in ("models", "utils")
            or name.startswith("models.")
            or name.startswith("utils.")
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
            upstream.load_state_dict(
                _remap_state_dict_to_reference(states), strict=False
            )
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
        num_electrodes=None,
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
        self.num_electrodes = num_electrodes
        self.flatten_electrode_sequence = (
            brainbert_upstream is not None
            and brainbert_electrode_sequence
            and num_electrodes is not None
        )
        self.cls_dim = (
            BRAINBERT_OUTPUT_DIM if brainbert_upstream is not None else input_dim
        )
        self.classifier_norm = nn.LayerNorm(hidden_dim)
        self.head = MLPDecoder(
            input_dim=(
                (num_electrodes + 1) * hidden_dim
                if self.flatten_electrode_sequence
                else hidden_dim
            ),
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
        seq_ids = torch.zeros(batch_size, num_channels, dtype=torch.long, device=device)
        return lip_coords, seq_ids

    def encode_features(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(
                "PopT finetuning expects STFT input with shape [batch, channels, time, freq]."
            )

        lip_coords = kwargs.get("lip_coords")
        batch_size, num_channels, time_steps, freq_channels = x.shape
        inputs = x.contiguous().view(
            batch_size * num_channels, time_steps, freq_channels
        )
        pad_mask = None

        if self.brainbert_upstream is not None:
            self.brainbert_upstream.eval()
            with torch.no_grad():
                features = self.brainbert_upstream(
                    inputs, pad_mask, intermediate_rep=True
                )

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
                if self.flatten_electrode_sequence:
                    features = self.classifier_norm(encoded)
                    return features.reshape(batch_size, -1)
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
                cls_repr = (
                    encoded[:, 0, :].view(batch_size, num_channels, -1).mean(dim=1)
                )
        else:
            if self.use_lip_coords:
                raise ValueError(
                    "use_lip_coords requires use_brainbert=True in this port."
                )
            cls = self._make_cls_token(
                batch_size * num_channels, inputs.device, inputs.dtype
            )
            seq = torch.cat([cls, inputs], dim=1)
            encoded = self.upstream(seq, pad_mask, intermediate_rep=True)
            cls_repr = encoded[:, 0, :].view(batch_size, num_channels, -1).mean(dim=1)

        return self.classifier_norm(cls_repr)

    def forward_from_features(self, features, **kwargs):
        return self.head(features)

    def forward(self, x, **kwargs):
        features = self.encode_features(x, **kwargs)
        if kwargs.get("return_feature_emb_instead_of_projection", False):
            return features
        return self.forward_from_features(features, **kwargs)


# =============================================================================
# PATTERN 2: FINETUNING (TRAINABLE MODEL)
# =============================================================================


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
    feature_cache = model_params.get("feature_cache", False)
    output_dim = model_params.get("output_dim")
    if output_dim is None:
        output_dim = 1
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
    brainbert_foundation_dir = (
        model_params.get("brainbert_foundation_dir")
        or model_params.get("brainbert_model_dir")
        or "models/brainbert/pretrained_model"
    )
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
                for k in (
                    "popt_model_dim",
                    "model_dim",
                    "popt_num_layers",
                    "num_layers",
                )
            ):
                upstream_cfg = _model_params_to_upstream_cfg(model_params)
            elif config_dir:
                upstream_cfg = _config_dict_to_upstream_cfg(
                    _load_config_yaml(config_dir)
                )
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
        if (
            name in ("models", "utils")
            or name.startswith("models.")
            or name.startswith("utils.")
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
                upstream.load_state_dict(states, strict=not drop_position_encoding)
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
            num_electrodes=model_params.get("num_electrodes"),
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
    feature_cache = target_spec.feature_cache or model_params.get("feature_cache", False)
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
    stft_config = configure_explicit_stft_preprocessor(
        data_params,
        sample_rate=int(sample_rate),
        model_name="PopT finetuning",
        extra_defaults={"batch_size": experiment_config.training_params.batch_size or 4},
    )
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

    if not feature_cache and model_params.get("output_dim") is None:
        output_dim = _default_output_dim_for_task(task_name, task_specific_config)
        if output_dim is not None:
            model_params["output_dim"] = output_dim

    if not feature_cache:
        losses = experiment_config.training_params.losses or []
        if not losses and experiment_config.training_params.loss_name:
            losses = [experiment_config.training_params.loss_name]
        model_params["_training_losses"] = losses
        model_params["_loss_name"] = experiment_config.training_params.loss_name
        model_params["output_activation"] = _resolve_output_activation(model_params)
    model_params["input_channels"] = stft_config.get("freq_channel_cutoff", 40)
    model_params["sample_rate"] = int(sample_rate)
    if use_lip_coords:
        model_params.setdefault(
            "popt_position_encoding", "multi_subj_position_encoding"
        )
    if not feature_cache and model_params.get("output_dim") is not None:
        model_params["embedding_dim"] = model_params["output_dim"]

    return experiment_config
