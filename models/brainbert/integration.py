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


def _adaptive_avg_pool_temporal_patches(features, temporal_patches_to_keep):
    if temporal_patches_to_keep < 1:
        raise ValueError(
            "BrainBERT temporal_patches_to_keep must be at least 1: "
            f"got {temporal_patches_to_keep}."
        )

    features = features.transpose(1, 2).contiguous()
    features = F.adaptive_avg_pool1d(features, temporal_patches_to_keep)
    return features.transpose(1, 2)


class ReferenceBrainBERTDecoder(nn.Module):
    def __init__(
        self,
        finetune_model,
        output_dim=1,
        num_electrodes=None,
        hidden_dim=768,
        temporal_patches_to_keep=10,
        mlp_layer_sizes=None,
        dropout=0.0,
        output_activation="linear",
    ):
        super().__init__()
        self.finetune_model = finetune_model
        self.output_dim = output_dim
        self.num_electrodes = num_electrodes
        self.hidden_dim = hidden_dim
        self.temporal_patches_to_keep = temporal_patches_to_keep
        self.output_activation = output_activation

        if self.num_electrodes is not None and self.hidden_dim is not None:
            input_dim = (
                self.num_electrodes
                * self.temporal_patches_to_keep
                * self.hidden_dim
            )
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
                self.projector = nn.Sequential(*layers)
            else:
                self.projector = nn.Linear(input_dim, output_dim)
                nn.init.normal_(self.projector.weight, mean=0.0, std=0.001)
                nn.init.zeros_(self.projector.bias)
        else:
            self.projector = None

    def encode_features(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(
                "BrainBERT finetuning expects STFT input with shape [batch, channels, time, freq]."
            )

        batch_size, num_channels, time_steps, freq_channels = x.shape
        inputs = x.contiguous().view(batch_size * num_channels, time_steps, freq_channels)
        pad_mask = None

        if self.projector is not None:
            if self.finetune_model.frozen_upstream:
                self.finetune_model.upstream.eval()
                with torch.no_grad():
                    features = self.finetune_model.upstream(
                        inputs, pad_mask, intermediate_rep=True
                    )
            else:
                features = self.finetune_model.upstream(
                    inputs, pad_mask, intermediate_rep=True
                )

            if features.shape[0] == batch_size * num_channels:
                features = _adaptive_avg_pool_temporal_patches(
                    features, self.temporal_patches_to_keep
                )
            else:
                raise ValueError(
                    "BrainBERT upstream returned an unexpected batch dimension: "
                    f"got {features.shape[0]}, expected {batch_size * num_channels}."
                )

            features = features.view(
                batch_size,
                num_channels,
                self.temporal_patches_to_keep,
                -1,
            )
            return features.reshape(batch_size, -1)

        out = self.finetune_model(inputs, pad_mask)
        return out.view(batch_size, num_channels, -1).mean(dim=1)

    def forward_from_features(self, features, **kwargs):
        out = self.projector(features) if self.projector is not None else features

        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.output_activation == "softmax":
            out = F.softmax(out, dim=-1)

        if self.output_dim == 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def forward(self, x, **kwargs):
        features = self.encode_features(x, **kwargs)
        if kwargs.get('return_feature_emb_instead_of_projection', False):
            return features
        return self.forward_from_features(features, **kwargs)


# =============================================================================
# PATTERN 2: FINETUNING (TRAINABLE MODEL)
# =============================================================================

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
    feature_cache = model_params.get("feature_cache", False)
    output_dim = model_params.get("output_dim")
    if output_dim is None:
        output_dim = 1
    frozen_upstream = model_params.get("frozen_upstream", False) or model_params.get(
        "freeze_foundation", False
    )
    mlp_layer_sizes = model_params.get("mlp_layer_sizes", [])
    dropout = model_params.get("dropout", 0.0)
    num_electrodes = model_params.get("num_electrodes")
    temporal_patches_to_keep = model_params.get("temporal_patches_to_keep", 10)

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

        finetune_cfg = _dict_to_cfg(
            {
                "name": "finetune_model",
                "frozen_upstream": frozen_upstream,
                "hidden_dim": getattr(upstream_cfg, "hidden_dim", 768),
            }
        )
        finetune_model = build_model(finetune_cfg, upstream)

        hidden_dim = getattr(upstream_cfg, "hidden_dim", 768)
        if not feature_cache and not num_electrodes:
            if mlp_layer_sizes:
                finetune_model.linear_out = MLPDecoder(
                    input_dim=hidden_dim,
                    layer_sizes=mlp_layer_sizes + [output_dim],
                    dropout=dropout,
                    use_layer_norm=True,
                    output_activation="linear",
                )
            else:
                finetune_model.linear_out = nn.Linear(hidden_dim, output_dim)

        decoder = ReferenceBrainBERTDecoder(
            finetune_model,
            output_dim=output_dim,
            num_electrodes=num_electrodes,
            hidden_dim=hidden_dim,
            temporal_patches_to_keep=temporal_patches_to_keep,
            mlp_layer_sizes=mlp_layer_sizes,
            dropout=dropout,
            output_activation=_resolve_output_activation(model_params, output_dim),
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
    feature_cache = target_spec.feature_cache or model_params.get("feature_cache", False)
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
    stft_config = configure_explicit_stft_preprocessor(
        data_params,
        sample_rate=int(sample_rate),
        model_name="BrainBERT finetuning",
    )

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
        model_params["output_activation"] = _resolve_output_activation(
            model_params, model_params.get("output_dim", 1)
        )

    model_params["input_channels"] = stft_config.get("freq_channel_cutoff", 40)
    model_params["sample_rate"] = int(sample_rate)
    model_params.setdefault("temporal_patches_to_keep", 10)
    if not feature_cache and model_params.get("output_dim") is not None:
        model_params["embedding_dim"] = model_params["output_dim"]

    return experiment_config
