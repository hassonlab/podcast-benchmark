"""
Integration code for DIVER-1 Foundation Model

This module integrates DIVER-1 model loading and finetuning into the Podcast benchmark framework.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import gc
from torch.nn import functional as F

from core import registry

# =============================================================================
# PYTHON PATH SETUP
# =============================================================================

# Add DIVER-1 module to Python path (lazy loading - only when needed)
# This allows imports like "from models.diver import DIVER" to work correctly
# Note: We don't add it at module import time to avoid importing DIVER-1's
# datasets module which requires lmdb and other dependencies
def _setup_diver1_path():
    """Setup DIVER-1 path when actually needed"""
    diver_dir = os.path.dirname(os.path.abspath(__file__))
    diver1_dir = os.path.join(diver_dir, "DIVER-1")
    if diver1_dir not in sys.path:
        sys.path.insert(0, diver1_dir)
    return diver1_dir

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_podcast_task_info(
    task_name: str, 
    subject_id: int, 
    patch_sampling_rate: int = 500,
    num_channels: int = None,
    num_targets: int = None,
    window_width: float = None
):
    """
    Get task info dict for Podcast task using podcast-benchmark's information.
    
    Args:
        task_name: Task name (e.g., "word_embedding", "volume_level")
        subject_id: Subject ID (for logging)
        patch_sampling_rate: Patch sampling rate in Hz (default: 500)
        num_channels: Number of channels (from actual data via set_input_channels)
        num_targets: Number of output targets (from config output_dim)
        window_width: Window width in seconds (from data_params.window_width)
    
    Returns:
        task_info_dict: Dictionary with task information for DIVER model
    """
    # Determine num_targets if not provided
    # NOTE: Binary classification tasks use output_dim=1 (single logit)
    if num_targets is None:
        num_targets_map = {
            "word_embedding": 50,  # Default PCA dim
            "word_embedding_decoding_task": 50,
            "whisper_embedding_decoding_task": 50,
            "whisper_embedding": 50,
            "gpt_surprise": 1,
            "gpt_surprise_task": 1,
            "sentence_onset": 1,  # Binary classification: output_dim=1 for BCE loss compatibility
            "sentence_onset_task": 1,
            "content_noncontent": 1,  # Binary classification: output_dim=1 for BCE loss compatibility
            "content_noncontent_task": 1,
            "gpt_surprise_multiclass": 3,
            "gpt_surprise_multiclass_task": 3,
            "pos": 5,
            "pos_task": 5,
            "volume_level": 1,
            "volume_level_decoding_task": 1,
        }
        num_targets = num_targets_map.get(task_name, 50)
    
    # Build task_info_dict using podcast-benchmark's values
    task_info = {
        'target_dynamics': 'discrete',  # Podcast tasks are always discrete
        'consistent_channels': True,  # Podcast uses consistent channels per subject
        'num_channels': num_channels,  # From actual data (set_input_channels)
        'num_seconds': window_width or 0.5,  # From data_params.window_width
        'num_targets': num_targets,  # From config output_dim
        'patch_sampling_rate': patch_sampling_rate,
    }
    
    return task_info


def create_diver1_params(model_params: dict):
    """
    Convert Podcast benchmark's model_params dict to DIVER-1's params object.
    
    Args:
        model_params: Podcast benchmark's model_params dictionary
        
    Returns:
        params: argparse.Namespace-like object (expected by DIVER-1)
    """
    class Params:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    params_dict = {
        'foundation_dir': model_params.get("foundation_dir", None),
        'patch_size': model_params.get("patch_size", 500),
        'width': model_params.get("width") or model_params.get("d_model", 512),
        'depth': model_params.get("depth") or model_params.get("e_layer", 12),
        'mup_weights': model_params.get("mup_weights", False),
        'ft_mup': model_params.get("ft_mup", False),
        'ft_config': model_params.get("ft_config", "flatten_linear"),
        'deepspeed_pth_format': model_params.get("deepspeed_pth_format", True),
        'model_dir': model_params.get("model_dir")
        or (
            os.path.dirname(model_params["foundation_dir"])
            if model_params.get("foundation_dir")
            else None
        ),
        'num_mlp_layers': model_params.get("num_mlp_layers", 2),
        'load_adapter_weights': model_params.get("load_adapter_weights", False),
        'adapter_path': model_params.get("adapter_path", None),
    }

    for key, value in model_params.items():
        if key not in params_dict:
            params_dict[key] = value

    params = Params(params_dict)
    
    return params


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


def _candidate_diver_strategy_dirs(model_params):
    strategy_base_dir = model_params.get("strategy_base_dir")
    if strategy_base_dir:
        return [strategy_base_dir]

    local_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_model")
    reference_base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)).replace(
            "/podcast-benchmark/models/diver", "/references_podcast/podcast-benchmark/models/diverclip"
        ),
        "pretrained_model",
    )
    reference_flatten_diveronly = os.path.join(
        os.path.dirname(os.path.abspath(__file__)).replace(
            "/podcast-benchmark/models/diver", "/references_podcast/podcast-benchmark/models/diverclip_flatten"
        ),
        "pretrained_model",
        "diveronly",
    )
    reference_attn_diveronly = os.path.join(
        os.path.dirname(os.path.abspath(__file__)).replace(
            "/podcast-benchmark/models/diver", "/references_podcast/podcast-benchmark/models/diverclip_attn"
        ),
        "pretrained_model",
        "diveronly",
    )
    candidates = [local_base]
    if os.path.isdir(reference_base):
        candidates.append(reference_base)
    if os.path.isdir(reference_flatten_diveronly):
        candidates.append(reference_flatten_diveronly)
    if os.path.isdir(reference_attn_diveronly):
        candidates.append(reference_attn_diveronly)
    return candidates


def _resolve_diver_foundation_dir(model_params):
    foundation_dir = model_params.get("foundation_dir")
    if foundation_dir:
        return foundation_dir

    strategy = model_params.get("pretrained_strategy") or model_params.get("strategy")
    if not strategy:
        return foundation_dir

    file_map = {
        "divertri": "diver_triplet_dev.pth",
        "diverbiaudio": "diver_bi_audio_dev.pth",
        "diverbitext": "diver_bi_text_dev.pth",
        "randomtri": "random_triplet_dev.pth",
        "randombiaudio": "random_bi_audio_dev.pth",
        "randombitext": "random_bi_text_dev.pth",
        "diveronly": "diveronly_converted.pth",
    }
    subject_id = model_params.get("subject_id")
    config_signature = model_params.get("config_signature")

    for base_dir in _candidate_diver_strategy_dirs(model_params):
        if strategy in ("default", "legacy"):
            for candidate in (
                os.path.join(base_dir, "legacy", "256_mp_rank_00_model_states.pt"),
                os.path.join(base_dir, "256_mp_rank_00_model_states.pt"),
            ):
                if os.path.exists(candidate):
                    return candidate
            continue

        strategy_dir = os.path.join(base_dir, strategy)
        if not os.path.isdir(strategy_dir):
            continue
        if subject_id is not None and config_signature is not None:
            subject_path = os.path.join(
                strategy_dir,
                f"sub-{int(subject_id):02d}",
                config_signature,
                "backbone.pt",
            )
            if os.path.exists(subject_path):
                return subject_path
        filename = file_map.get(strategy)
        if filename:
            candidate = os.path.join(strategy_dir, filename)
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(f"Could not resolve DIVER pretrained checkpoint for strategy='{strategy}'")


def _resolve_diver_adapter_path(model_params):
    adapter_path = model_params.get("adapter_path")
    if adapter_path or not model_params.get("load_adapter_weights", False):
        return adapter_path

    strategy = model_params.get("pretrained_strategy") or model_params.get("strategy")
    if not strategy:
        return None

    for base_dir in _candidate_diver_strategy_dirs(model_params):
        candidate = os.path.join(base_dir, strategy, "diver_adapter_best.pt")
        if os.path.exists(candidate):
            return candidate
    return None


def create_data_info_list(batch_size: int, num_channels: int, channel_names: list = None, xyz_id: np.ndarray = None):
    """
    Create data_info_list for DIVER model forward pass.
    
    Args:
        batch_size: Batch size
        num_channels: Number of channels
        channel_names: List of channel names (optional)
        xyz_id: MNI xyz coordinates [num_channels, 3] (optional)
                If provided, will be included in each sample's data_info_dict.
                Podcast Benchmark uses consistent channel order, so the same xyz_id
                can be used for all samples in a batch.
    
    Returns:
        data_info_list: List of dicts, one per sample
                        Each dict contains 'num_channels', 'modality', and optionally 'channel_names' and 'xyz_id'
    
    Note:
        This matches the original DIVER-1 implementation where each sample has a data_info_dict
        with 'xyz_id' for PositionalEncoding3D and 'modality' for ChannelTypeEmbedding.
        In Podcast Benchmark, all samples use the same channel configuration, so we use the same
        xyz_id and modality for all samples.
    """
    data_info_list = []
    for _ in range(batch_size):
        sample_info = {
            'num_channels': num_channels,
            'modality': 'iEEG',  # Required by ChannelTypeEmbedding (available_channel_types = ['EEG','iEEG'])
        }
        if channel_names is not None:
            sample_info['channel_names'] = channel_names
        if xyz_id is not None:
            # Include xyz_id for PositionalEncoding3D (required by DIVER-1)
            sample_info['xyz_id'] = xyz_id  # [num_channels, 3]
        data_info_list.append(sample_info)
    return data_info_list


# =============================================================================
# DIVER DECODER CLASS
# =============================================================================

class DIVEREncoder(nn.Module):
    def __init__(self, diver_model):
        super().__init__()
        self.diver_model = diver_model

    def forward(self, x, **kwargs):
        xyz_id = kwargs.get("xyz_id", None)
        data_info_list = kwargs.get("data_info_list", None)

        batch_size = x.shape[0]
        num_channels = x.shape[1]

        if xyz_id is not None:
            data_info_list = []
            for i in range(batch_size):
                coords = xyz_id[i]
                if torch.is_tensor(coords):
                    coords = coords.cpu().numpy()
                data_info_list.append(
                    {
                        "num_channels": num_channels,
                        "modality": "iEEG",
                        "xyz_id": coords,
                    }
                )
        elif data_info_list is None:
            data_info_list = create_data_info_list(batch_size, num_channels)

        backbone_out = self.diver_model.backbone(
            x, data_info_list=data_info_list, use_mask=False, return_encoder_output=True
        )
        features = self.diver_model.feature_extraction_func(
            backbone_out, data_info_list=data_info_list
        )
        return self.diver_model.ft_model_input_adapter(
            features, data_info_list=data_info_list
        )


class DIVERHead(nn.Module):
    def __init__(self, ft_core_model, ft_model_output_adapter):
        super().__init__()
        self.ft_core_model = ft_core_model
        self.ft_model_output_adapter = ft_model_output_adapter

    def forward(self, x, **kwargs):
        data_info_list = kwargs.get("data_info_list", None)
        out = self.ft_core_model(x, data_info_list=data_info_list)
        return self.ft_model_output_adapter(out, data_info_list=data_info_list)


class DIVERDecoder(nn.Module):
    """
    DIVER model decoder for Podcast benchmark.
    
    Wraps DIVER-1's FineTuneModel to match Podcast benchmark interface.
    Handles data format conversion and output activation.
    """
    
    def __init__(
        self,
        diver_model,
        output_activation: str = "linear",
        output_dim: int = None,
        encoder_model=None,
        head_model=None,
    ):
        """
        Args:
            diver_model: DIVER FineTuneModel instance (flatten_linear_finetune or flatten_mlp_finetune)
            output_activation: Activation function for output ("linear", "sigmoid", "softmax")
                - "sigmoid": For binary classification (output_dim=1)
                - "softmax": For multiclass classification (output_dim>1)
                - "linear": No activation (for regression)
            output_dim: Output dimension of the model
        """
        super().__init__()
        self.diver_model = diver_model
        if encoder_model is not None:
            self.encoder_model = encoder_model
            self.head_model = head_model
        elif hasattr(diver_model, "ft_core_model") and hasattr(
            diver_model, "ft_model_output_adapter"
        ):
            self.encoder_model = DIVEREncoder(diver_model)
            self.head_model = DIVERHead(
                diver_model.ft_core_model, diver_model.ft_model_output_adapter
            )
        else:
            self.encoder_model = None
            self.head_model = None
        self.output_activation = output_activation
        self.output_dim = output_dim
    
    def forward(self, x, **kwargs):
        """
        Forward pass through DIVER model.

        Args:
            x: Input tensor [batch_size, num_channels, seq_len]
            **kwargs: Additional keyword arguments
                - xyz_id: MNI coordinates [batch_size, num_channels, 3] (preferred)
                - data_info_list: List of dicts with metadata for each sample (legacy)

        Returns:
            Output tensor [batch_size, output_dim] (probabilities if activation applied)
        """
        return_feature_emb = kwargs.pop("return_feature_emb_instead_of_projection", False)
        if self.encoder_model is not None and self.head_model is not None:
            encoder_out = self.encoder_model(x, **kwargs)
            if return_feature_emb:
                return encoder_out
            output = self.head_model(encoder_out, **kwargs)
        else:
            xyz_id = kwargs.get("xyz_id", None)
            data_info_list = kwargs.get("data_info_list", None)

            batch_size = x.shape[0]
            num_channels = x.shape[1]

            if xyz_id is not None:
                data_info_list = []
                for i in range(batch_size):
                    coords = xyz_id[i]
                    if torch.is_tensor(coords):
                        coords = coords.cpu().numpy()
                    data_info_list.append(
                        {
                            "num_channels": num_channels,
                            "modality": "iEEG",
                            "xyz_id": coords,
                        }
                    )
            elif data_info_list is None:
                data_info_list = create_data_info_list(batch_size, num_channels)

            output = self.diver_model(x, data_info_list=data_info_list)
            if return_feature_emb:
                return output

        # Apply output activation to convert logits to probabilities
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        elif self.output_activation == "softmax":
            output = F.softmax(output, dim=-1)
        # "linear" means no activation (default for regression)
        
        # Squeeze if output_dim == 1 (for binary classification compatibility)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        
        return output


# =============================================================================
# MODEL CONSTRUCTOR
# =============================================================================

@registry.register_model_data_getter("diver_data_info")
def get_diver_data_info(task_df, raws, model_params):
    """
    Add xyz_id column to task_df for DIVER model.

    The xyz_id contains MNI coordinates for PositionalEncoding3D.
    Each sample gets a copy of the same [num_channels, 3] array since all samples
    share the same electrode configuration.

    Args:
        task_df: DataFrame containing task-specific data
        raws: List of MNE Raw objects
        model_params: Dictionary of model parameters from ModelSpec

    Returns:
        Tuple of (enriched_df, list of added column names)
    """
    from utils.data_utils import extract_subject_id_from_raw, get_mni_coordinates

    data_root = model_params.get("data_root", "data")
    coord_blocks = []
    for raw in raws:
        subject_id = extract_subject_id_from_raw(raw)
        channel_names = raw.ch_names
        coords = get_mni_coordinates(subject_id, channel_names, data_root)
        coord_blocks.append(coords)

    if not coord_blocks:
        raise ValueError("DIVER: No raws available to construct xyz_id")

    mni_coords = np.concatenate(coord_blocks, axis=0)

    # Add xyz_id column - same coords for all samples
    num_samples = len(task_df)
    task_df = task_df.copy()  # Avoid modifying original
    task_df["xyz_id"] = [mni_coords.copy() for _ in range(num_samples)]

    return task_df, ["xyz_id"]


@registry.register_model_constructor("diver_finetune", required_data_getter="diver_data_info")
def create_diver_finetuning_model(model_params):
    """
    Create DIVER finetuning model using DIVER-1's finetune_model classes.
    
    Expected model_params:
        - foundation_dir: Path to pretrained weights file (required)
        - output_dim: Output dimension (required)
        - task_name: Podcast task name (required)
        - subject_id: Subject ID (required)
        - patch_sampling_rate: Patch sampling rate in Hz (default: 500)
        - patch_size: Patch size in samples (default: 500)
        - d_model: Model dimension (default: 512)
        - e_layer: Number of encoder layers (default: 12)
        - ft_config: Finetuning config ("flatten_linear" or "flatten_mlp", default: "flatten_linear")
        - deepspeed_pth_format: Whether weights are in DeepSpeed format (default: True)
        - freeze_foundation: Whether to freeze backbone (default: False)
        - mup_weights: Backbone weights were trained with MuP (default: False)
        - ft_mup: Use MuP for finetuning (default: False)
        - model_dir: Model directory for MuP base shapes (optional)
        - width: MuP width override (optional)
        - depth: MuP depth override (optional)
        - num_mlp_layers: Number of MLP layers for flatten_mlp (default: 2)
    """
    # Required parameters
    foundation_dir = _resolve_diver_foundation_dir(model_params)
    if foundation_dir is not None:
        model_params["foundation_dir"] = foundation_dir
    adapter_path = _resolve_diver_adapter_path(model_params)
    if adapter_path is not None:
        model_params["adapter_path"] = adapter_path
    output_dim = model_params["output_dim"]
    task_name = model_params.get("task_name", "word_embedding")
    subject_id = model_params.get("subject_id", 1)
    
    # Optional parameters with defaults
    patch_sampling_rate = model_params.get("patch_sampling_rate", 500)
    patch_size = model_params.get("patch_size", 500)
    d_model = model_params.get("d_model", 512)
    e_layer = model_params.get("e_layer", 12)
    ft_config = model_params.get("ft_config", "flatten_linear")
    deepspeed_pth_format = model_params.get("deepspeed_pth_format", True)
    freeze_foundation = model_params.get("freeze_foundation", False)
    mup_weights = model_params.get("mup_weights", False)
    ft_mup = model_params.get("ft_mup", False)
    model_dir = model_params.get("model_dir", None)
    width = model_params.get("width", d_model)
    depth = model_params.get("depth", e_layer)
    num_mlp_layers = model_params.get("num_mlp_layers", 2)
    
    # Get task_info_dict from model_params (set by config setter)
    task_info_dict = model_params.get("_task_info_dict")
    if task_info_dict is None:
        # Fallback: construct from available information
        num_channels = model_params.get("input_channels")
        window_width = model_params.get("window_width", 0.5)
        
        task_info_dict = get_podcast_task_info(
            task_name=task_name,
            subject_id=subject_id,
            patch_sampling_rate=patch_sampling_rate,
            num_channels=num_channels,
            num_targets=output_dim,
            window_width=window_width
        )
    
    # Create params object for DIVER-1
    params = create_diver1_params(model_params)
    
    # Setup DIVER-1 path and import model classes
    # Note: We need to handle the conflict between podcast-benchmark's 'models' and 'utils' packages
    # (already loaded in main.py) and DIVER-1's 'models' and 'utils' packages.
    # Solution: Temporarily remove 'models' and 'utils' (and all their submodules) from sys.modules,
    # import DIVER-1's modules, create the model, then restore them. This allows DIVER-1's internal imports
    # (models.diver, utils.mup_utils, etc.) to work correctly during model creation.
    diver1_dir = _setup_diver1_path()
    
    # Save the original 'models' and 'utils' modules and their submodules
    original_modules = {}
    modules_to_remove = []
    
    # Collect all modules starting with 'models.' or 'utils.'
    for module_name in list(sys.modules.keys()):
        if module_name == 'models' or module_name == 'utils' or \
           module_name.startswith('models.') or module_name.startswith('utils.'):
            original_modules[module_name] = sys.modules[module_name]
            modules_to_remove.append(module_name)
    
    # Remove all collected modules from sys.modules
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    try:
        # Import DIVER modules
        from models.finetune_model import flatten_linear_finetune, flatten_mlp_finetune
        from utils import mup_utils as diver1_mup_utils
        from mup import set_base_shapes as mup_set_base_shapes

        base_shapes_path = None
        if getattr(params, "ft_mup", False):
            # Define builder (same as existing logic)
            def _full_finetune_builder(w: int, d: int):
                class _P: pass
                p = _P()
                for k, v in params.__dict__.items():
                    setattr(p, k, v)
                p.width = w
                p.depth = d
                p.mup_weights = True
                p.ft_mup = True
                p.foundation_dir = None  # No checkpoint load

                if ft_config == "flatten_linear":
                    return flatten_linear_finetune(p, task_info_dict)
                else:
                    return flatten_mlp_finetune(p, task_info_dict)

            identifier = "DIVER_iEEG_FINAL_model_finetune"
            if getattr(params, "patch_size", None) == 50:
                identifier += "_patch50"

            save_dir = getattr(params, "model_dir", None) or os.path.dirname(foundation_dir)
            
            base_shapes_path = diver1_mup_utils.ensure_base_shapes(
                model_builder=_full_finetune_builder,
                identifier=identifier,
                width=getattr(params, "width", d_model),
                depth=getattr(params, "depth", e_layer),
                save_dir=save_dir,
            )
            del _full_finetune_builder
            gc.collect()
            torch.cuda.empty_cache()

        original_foundation_dir = getattr(params, "foundation_dir", None)
        params.foundation_dir = None

        if ft_config == "flatten_linear":
            diver_model = flatten_linear_finetune(params, task_info_dict)
        elif ft_config == "flatten_mlp":
            diver_model = flatten_mlp_finetune(params, task_info_dict)
        else:
            raise ValueError(f"Unknown ft_config: {ft_config}")

        if base_shapes_path is not None:
            mup_set_base_shapes(diver_model, base_shapes_path, rescale_params=False)

        if original_foundation_dir:
            params.foundation_dir = original_foundation_dir
            diver_model.load_backbone_checkpoint(
                original_foundation_dir,
                device="cpu",
                deepspeed_pth_format=params.deepspeed_pth_format,
            )
    
    
    finally:
        # Restore all original modules AFTER model creation is complete
        # This ensures podcast-benchmark's code continues to work correctly
        for module_name, module_obj in original_modules.items():
            sys.modules[module_name] = module_obj
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diver_model.to(device)
    
    # Freeze backbone if requested
    if freeze_foundation and hasattr(diver_model, 'backbone'):
        for p in diver_model.backbone.parameters():
            p.requires_grad = False
        print("DIVER backbone frozen")
    
    output_activation = _resolve_output_activation(model_params)
    
    # Wrap in DIVERDecoder for Podcast benchmark interface
    encoder_model = model_params.get("encoder_model")
    head_model = model_params.get("head_model")
    if encoder_model is not None and head_model is None and hasattr(
        diver_model, "ft_core_model"
    ) and hasattr(diver_model, "ft_model_output_adapter"):
        head_model = DIVERHead(
            diver_model.ft_core_model, diver_model.ft_model_output_adapter
        )

    decoder = DIVERDecoder(
        diver_model,
        output_activation=output_activation,
        output_dim=output_dim,
        encoder_model=encoder_model,
        head_model=head_model,
    )
    
    return decoder


@registry.register_model_constructor("diver_encoder", required_data_getter="diver_data_info")
def create_diver_encoder(model_params):
    decoder = create_diver_finetuning_model(dict(model_params))
    return decoder.encoder_model


# =============================================================================
# CONFIG SETTER
# =============================================================================

@registry.register_config_setter("diver_finetune")
def set_diver_finetuning_config(experiment_config, raws, _df_word):
    """
    Config setter for DIVER finetuning.
    """
    from models.shared_config_setters import set_input_channels

    # 1. Set input_channels using common utility (same as BrainBERT/PopT)
    experiment_config = set_input_channels(
        experiment_config, raws, _df_word, ["diver_finetune"]
    )
    experiment_config.task_config.data_params.use_stft_preprocessing = False

    diver_model_spec = _find_first_model_spec_by_constructor(
        experiment_config.model_spec, "diver_finetune"
    )
    if diver_model_spec is None:
        raise ValueError("Could not find diver_finetune model spec.")

    # 2. Get task information from podcast-benchmark
    model_params = diver_model_spec.params
    task_name = model_params.get("task_name") or experiment_config.task_config.task_name
    model_params["task_name"] = task_name
    subject_id = model_params.get("subject_id", 1)
    patch_sampling_rate = model_params.get("patch_sampling_rate", 500)
    num_channels = model_params.get("input_channels")
    task_specific_config = experiment_config.task_config.task_specific_config

    # 3. Get window_width from data_params
    data_params = experiment_config.task_config.data_params
    window_width = getattr(data_params, 'window_width', None)
    if window_width is None or window_width <= 0:
        window_width = 0.5
        print(f"DIVER: Using default window_width={window_width}")
    data_params.window_width = window_width

    if experiment_config.model_spec.constructor_name == "gpt2_brain":
        experiment_config.model_spec.model_data_getter = "diver_data_info"
        experiment_config.model_spec.params["data_root"] = data_params.data_root

    # 4. Determine output_dim (podcast-benchmark task mapping)
    if "output_dim" not in model_params or model_params["output_dim"] is None:
        num_targets = _default_output_dim_for_task(task_name, task_specific_config) or 50
        model_params["output_dim"] = num_targets
    else:
        num_targets = model_params["output_dim"]

    # 5. Create task_info_dict using podcast-benchmark information
    task_info_dict = get_podcast_task_info(
        task_name=task_name,
        subject_id=subject_id,
        patch_sampling_rate=patch_sampling_rate,
        num_channels=num_channels,
        num_targets=num_targets,
        window_width=window_width
    )

    # 6. Store for model constructor
    model_params["_task_info_dict"] = task_info_dict

    losses = experiment_config.training_params.losses or []
    if not losses and experiment_config.training_params.loss_name:
        losses = [experiment_config.training_params.loss_name]
    model_params["_training_losses"] = losses
    model_params["_loss_name"] = experiment_config.training_params.loss_name
    model_params["output_activation"] = _resolve_output_activation(model_params)

    if model_params.get("ft_mup") and experiment_config.training_params.optimizer == "AdamW":
        experiment_config.training_params.optimizer = "MuAdamW"

    # 7. Copy output_dim to embedding_dim (same as BrainBERT/PopT)
    model_params["embedding_dim"] = model_params["output_dim"]

    encoder_spec = _find_nested_encoder_model_spec(diver_model_spec, "diver_encoder")
    if encoder_spec is not None:
        encoder_spec.params.update(model_params)

    return experiment_config
