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
            "gpt_surprise": 1,
            "sentence_onset": 1,  # Binary classification: output_dim=1 for BCE loss compatibility
            "content_noncontent": 1,  # Binary classification: output_dim=1 for BCE loss compatibility
            "gpt_surprise_multiclass": 3,
            "pos": 5,
            "volume_level": 1,
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
    
    # Extract only parameters needed for DIVER-1 model loading
    params = Params({
        'foundation_dir': model_params["foundation_dir"],
        'patch_size': model_params.get("patch_size", 500),
        'width': model_params.get("width") or model_params.get("d_model", 512),
        'depth': model_params.get("depth") or model_params.get("e_layer", 12),
        'mup_weights': model_params.get("mup_weights", False),
        'ft_mup': model_params.get("ft_mup", False),
        'ft_config': model_params.get("ft_config", "flatten_linear"),
        'deepspeed_pth_format': model_params.get("deepspeed_pth_format", True),
        'model_dir': model_params.get("model_dir") or \
                    os.path.dirname(model_params["foundation_dir"]) if model_params.get("foundation_dir") else None,
        'num_mlp_layers': model_params.get("num_mlp_layers", 2),
    })
    
    return params


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

class DIVERDecoder(nn.Module):
    """
    DIVER model decoder for Podcast benchmark.
    
    Wraps DIVER-1's FineTuneModel to match Podcast benchmark interface.
    Handles data format conversion and output activation.
    """
    
    def __init__(self, diver_model, output_activation: str = "linear"):
        """
        Args:
            diver_model: DIVER FineTuneModel instance (flatten_linear_finetune or flatten_mlp_finetune)
            output_activation: Activation function for output ("linear", "sigmoid", "softmax")
                - "sigmoid": For binary classification (output_dim=1)
                - "softmax": For multiclass classification (output_dim>1)
                - "linear": No activation (for regression)
        """
        super().__init__()
        self.diver_model = diver_model
        self.output_activation = output_activation
    
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
        # Accept xyz_id directly (flattened approach) or legacy data_info_list
        xyz_id = kwargs.get('xyz_id', None)
        data_info_list = kwargs.get('data_info_list', None)

        batch_size = x.shape[0]
        num_channels = x.shape[1]

        # Build data_info_list for underlying DIVER-1 model
        if xyz_id is not None:
            # xyz_id is [batch_size, num_channels, 3]
            data_info_list = []
            for i in range(batch_size):
                # Convert tensor to numpy if needed
                coords = xyz_id[i]
                if torch.is_tensor(coords):
                    coords = coords.cpu().numpy()
                data_info_list.append({
                    'num_channels': num_channels,
                    'modality': 'iEEG',
                    'xyz_id': coords,
                })
        elif data_info_list is None:
            # Fallback: create default data_info_list without coordinates
            data_info_list = create_data_info_list(batch_size, num_channels)

        # DIVER model forward (outputs logits)
        output = self.diver_model(x, data_info_list=data_info_list)
        
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

    # Extract subject ID and channel names from first raw
    subject_id = extract_subject_id_from_raw(raws[0])
    channel_names = raws[0].ch_names

    # Load MNI coordinates
    data_root = model_params.get("data_root", "data")
    mni_coords = get_mni_coordinates(subject_id, channel_names, data_root)
    print(f"DIVER: Loaded MNI coordinates for subject {subject_id}: shape {mni_coords.shape}")

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
    foundation_dir = model_params["foundation_dir"]
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

        if ft_config == "flatten_linear":
            diver_model = flatten_linear_finetune(params, task_info_dict)
        elif ft_config == "flatten_mlp":
            diver_model = flatten_mlp_finetune(params, task_info_dict)
        else:
            raise ValueError(f"Unknown ft_config: {ft_config}")

        if base_shapes_path is not None:
            mup_set_base_shapes(diver_model, base_shapes_path, rescale_params=False)
    
    
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
    
    # Auto-determine output_activation based on output_dim (same as BrainBERT)
    output_activation = model_params.get("output_activation", None)
    if output_activation is None:
        if output_dim == 1:
            # Binary classification: use sigmoid for BCE loss compatibility
            output_activation = "sigmoid"
        elif output_dim > 1:
            # Multiclass classification: use softmax for cross_entropy loss
            output_activation = "softmax"
        else:
            # Regression or other: no activation
            output_activation = "linear"
    
    # Wrap in DIVERDecoder for Podcast benchmark interface
    decoder = DIVERDecoder(diver_model, output_activation=output_activation)
    
    return decoder


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
    experiment_config = set_input_channels(experiment_config, raws, _df_word)
    experiment_config.data_params.use_stft_preprocessing = False
    
    # 2. Get task information from podcast-benchmark
    task_name = experiment_config.model_params.get("task_name", "word_embedding")
    subject_id = experiment_config.model_params.get("subject_id", 1)
    patch_sampling_rate = experiment_config.model_params.get("patch_sampling_rate", 500)
    num_channels = experiment_config.model_params.get("input_channels")
    
    # 3. Get window_width from data_params
    window_width = getattr(experiment_config.data_params, 'window_width', None)
    if window_width is None or window_width <= 0:
        window_width = 0.5
        print(f"DIVER: Using default window_width={window_width}")
    experiment_config.data_params.window_width = window_width
    
    # 4. Determine output_dim (podcast-benchmark task mapping)
    if "output_dim" not in experiment_config.model_params or experiment_config.model_params["output_dim"] is None:
        num_targets_map = {
            "word_embedding": 50,
            "gpt_surprise": 1,
            "sentence_onset": 1,
            "content_noncontent": 1,
            "gpt_surprise_multiclass": 3,
            "pos": 5,
            "volume_level": 1,
        }
        num_targets = num_targets_map.get(task_name, 50)
        experiment_config.model_params["output_dim"] = num_targets
    else:
        num_targets = experiment_config.model_params["output_dim"]
    
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
    experiment_config.model_params["_task_info_dict"] = task_info_dict
    
    # 7. Copy output_dim to embedding_dim (same as BrainBERT/PopT)
    experiment_config.model_params["embedding_dim"] = experiment_config.model_params["output_dim"]
    
    return experiment_config



