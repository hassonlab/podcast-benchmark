"""
Registration of VAE components into the podcast-benchmark registry.

Registered items:
    - data preprocessor  "vae_reconstruct"      : applies pretrained VAE reconstruction
    - config setter      "neural_conv_vae"      : extends neural_conv, stores patient electrode indices
"""

import numpy as np
import torch
from core import registry
from core.config import ExperimentConfig
from models.shared_config_setters import set_input_channels, set_model_spec_fields
from models.neural_conv_decoder.neural_conv_utils import neural_conv_config_setter


# ---------------------------------------------------------------------------
# Module-level cache so the VAE checkpoint is loaded only once per process
# ---------------------------------------------------------------------------
_vae_cache: dict = {}


def _load_vae(checkpoint_path: str):
    """Load (and cache) a MultiPatientVAE from checkpoint_path."""
    if checkpoint_path not in _vae_cache:
        from shared_space.models.patient_vae import MultiPatientVAE
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = MultiPatientVAE.load(checkpoint_path, map_location="cpu").to(device)
        vae.eval()
        _vae_cache[checkpoint_path] = vae
    return _vae_cache[checkpoint_path]


# ---------------------------------------------------------------------------
# Data preprocessor: "vae_reconstruct"
# ---------------------------------------------------------------------------

@registry.register_data_preprocessor("vae_reconstruct")
def vae_reconstruct(data: np.ndarray, preprocessor_params: dict) -> np.ndarray:
    """Apply pretrained VAE reconstruction to binned neural data.

    Expects preprocessor_params to contain:
        vae_checkpoint_path (str): path to the saved MultiPatientVAE checkpoint
        patient_electrode_indices (list[list[int]]): electrode indices per patient
            in the concatenated data array (set automatically by neural_conv_vae
            config setter)
        vae_batch_size (int, optional): mini-batch size for reconstruction, default 256

    Args:
        data: numpy array (n_samples, n_total_electrodes, T)
              already averaged into T time bins (output of window_average_neural_data)

    Returns:
        numpy array of same shape — VAE-reconstructed signals
    """
    checkpoint_path = preprocessor_params["vae_checkpoint_path"]
    patient_electrode_indices = preprocessor_params.get("patient_electrode_indices")
    vae_batch_size = preprocessor_params.get("vae_batch_size", 256)

    if patient_electrode_indices is None:
        raise ValueError(
            "vae_reconstruct: patient_electrode_indices is None. "
            "Make sure you use the 'neural_conv_vae' config_setter_name in your config."
        )

    vae = _load_vae(checkpoint_path)

    # Split concatenated data into per-patient arrays
    xs = [data[:, idx] for idx in patient_electrode_indices]

    # Reconstruct (uses mu_avg, no sampling)
    x_recs = vae.reconstruct(xs, batch_size=vae_batch_size)

    # Re-concatenate in original order
    return np.concatenate(x_recs, axis=1).astype(data.dtype)


# ---------------------------------------------------------------------------
# Config setter: "neural_conv_vae"
# ---------------------------------------------------------------------------

@registry.register_config_setter("neural_conv_vae")
def neural_conv_vae_config_setter(
    experiment_config: ExperimentConfig, raws, task_df
) -> ExperimentConfig:
    """Extends neural_conv config setter with VAE-specific setup.

    In addition to the standard neural_conv processing (input_channels,
    input_timesteps), this setter:
        1. Computes per-patient electrode indices in the concatenated array.
        2. Injects them into the preprocessor_params of the vae_reconstruct
           preprocessor (identified by the presence of 'vae_checkpoint_path').
    """
    # Run the standard neural_conv setter first
    experiment_config = neural_conv_config_setter(experiment_config, raws, task_df)

    # Compute per-patient electrode index ranges
    patient_electrode_indices = []
    offset = 0
    for raw in raws:
        n = len(raw.ch_names)
        patient_electrode_indices.append(list(range(offset, offset + n)))
        offset += n

    # Inject into the vae_reconstruct preprocessor params
    pp = experiment_config.task_config.data_params.preprocessor_params
    if isinstance(pp, list):
        for params_dict in pp:
            if isinstance(params_dict, dict) and "vae_checkpoint_path" in params_dict:
                params_dict["patient_electrode_indices"] = patient_electrode_indices
                break
    elif isinstance(pp, dict) and "vae_checkpoint_path" in pp:
        pp["patient_electrode_indices"] = patient_electrode_indices

    return experiment_config
