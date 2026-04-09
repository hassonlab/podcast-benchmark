"""
Registration of shared-space components into the podcast-benchmark registry.

Registered items:
    Experiment 2 — Global PCA (k=8) Reconstruct→Decode:
        - data preprocessor  "global_pca_reconstruct"       : PCA denoise → 183 electrodes
        - (uses standard "neural_conv" config setter)

    Experiment 3 — Temporal VAE Reconstruct→Decode:
        - data preprocessor  "temporal_vae_reconstruct"     : encode→avg→decode back to 183 electrodes
        - config setter      "neural_conv_temporal_vae"     : injects patient_electrode_indices

    Experiment 4 — SRM (k=8) Encode→Decode:
        - data preprocessor  "srm_encode"                   : project to shared (k, T), avg across patients
        - config setter      "neural_conv_srm_encode"       : sets input_channels=k

    Experiment 5 — Temporal VAE Encode→Decode (best):
        - data preprocessor  "temporal_vae_encode"          : encode to mu_avg (k, T)
        - config setter      "neural_conv_temporal_vae_encode": sets input_channels=k
"""

import pickle

import numpy as np
import torch
from core import registry
from core.config import ExperimentConfig
from models.shared_config_setters import set_input_channels, set_model_spec_fields
from models.neural_conv_decoder.neural_conv_utils import neural_conv_config_setter


# ---------------------------------------------------------------------------
# Helpers: compute patient electrode index ranges
# ---------------------------------------------------------------------------

def _compute_patient_electrode_indices(raws) -> list[list[int]]:
    """Compute per-patient electrode indices in the concatenated data array."""
    indices = []
    offset = 0
    for raw in raws:
        n = len(raw.ch_names)
        indices.append(list(range(offset, offset + n)))
        offset += n
    return indices


def _inject_indices_into_preprocessor_params(pp, key: str, patient_electrode_indices):
    """Find the preprocessor_params dict containing `key` and inject indices."""
    if isinstance(pp, list):
        for params_dict in pp:
            if isinstance(params_dict, dict) and key in params_dict:
                params_dict["patient_electrode_indices"] = patient_electrode_indices
                return params_dict.get(key)
    elif isinstance(pp, dict) and key in pp:
        pp["patient_electrode_indices"] = patient_electrode_indices
        return pp[key]
    return None


# ===========================================================================
# Global PCA (Reconstruct→Decode)
# ===========================================================================

_global_pca_cache: dict = {}


def _load_global_pca(checkpoint_path: str) -> dict:
    """Load (and cache) global PCA projection matrix from checkpoint."""
    if checkpoint_path not in _global_pca_cache:
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        W = np.array(ckpt["W"], dtype=np.float32)   # (n_elec_total, k)
        _global_pca_cache[checkpoint_path] = {
            "P":          (W @ W.T).astype(np.float32),
            "norm_means": np.array(ckpt["norm_means_flat"], dtype=np.float32),
            "norm_stds":  np.array(ckpt["norm_stds_flat"],  dtype=np.float32),
        }
    return _global_pca_cache[checkpoint_path]


@registry.register_data_preprocessor("global_pca_reconstruct")
def global_pca_reconstruct(data: np.ndarray, preprocessor_params: dict) -> np.ndarray:
    """Apply global PCA reconstruction (Reconstruct→Decode strategy).

    Projects all 183 electrodes jointly:  x_rec = W @ W^T @ x_norm → denormalize.
    Returns (n_samples, n_total_electrodes, T) — same shape as input.
    """
    checkpoint_path = preprocessor_params["global_pca_checkpoint_path"]
    ckpt       = _load_global_pca(checkpoint_path)
    P          = ckpt["P"]
    norm_means = ckpt["norm_means"]
    norm_stds  = ckpt["norm_stds"]

    x_norm = (data - norm_means[None, :, None]) / norm_stds[None, :, None]

    n_samples, n_elec, T = x_norm.shape
    x_flat     = x_norm.transpose(1, 0, 2).reshape(n_elec, -1)
    x_rec_flat = P @ x_flat
    x_rec_norm = x_rec_flat.reshape(n_elec, n_samples, T).transpose(1, 0, 2)

    return (x_rec_norm * norm_stds[None, :, None] + norm_means[None, :, None]).astype(data.dtype)


# ===========================================================================
# SRM Encode→Decode
# ===========================================================================

_srm_cache: dict = {}


def _load_srm(checkpoint_path: str) -> dict:
    """Load (and cache) SRM projection matrices from checkpoint_path."""
    if checkpoint_path not in _srm_cache:
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        W_matrices = [np.array(W) for W in ckpt["W"]]
        _srm_cache[checkpoint_path] = {
            "W":          W_matrices,
            "norm_means": [np.array(m, dtype=np.float32) for m in ckpt["norm_means"]],
            "norm_stds":  [np.array(s, dtype=np.float32) for s in ckpt["norm_stds"]],
        }
    return _srm_cache[checkpoint_path]


@registry.register_data_preprocessor("srm_encode")
def srm_encode(data: np.ndarray, preprocessor_params: dict) -> np.ndarray:
    """Project each patient to SRM shared space and average (Encode→Decode strategy).

    For each patient i:  s_i = W_i^T @ x_i_norm   shape (k, T)
    Average:             s   = mean(s_i)            shape (k, T)

    Returns (n_samples, k, T) — fed directly to the word decoder.
    """
    checkpoint_path           = preprocessor_params["srm_checkpoint_path"]
    patient_electrode_indices = preprocessor_params.get("patient_electrode_indices")

    if patient_electrode_indices is None:
        raise ValueError(
            "srm_encode: patient_electrode_indices is None. "
            "Use 'neural_conv_srm_encode' as config_setter_name."
        )

    ckpt       = _load_srm(checkpoint_path)
    W_matrices = ckpt["W"]
    norm_means = ckpt["norm_means"]
    norm_stds  = ckpt["norm_stds"]

    shared_reps = []
    for i, idx in enumerate(patient_electrode_indices):
        x      = data[:, idx, :]
        mean   = norm_means[i][None, :, None]
        std    = norm_stds[i][None, :, None]
        x_norm = (x - mean) / std

        W_i = W_matrices[i]
        n_samples, n_elec, T = x_norm.shape
        x_flat = x_norm.transpose(1, 0, 2).reshape(n_elec, -1)
        s_flat = W_i.T @ x_flat
        k      = W_i.shape[1]
        s      = s_flat.reshape(k, n_samples, T).transpose(1, 0, 2)
        shared_reps.append(s)

    return np.mean(shared_reps, axis=0).astype(data.dtype)


@registry.register_config_setter("neural_conv_srm_encode")
def neural_conv_srm_encode_config_setter(
    experiment_config: ExperimentConfig, raws, task_df
) -> ExperimentConfig:
    """Sets input_channels=k (shared space dimension from SRM checkpoint)."""
    experiment_config = neural_conv_config_setter(experiment_config, raws, task_df)
    indices = _compute_patient_electrode_indices(raws)
    pp = experiment_config.task_config.data_params.preprocessor_params
    ckpt_path = _inject_indices_into_preprocessor_params(pp, "srm_checkpoint_path", indices)

    if ckpt_path:
        ckpt_data = _load_srm(ckpt_path)
        k = ckpt_data["W"][0].shape[1]
        relevant = ["pitom_model", "ensemble_pitom_model", "decoder_mlp"]
        set_model_spec_fields(experiment_config.model_spec, {"input_channels": k}, relevant)

    return experiment_config


# ===========================================================================
# Temporal VAE: shared loader + cache
# ===========================================================================

_temporal_vae_cache: dict = {}


def _load_temporal_vae(checkpoint_path: str):
    """Load (and cache) a MultiPatientTemporalVAE from checkpoint_path."""
    if checkpoint_path not in _temporal_vae_cache:
        from shared_space.models.patient_temporal_vae import MultiPatientTemporalVAE
        model = MultiPatientTemporalVAE.load(checkpoint_path, map_location="cpu")
        model.eval()
        _temporal_vae_cache[checkpoint_path] = model
    return _temporal_vae_cache[checkpoint_path]


# ===========================================================================
# Temporal VAE Reconstruct→Decode
# ===========================================================================

@registry.register_data_preprocessor("temporal_vae_reconstruct")
def temporal_vae_reconstruct(data: np.ndarray, preprocessor_params: dict) -> np.ndarray:
    """Reconstruct electrode signals from TemporalVAE (Reconstruct→Decode strategy).

    Encode all patients → mu_avg (k, T) → decode back to each patient's
    electrode space → concatenate → (n_samples, n_total_electrodes, T).
    """
    checkpoint_path           = preprocessor_params["temporal_vae_checkpoint_path"]
    patient_electrode_indices = preprocessor_params.get("patient_electrode_indices")
    batch_size                = preprocessor_params.get("vae_batch_size", 256)

    if patient_electrode_indices is None:
        raise ValueError(
            "temporal_vae_reconstruct: patient_electrode_indices is None. "
            "Use 'neural_conv_temporal_vae' as config_setter_name."
        )

    model  = _load_temporal_vae(checkpoint_path)
    xs     = [data[:, idx] for idx in patient_electrode_indices]
    x_recs = model.reconstruct(xs, batch_size=batch_size)
    return np.concatenate(x_recs, axis=1).astype(data.dtype)


@registry.register_config_setter("neural_conv_temporal_vae")
def neural_conv_temporal_vae_config_setter(
    experiment_config: ExperimentConfig, raws, task_df
) -> ExperimentConfig:
    """Injects patient_electrode_indices for temporal VAE reconstruct path."""
    experiment_config = neural_conv_config_setter(experiment_config, raws, task_df)
    indices = _compute_patient_electrode_indices(raws)
    pp = experiment_config.task_config.data_params.preprocessor_params
    _inject_indices_into_preprocessor_params(pp, "temporal_vae_checkpoint_path", indices)
    return experiment_config


# ===========================================================================
# Temporal VAE Encode→Decode (best performing)
# ===========================================================================

@registry.register_data_preprocessor("temporal_vae_encode")
def temporal_vae_encode(data: np.ndarray, preprocessor_params: dict) -> np.ndarray:
    """Encode to shared latent mu_avg (Encode→Decode strategy, best performing).

    For each patient i:  mu_i = encoder_i(normalize(x_i))   shape (k, T)
    Average:             mu_avg = mean(mu_i)                  shape (k, T)

    Returns (n_samples, k, T) — k "virtual electrodes", fed directly to decoder.
    """
    checkpoint_path           = preprocessor_params["temporal_vae_checkpoint_path"]
    patient_electrode_indices = preprocessor_params.get("patient_electrode_indices")
    batch_size                = preprocessor_params.get("vae_batch_size", 256)

    if patient_electrode_indices is None:
        raise ValueError(
            "temporal_vae_encode: patient_electrode_indices is None. "
            "Use 'neural_conv_temporal_vae_encode' as config_setter_name."
        )

    model = _load_temporal_vae(checkpoint_path)
    xs    = [data[:, idx] for idx in patient_electrode_indices]
    return model.encode_avg(xs, batch_size=batch_size).astype(data.dtype)


@registry.register_config_setter("neural_conv_temporal_vae_encode")
def neural_conv_temporal_vae_encode_config_setter(
    experiment_config: ExperimentConfig, raws, task_df
) -> ExperimentConfig:
    """Sets input_channels=k (shared_channels from temporal VAE checkpoint)."""
    experiment_config = neural_conv_config_setter(experiment_config, raws, task_df)
    indices = _compute_patient_electrode_indices(raws)
    pp = experiment_config.task_config.data_params.preprocessor_params
    ckpt_path = _inject_indices_into_preprocessor_params(pp, "temporal_vae_checkpoint_path", indices)

    if ckpt_path:
        model    = _load_temporal_vae(ckpt_path)
        k        = model.shared_channels
        relevant = ["pitom_model", "ensemble_pitom_model", "decoder_mlp"]
        set_model_spec_fields(
            experiment_config.model_spec,
            {"input_channels": k},
            relevant,
        )

    return experiment_config
