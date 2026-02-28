"""
Multi-Patient VAE for podcast iEEG denoising.

Each patient has its own 1D-conv encoder and decoder operating over
the time-bin dimension (default: 10 bins from 625 ms window).

All patient encoders project to a shared latent space:
  mu_avg     = mean(mu_1, ..., mu_N)
  log_var_avg = mean(log_var_1, ..., log_var_N)  ← geometric mean of variances
  z           = mu_avg + eps * exp(0.5 * log_var_avg)  ← reparameterization

Normalization:
  Per-electrode z-score statistics are stored as model buffers and applied
  internally. The VAE encodes/decodes in normalized space; `reconstruct()`
  returns data in the original amplitude scale.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatientEncoder(nn.Module):
    """1D-conv encoder for a single patient.

    Architecture (time is the conv dimension, electrodes are channels):
        Input: (batch, n_electrodes, input_timesteps)
        Conv1d(n_elec → ch[0], k=3, pad=1) + ReLU + BN
        Conv1d(ch[0] → ch[1], k=3, pad=1) + ReLU + BN
        Flatten → FC → mu, log_var  each of shape (batch, latent_dim)
    """

    def __init__(
        self,
        n_electrodes: int,
        encoder_channels: list[int],
        latent_dim: int,
        input_timesteps: int,
    ):
        super().__init__()
        assert len(encoder_channels) == 2, "encoder_channels must have exactly 2 values"

        self.conv = nn.Sequential(
            nn.Conv1d(n_electrodes, encoder_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[0]),
            nn.Conv1d(encoder_channels[0], encoder_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[1]),
        )
        flat_dim = encoder_channels[1] * input_timesteps
        self.mu_head = nn.Linear(flat_dim, latent_dim)
        self.log_var_head = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """x: (batch, n_electrodes, T) → (mu, log_var) each (batch, latent_dim)"""
        h = self.conv(x)          # (batch, ch[-1], T)
        h = h.flatten(1)          # (batch, ch[-1] * T)
        return self.mu_head(h), self.log_var_head(h)


class PatientDecoder(nn.Module):
    """1D-conv decoder for a single patient.

    Architecture:
        Input: z (batch, latent_dim)
        FC → reshape (batch, ch[0], input_timesteps)
        ConvTranspose1d(ch[0] → ch[1], k=3, pad=1) + ReLU + BN
        ConvTranspose1d(ch[1] → n_electrodes, k=3, pad=1)
    """

    def __init__(
        self,
        n_electrodes: int,
        decoder_channels: list[int],
        latent_dim: int,
        input_timesteps: int,
    ):
        super().__init__()
        assert len(decoder_channels) == 2, "decoder_channels must have exactly 2 values"

        self.input_timesteps = input_timesteps
        flat_dim = decoder_channels[0] * input_timesteps
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(decoder_channels[0], decoder_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[1]),
            nn.ConvTranspose1d(decoder_channels[1], n_electrodes, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor):
        """z: (batch, latent_dim) → x_rec: (batch, n_electrodes, T)"""
        h = self.fc(z)                                               # (batch, ch[0]*T)
        h = h.view(h.shape[0], -1, self.input_timesteps)            # (batch, ch[0], T)
        return self.deconv(h)                                        # (batch, n_elec, T)


class MultiPatientVAE(nn.Module):
    """VAE with one encoder and one decoder per patient.

    The VAE operates internally in per-electrode z-score space.
    Call `set_normalization_stats()` once after construction (before training)
    to set the statistics. They are stored as model buffers and saved/loaded
    with the checkpoint automatically.

    Forward pass (training):
        xs: list of (batch, n_elec_i, T) tensors in ORIGINAL scale
        Returns: (x_recs_norm, xs_norm, mu_avg, log_var_avg)
                 x_recs_norm and xs_norm are in NORMALIZED space — pass to vae_loss()

    Reconstruction pass (evaluation):
        reconstruct() takes original-scale data and returns original-scale reconstructions.
    """

    def __init__(
        self,
        n_electrodes_list: list[int],
        encoder_channels: list[int],
        decoder_channels: list[int],
        latent_dim: int,
        input_timesteps: int,
    ):
        super().__init__()
        self._config = {
            "n_electrodes_list": list(n_electrodes_list),
            "encoder_channels": list(encoder_channels),
            "decoder_channels": list(decoder_channels),
            "latent_dim": latent_dim,
            "input_timesteps": input_timesteps,
        }
        self.encoders = nn.ModuleList([
            PatientEncoder(n, encoder_channels, latent_dim, input_timesteps)
            for n in n_electrodes_list
        ])
        self.decoders = nn.ModuleList([
            PatientDecoder(n, decoder_channels, latent_dim, input_timesteps)
            for n in n_electrodes_list
        ])
        self.latent_dim = latent_dim
        self.input_timesteps = input_timesteps

        # Per-electrode normalization stats — shape (1, n_elec_i, 1) for broadcasting
        for i, n_elec in enumerate(n_electrodes_list):
            self.register_buffer(f"norm_mean_{i}", torch.zeros(1, n_elec, 1))
            self.register_buffer(f"norm_std_{i}", torch.ones(1, n_elec, 1))

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def set_normalization_stats(
        self,
        means: list[torch.Tensor],
        stds: list[torch.Tensor],
        eps: float = 1e-6,
    ):
        """Set per-patient, per-electrode z-score statistics.

        Args:
            means: list of (n_elec_i,) tensors — per-electrode mean
            stds:  list of (n_elec_i,) tensors — per-electrode std
            eps:   floor for std to avoid division by zero
        """
        for i, (mean, std) in enumerate(zip(means, stds)):
            getattr(self, f"norm_mean_{i}").data.copy_(
                mean.float().view(1, -1, 1)
            )
            getattr(self, f"norm_std_{i}").data.copy_(
                std.float().clamp(min=eps).view(1, -1, 1)
            )

    def _normalize(self, i: int, x: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, f"norm_mean_{i}")
        std  = getattr(self, f"norm_std_{i}")
        return (x - mean) / std

    def _denormalize(self, i: int, x: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, f"norm_mean_{i}")
        std  = getattr(self, f"norm_std_{i}")
        return x * std + mean

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, xs: list[torch.Tensor]):
        """
        xs: list of (batch, n_elec_i, T) tensors in ORIGINAL scale

        Returns:
            x_recs_norm: list of (batch, n_elec_i, T) — reconstructions in NORMALIZED space
            xs_norm:     list of (batch, n_elec_i, T) — inputs in NORMALIZED space
            mu_avg:      (batch, latent_dim)
            log_var_avg: (batch, latent_dim)

        Pass x_recs_norm and xs_norm to vae_loss() so that the MSE is computed
        in normalized space — this makes all patients contribute equally regardless
        of their amplitude scale.
        """
        # Normalize inputs
        xs_norm = [self._normalize(i, x) for i, x in enumerate(xs)]

        mus, log_vars = [], []
        for i, x in enumerate(xs_norm):
            mu, log_var = self.encoders[i](x)
            mus.append(mu)
            log_vars.append(log_var)

        mu_avg      = torch.stack(mus, dim=0).mean(0)      # (batch, latent_dim)
        log_var_avg = torch.stack(log_vars, dim=0).mean(0)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var_avg)
        z   = mu_avg + torch.randn_like(std) * std

        # Decode in normalized space
        x_recs_norm = [self.decoders[i](z) for i in range(len(xs))]

        return x_recs_norm, xs_norm, mu_avg, log_var_avg

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def vae_loss(
        self,
        x_recs_norm: list[torch.Tensor],
        xs_norm: list[torch.Tensor],
        mu_avg: torch.Tensor,
        log_var_avg: torch.Tensor,
        beta: float = 1.0,
    ):
        """VAE loss computed in normalized space for equal patient contribution.

        Returns:
            total_loss:       scalar tensor
            mse_avg:          float (average normalized MSE across patients)
            mse_per_patient:  list of floats (normalized MSE per patient)
            kl:               float
        """
        mse_tensors = [F.mse_loss(x_rec, x) for x_rec, x in zip(x_recs_norm, xs_norm)]
        total_mse   = torch.stack(mse_tensors).mean()

        # KL divergence: KL(N(mu_avg, exp(log_var_avg)) || N(0, I))
        kl = -0.5 * torch.mean(1 + log_var_avg - mu_avg.pow(2) - log_var_avg.exp())

        total_loss = total_mse + beta * kl

        return (
            total_loss,
            total_mse.item(),
            [m.item() for m in mse_tensors],
            kl.item(),
        )

    # ------------------------------------------------------------------
    # Reconstruction (evaluation) — returns ORIGINAL scale
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(
        self,
        xs: list,
        batch_size: int = 256,
    ) -> list[np.ndarray]:
        """Reconstruct using mu_avg (no sampling noise) — for evaluation.

        Args:
            xs:         list of (n_samples, n_elec_i, T) numpy arrays or tensors
                        in ORIGINAL (un-normalized) scale
            batch_size: mini-batch size to avoid OOM

        Returns:
            list of (n_samples, n_elec_i, T) numpy arrays in ORIGINAL scale
        """
        device   = next(self.parameters()).device
        n_samples = xs[0].shape[0] if hasattr(xs[0], "shape") else len(xs[0])

        all_recs = [[] for _ in xs]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            xs_batch = []
            for x in xs:
                t = (
                    torch.tensor(x[start:end], dtype=torch.float32)
                    if isinstance(x, np.ndarray)
                    else x[start:end].float()
                )
                xs_batch.append(t.to(device))

            # Encode in normalized space, average mu
            mus = []
            for i, x in enumerate(xs_batch):
                mu, _ = self.encoders[i](self._normalize(i, x))
                mus.append(mu)
            mu_avg = torch.stack(mus, dim=0).mean(0)

            # Decode in normalized space, then denormalize
            for i in range(len(xs)):
                x_rec_norm = self.decoders[i](mu_avg)
                x_rec      = self._denormalize(i, x_rec_norm)
                all_recs[i].append(x_rec.cpu().numpy())

        return [np.concatenate(recs, axis=0) for recs in all_recs]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save model state (including normalization buffers) and config."""
        torch.save(
            {"state_dict": self.state_dict(), "config": self._config},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "MultiPatientVAE":
        """Load a model from checkpoint. Normalization stats are restored from state_dict."""
        checkpoint = torch.load(path, map_location=map_location or "cpu")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
