"""
Multi-Patient Temporal VAE with cross-reconstruction loss.

Design goal: fix two weaknesses of the original VAE:
  1. Temporal collapse: original VAE collapses (n_elec, T) → scalar (latent_dim,)
     Fix: latent shape is (k, T) — temporal dynamics fully preserved.
  2. Weak shared space: averaging mu_i across patients is only implicitly aligned.
     Fix: cross-reconstruction loss decoder_j(mu_i) → x_j forces all encoders
          into the same coordinate system explicitly.

Architecture
------------
TemporalPatientEncoder:
    Input:  (batch, n_elec_i, T)
    Conv1d(n_elec_i → enc_ch, kernel=3, pad=1) + ReLU + BN
    Conv1d(enc_ch → enc_ch, kernel=3, pad=1) + ReLU + BN
    Conv1d(enc_ch → k, kernel=1)       →  h   (batch, k, T)  # bottleneck
    Conv1d(k → k, kernel=1)            →  mu       (batch, k, T)
    Conv1d(k → k, kernel=1)            →  logvar   (batch, k, T)

TemporalPatientDecoder:
    Input:  z  (batch, k, T)
    ConvTranspose1d(k → dec_ch, kernel=3, pad=1) + ReLU + BN
    ConvTranspose1d(dec_ch → dec_ch, kernel=3, pad=1) + ReLU + BN
    ConvTranspose1d(dec_ch → n_elec_i, kernel=1)
    Output: (batch, n_elec_i, T)

Forward (training):
    mu_i, logvar_i = encoder_i(normalize(x_i))     per patient, shape (batch, k, T)
    mu_avg     = mean(mu_i)                          (batch, k, T)  shared stimulus
    logvar_avg = mean(logvar_i)
    z          = mu_avg + eps * exp(0.5 * logvar_avg)   reparameterization

    x_i_rec = decoder_i(z)                          regular reconstruction
    cross_recs[(j,i)] = decoder_j(mu_i)             cross-reconstruction (i≠j)

    L_recon = (1/N) * sum_i MSE(decoder_i(z), x_i_norm)
    L_cross = (1/N(N-1)) * sum_{i≠j} MSE(decoder_j(mu_i), x_j_norm)
    L_kl    = KL( N(mu_avg, exp(logvar_avg)) || N(0,I) )
    loss    = L_recon + alpha * L_cross + beta * L_kl

Encode (evaluation):
    mu_avg = mean(encoder_i(x_i_norm))   shape (n_samples, k, T)
    → fed directly to neural_conv_decoder
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class TemporalPatientEncoder(nn.Module):
    """Per-patient encoder: (n_elec_i, T) → mu (k, T), logvar (k, T)."""

    def __init__(self, n_electrodes: int, enc_ch: int, shared_channels: int):
        super().__init__()
        k = shared_channels
        self.backbone = nn.Sequential(
            nn.Conv1d(n_electrodes, enc_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(enc_ch),
            nn.Conv1d(enc_ch, enc_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(enc_ch),
            nn.Conv1d(enc_ch, k, kernel_size=1),   # temporal bottleneck
        )
        self.mu_head     = nn.Conv1d(k, k, kernel_size=1)
        self.logvar_head = nn.Conv1d(k, k, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """x: (batch, n_elec, T) → mu, logvar each (batch, k, T)."""
        h = self.backbone(x)
        return self.mu_head(h), self.logvar_head(h)


class TemporalPatientDecoder(nn.Module):
    """Per-patient decoder: (k, T) → (n_elec_i, T)."""

    def __init__(self, n_electrodes: int, dec_ch: int, shared_channels: int):
        super().__init__()
        k = shared_channels
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(k, dec_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dec_ch),
            nn.ConvTranspose1d(dec_ch, dec_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dec_ch),
            nn.ConvTranspose1d(dec_ch, n_electrodes, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, k, T) → (batch, n_elec, T)."""
        return self.deconv(z)


class MultiPatientTemporalVAE(nn.Module):
    """Temporal VAE: latent shape (k, T), cross-reconstruction loss.

    Normalization stats are stored as model buffers and saved/loaded with checkpoint.
    """

    def __init__(
        self,
        n_electrodes_list: list[int],
        enc_ch: int,
        dec_ch: int,
        shared_channels: int,
        input_timesteps: int,
    ):
        super().__init__()
        self._config = {
            "n_electrodes_list": list(n_electrodes_list),
            "enc_ch": enc_ch,
            "dec_ch": dec_ch,
            "shared_channels": shared_channels,
            "input_timesteps": input_timesteps,
        }
        self.shared_channels = shared_channels
        self.input_timesteps = input_timesteps

        self.encoders = nn.ModuleList([
            TemporalPatientEncoder(n, enc_ch, shared_channels)
            for n in n_electrodes_list
        ])
        self.decoders = nn.ModuleList([
            TemporalPatientDecoder(n, dec_ch, shared_channels)
            for n in n_electrodes_list
        ])

        # Per-electrode normalization stats
        for i, n_elec in enumerate(n_electrodes_list):
            self.register_buffer(f"norm_mean_{i}", torch.zeros(1, n_elec, 1))
            self.register_buffer(f"norm_std_{i}",  torch.ones(1, n_elec, 1))

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def set_normalization_stats(self, means: list[torch.Tensor], stds: list[torch.Tensor], eps: float = 1e-6):
        for i, (mean, std) in enumerate(zip(means, stds)):
            getattr(self, f"norm_mean_{i}").data.copy_(mean.float().view(1, -1, 1))
            getattr(self, f"norm_std_{i}").data.copy_(std.float().clamp(min=eps).view(1, -1, 1))

    def _normalize(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return (x - getattr(self, f"norm_mean_{i}")) / getattr(self, f"norm_std_{i}")

    def _denormalize(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return x * getattr(self, f"norm_std_{i}") + getattr(self, f"norm_mean_{i}")

    # ------------------------------------------------------------------
    # Forward (training) — works in normalized space
    # ------------------------------------------------------------------

    def forward(self, xs: list[torch.Tensor]):
        """
        xs: list of (batch, n_elec_i, T) tensors in ORIGINAL scale.

        Returns:
            x_recs_norm: list of (batch, n_elec_i, T) — reconstructions from z_avg
            xs_norm:     list of (batch, n_elec_i, T) — inputs in normalized space
            mu_avg:      (batch, k, T) — shared latent mean
            logvar_avg:  (batch, k, T) — shared latent log-variance
            cross_recs:  dict (j, i) → decoder_j(mu_i), for all i≠j pairs
        """
        xs_norm = [self._normalize(i, x) for i, x in enumerate(xs)]

        mus, logvars = [], []
        for i, xn in enumerate(xs_norm):
            mu, logvar = self.encoders[i](xn)
            mus.append(mu)
            logvars.append(logvar)

        mu_avg     = torch.stack(mus, dim=0).mean(0)       # (batch, k, T)
        logvar_avg = torch.stack(logvars, dim=0).mean(0)

        std = torch.exp(0.5 * logvar_avg)
        z   = mu_avg + torch.randn_like(std) * std         # reparameterize

        # Regular reconstruction from z_avg
        x_recs_norm = [self.decoders[i](z) for i in range(len(xs))]

        # Cross-reconstruction: decoder_j(mu_i) for all i≠j
        cross_recs = {}
        N = len(xs)
        for i in range(N):
            for j in range(N):
                if i != j:
                    cross_recs[(j, i)] = self.decoders[j](mus[i])

        return x_recs_norm, xs_norm, mu_avg, logvar_avg, cross_recs

    # ------------------------------------------------------------------
    # Encode (evaluation) — returns averaged mu across patients
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_avg(self, xs: list, batch_size: int = 256) -> np.ndarray:
        """Encode and average mu across patients.

        Returns: (n_samples, k, T)
        """
        self.eval()
        device    = next(self.parameters()).device
        n_samples = xs[0].shape[0]
        all_h = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xs_batch = [
                torch.tensor(x[start:end], dtype=torch.float32).to(device)
                if isinstance(x, np.ndarray) else x[start:end].float().to(device)
                for x in xs
            ]
            xs_norm = [self._normalize(i, x) for i, x in enumerate(xs_batch)]
            mus = [self.encoders[i](xn)[0] for i, xn in enumerate(xs_norm)]
            mu_avg = torch.stack(mus, dim=0).mean(0)    # (batch, k, T)
            all_h.append(mu_avg.cpu().numpy())

        return np.concatenate(all_h, axis=0)             # (n_samples, k, T)

    # ------------------------------------------------------------------
    # Reconstruct (evaluation) — decode mu_avg back to original electrode space
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, xs: list, batch_size: int = 256) -> list[np.ndarray]:
        """Reconstruct from mu_avg, returning original-scale per-patient arrays.

        Returns: list of (n_samples, n_elec_i, T) in original (denormalized) scale.
        """
        self.eval()
        device    = next(self.parameters()).device
        n_samples = xs[0].shape[0]
        all_recs  = [[] for _ in xs]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xs_batch = [
                torch.tensor(x[start:end], dtype=torch.float32).to(device)
                if isinstance(x, np.ndarray) else x[start:end].float().to(device)
                for x in xs
            ]
            xs_norm = [self._normalize(i, x) for i, x in enumerate(xs_batch)]
            mus     = [self.encoders[i](xn)[0] for i, xn in enumerate(xs_norm)]
            mu_avg  = torch.stack(mus, dim=0).mean(0)   # (batch, k, T)

            for i in range(len(xs)):
                x_rec_norm = self.decoders[i](mu_avg)
                x_rec      = self._denormalize(i, x_rec_norm)
                all_recs[i].append(x_rec.cpu().numpy())

        return [np.concatenate(recs, axis=0) for recs in all_recs]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "config": self._config}, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "MultiPatientTemporalVAE":
        ckpt  = torch.load(path, map_location=map_location or "cpu")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
