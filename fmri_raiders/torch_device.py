"""Resolve torch device for Raider VAE scripts (GPU-first)."""

from __future__ import annotations

import subprocess

import torch


def _cuda_diagnostics() -> str:
    lines = [
        f"  PyTorch {torch.__version__}, built with CUDA: {torch.version.cuda!r}",
    ]
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout:
            head = "\n".join(completed.stdout.strip().splitlines()[:4])
            lines.append("  nvidia-smi (top lines):\n" + "\n".join(f"    {ln}" for ln in head.splitlines()))
        else:
            lines.append(f"  nvidia-smi failed (exit {completed.returncode})")
    except FileNotFoundError:
        lines.append("  nvidia-smi: not found (no driver / not in PATH)")
    except Exception as e:
        lines.append(f"  nvidia-smi: {e}")
    lines.extend(
        [
            "  If you saw “NVIDIA driver … is too old” above: your **GPU driver** is older than",
            "  the **CUDA version PyTorch was compiled for**. Fix one of:",
            "    • Upgrade the NVIDIA driver from https://www.nvidia.com/Download/index.aspx",
            "    • Or reinstall PyTorch with an **older CUDA wheel** that your driver supports, e.g.:",
            "        pip uninstall -y torch torchvision torchaudio",
            "        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "      (try cu121, or cu118 from the same index if needed — see pytorch.org/get-started)",
        ]
    )
    return "\n".join(lines)


def pick_device(mode: str) -> torch.device:
    """
    ``cuda`` — use GPU; raise if CUDA is not available.
    ``cpu`` — CPU only.
    ``auto`` — CUDA if available, else CPU.
    """
    m = mode.lower().strip()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but torch.cuda.is_available() is False.\n\n"
                "Diagnostics:\n"
                f"{_cuda_diagnostics()}\n\n"
                "To train without a GPU for now: add --device auto or --device cpu.\n"
                "Docs: https://pytorch.org/get-started/locally/"
            )
        return torch.device("cuda")
    if m == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device mode: {mode!r} (use cuda, cpu, or auto)")
