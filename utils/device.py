"""Accelerator selection that also sees Intel XPU.

Call sites used ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``.
On an Intel GPU (Aurora) ``torch.cuda.is_available()`` is False, so that
expression silently returns CPU: the run completes, slowly, with no error and
nothing in the log to say the accelerator was skipped. This picks CUDA, then
XPU, then CPU, and the matching helpers below replace the ``torch.cuda.*``
calls that are likewise no-ops off CUDA.

Override with NEUROCAST_DEVICE=cpu|cuda|xpu|cuda:1 when the automatic choice is
wrong (profiling on CPU, pinning one of several visible devices).
"""

import os
import sys

import torch


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _log_choice(device: torch.device, reason: str) -> torch.device:
    # One line per call, so a log shows which device every call site got.
    caller = sys._getframe(2)
    print(
        f"[neurocast.device] pick_device() -> {device} ({reason}) "
        f"at {os.path.relpath(caller.f_code.co_filename)}:{caller.f_lineno}",
        file=sys.stderr,
        flush=True,
    )
    return device


def pick_device() -> torch.device:
    """CUDA, else XPU, else CPU. NEUROCAST_DEVICE overrides."""
    forced = os.environ.get("NEUROCAST_DEVICE")
    if forced:
        return _log_choice(torch.device(forced), "NEUROCAST_DEVICE")
    if torch.cuda.is_available():
        return _log_choice(torch.device("cuda"), "cuda available")
    if _xpu_available():
        return _log_choice(
            torch.device("xpu"), f"xpu available, {torch.xpu.device_count()} visible"
        )
    return _log_choice(torch.device("cpu"), "no accelerator")


def empty_cache(device: "torch.device | None" = None) -> None:
    """Free cached blocks on whichever accelerator is in use."""
    device = device or pick_device()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and _xpu_available():
        torch.xpu.empty_cache()


def manual_seed_all(seed: int) -> None:
    """Seed the accelerator RNG as well as the CPU one."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _xpu_available():
        torch.xpu.manual_seed_all(seed)
