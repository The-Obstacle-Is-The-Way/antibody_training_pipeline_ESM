"""
Device resolution helpers shared across training and inference.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    """Return True if MPS is available and supported in this build."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device: str | None) -> str:
    """
    Resolve a requested device string to a concrete, available device.

    Rules:
    - If device is None or "auto": prefer CUDA, then MPS, else CPU.
    - If an explicit device is requested but unavailable, raise a clear error.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested device 'cuda' but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or choose hardware.device=cpu."
        )

    if device == "mps" and not _mps_available():
        raise RuntimeError(
            "Requested device 'mps' but torch.backends.mps.is_available() is False. "
            "Use hardware.device=cpu or install a PyTorch build with MPS support."
        )

    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unknown device '{device}'. Expected one of: cpu, cuda, mps, or auto."
        )

    return device
