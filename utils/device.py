"""
Shared device resolution for all PyTorch-based backends.

Every model and trainer that needs to pick a device imports from here
instead of carrying its own copy of the same logic.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def resolve_torch_device(
    device: str = "auto",
    *,
    backend_name: str = "model",
    install_hint: str = "",
) -> str:
    """
    Resolve a user-facing device string to a concrete PyTorch device name.

    Args:
        device: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.
        backend_name: Human-readable name for error messages (e.g. ``"GigaAM v3"``).
        install_hint: Multi-line install suggestion appended to ImportError.

    Returns:
        Resolved device string (``"cuda"``, ``"mps"``, or ``"cpu"``).

    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If an explicit device is requested but unavailable.
        ValueError: If ``device`` is not a recognised string.
    """
    if torch is None:
        msg = f"PyTorch is required for {backend_name}."
        if install_hint:
            msg = f"{msg}\n{install_hint}"
        raise ImportError(msg)

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested for {backend_name}, but no NVIDIA GPU is available."
            )
        return "cuda"

    if device == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError(
                f"MPS requested for {backend_name}, but it is unavailable."
            )
        return "mps"

    if device == "cpu":
        return "cpu"

    raise ValueError(f"device must be one of: auto, cuda, mps, cpu (got '{device}')")


def auto_select_device() -> str:
    """
    Pick the best available device without raising on missing hardware.

    Used by the trainer where the user does not pass a device string.
    """
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
