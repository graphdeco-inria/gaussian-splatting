#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
Device abstraction utilities for Intel GPU (XPU) support.

Provides auto-detection of the best available accelerator and thin
wrappers that replace CUDA-specific torch APIs:

    torch.cuda.Event          -> device_event() / DeviceTimer
    torch.cuda.empty_cache()  -> device_empty_cache()
    torch.cuda.set_device()   -> device_set_device()
    torch.cuda.is_available() -> device_is_available()
    tensor.cuda()             -> tensor.to(DEVICE)

The module-level constant ``DEVICE`` is set once at import time to
"xpu" (Intel Arc/Xe), "cuda" (NVIDIA), or "cpu" as a fallback.
"""

import time
import torch


def _detect_device() -> str:
    """Return the best available accelerator device string."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Public constant – use as the target device throughout the codebase.
# ---------------------------------------------------------------------------
DEVICE: str = _detect_device()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class DeviceTimer:
    """
    Portable wall-clock timer that mirrors the ``torch.cuda.Event`` API
    used for training-loop iteration timing.

    Usage::

        start = device_event()
        end   = device_event()
        start.record()
        # ... GPU work ...
        end.record()
        elapsed_ms = start.elapsed_time(end)
    """

    def __init__(self) -> None:
        self._time: float | None = None

    def record(self) -> None:
        """Snapshot the current wall-clock time."""
        self._time = time.perf_counter()

    def elapsed_time(self, end: "DeviceTimer") -> float:
        """
        Return milliseconds elapsed between *self* (start) and *end*.
        Matches the ``torch.cuda.Event.elapsed_time(end)`` signature.
        """
        if self._time is None or end._time is None:
            return 0.0
        return (end._time - self._time) * 1000.0


def device_event() -> DeviceTimer:
    """Return a new timing event. Replaces ``torch.cuda.Event(enable_timing=True)``."""
    return DeviceTimer()


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def device_empty_cache() -> None:
    """Release unused cached memory on the active device.

    Replaces ``torch.cuda.empty_cache()``.
    """
    if DEVICE == "xpu":
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def device_set_device(device) -> None:
    """Set the active device index.

    Replaces ``torch.cuda.set_device()``.

    Args:
        device: A ``torch.device``, integer index, or device string.
    """
    if DEVICE == "xpu":
        if hasattr(torch, "xpu"):
            torch.xpu.set_device(device)
    elif DEVICE == "cuda":
        torch.cuda.set_device(device)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def device_is_available() -> bool:
    """Return True if a hardware accelerator is available.

    Replaces ``torch.cuda.is_available()``.
    """
    if DEVICE == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if DEVICE == "cuda":
        return torch.cuda.is_available()
    return False
