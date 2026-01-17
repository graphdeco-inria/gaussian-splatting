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
Oracle dump utility for capturing training state snapshots.

This module provides functions to capture and save complete training state
for reproducibility verification and debugging purposes.
"""

from __future__ import annotations

import os
import random as pyrandom
import time
from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_cpu(x: Any) -> Any:
    """Move tensor to CPU if it's a CUDA tensor."""
    if torch.is_tensor(x):
        return x.detach().contiguous().cpu()
    return x


def _safe_getattr(obj: Any, name: str) -> Optional[Any]:
    """Safely get attribute from object."""
    return getattr(obj, name, None)


def capture_gaussians(gaussians: Any) -> Dict[str, Any]:
    """
    Capture GaussianModel state.

    The official GaussianModel.capture() returns a tuple, so we convert
    it to a dict with named keys for easier inspection.

    Args:
        gaussians: GaussianModel instance

    Returns:
        Dict containing gaussian parameters
    """
    result: Dict[str, Any] = {}

    # Try capture() first - returns a tuple in official implementation
    if hasattr(gaussians, "capture") and callable(getattr(gaussians, "capture")):
        cap = gaussians.capture()
        if isinstance(cap, tuple):
            # Official capture() returns:
            # (active_sh_degree, _xyz, _features_dc, _features_rest,
            #  _scaling, _rotation, _opacity, max_radii2D,
            #  xyz_gradient_accum, denom, optimizer.state_dict(), spatial_lr_scale)
            keys = [
                "active_sh_degree",
                "_xyz",
                "_features_dc",
                "_features_rest",
                "_scaling",
                "_rotation",
                "_opacity",
                "max_radii2D",
                "xyz_gradient_accum",
                "denom",
                "optimizer_state_dict",
                "spatial_lr_scale",
            ]
            for i, key in enumerate(keys):
                if i < len(cap):
                    result[key] = _to_cpu(cap[i])
            return result
        elif isinstance(cap, dict):
            return {k: _to_cpu(v) for k, v in cap.items()}
        return {"capture": _to_cpu(cap)}

    # Fallback: try state_dict()
    if hasattr(gaussians, "state_dict") and callable(getattr(gaussians, "state_dict")):
        sd = gaussians.state_dict()
        return {k: _to_cpu(v) for k, v in sd.items()}

    # Last resort: directly access common attributes
    cand: Dict[str, Any] = {}
    for name in [
        "_xyz", "_features_dc", "_features_rest",
        "_opacity", "_scaling", "_rotation",
        "max_radii2D", "xyz_gradient_accum", "denom",
    ]:
        v = _safe_getattr(gaussians, name)
        if v is not None:
            cand[name] = _to_cpu(v)

    for name in ["active_sh_degree", "max_sh_degree", "spatial_lr_scale"]:
        v = _safe_getattr(gaussians, name)
        if v is not None and not torch.is_tensor(v):
            cand[name] = v

    if not cand:
        cand["warning"] = "failed_to_capture_gaussians"
        cand["dir"] = sorted([x for x in dir(gaussians) if not x.startswith("__")])[:200]

    return cand


def capture_camera(viewpoint_cam: Any) -> Dict[str, Any]:
    """
    Capture camera parameters.

    Args:
        viewpoint_cam: Camera instance

    Returns:
        Dict containing camera parameters
    """
    out: Dict[str, Any] = {}
    if viewpoint_cam is None:
        return out

    for name in [
        "image_width", "image_height",
        "FoVx", "FoVy",
        "world_view_transform", "full_proj_transform",
        "camera_center",
        "uid", "colmap_id",
        "image_name",
    ]:
        v = _safe_getattr(viewpoint_cam, name)
        if v is None:
            continue
        out[name] = _to_cpu(v)

    return out


def capture_rng() -> Dict[str, Any]:
    """
    Capture all random number generator states.

    Returns:
        Dict containing RNG states for python, numpy, torch, and CUDA
    """
    rng: Dict[str, Any] = {}
    rng["python_random_state"] = pyrandom.getstate()
    rng["numpy_random_state"] = np.random.get_state()
    rng["torch_rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        rng["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return rng


def _recursive_to_cpu(obj: Any) -> Any:
    """Recursively move all tensors in nested structure to CPU."""
    if torch.is_tensor(obj):
        return _to_cpu(obj)
    if isinstance(obj, dict):
        return {k: _recursive_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_recursive_to_cpu(v) for v in obj]
        return type(obj)(t)
    return obj


def dump_oracle(
    *,
    dump_path: str,
    iteration: int,
    gaussians: Any,
    optimizer: Optional[torch.optim.Optimizer] = None,
    viewpoint_cam: Any = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save oracle dump to file.

    Args:
        dump_path: Path to save the dump file
        iteration: Current training iteration
        gaussians: GaussianModel instance
        optimizer: Optimizer instance (optional, can be extracted from gaussians)
        viewpoint_cam: Camera instance for the current view (optional)
        meta: Additional metadata dict (optional)
    """
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)

    # Get optimizer from gaussians if not provided
    if optimizer is None:
        optimizer = _safe_getattr(gaussians, "optimizer")

    # Also capture exposure optimizer if available
    exposure_optimizer = _safe_getattr(gaussians, "exposure_optimizer")

    payload: Dict[str, Any] = {
        "iteration": int(iteration),
        "timestamp": time.time(),
        "gaussians": capture_gaussians(gaussians),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "exposure_optimizer": exposure_optimizer.state_dict() if exposure_optimizer is not None else {},
        "camera": capture_camera(viewpoint_cam),
        "rng": capture_rng(),
        "meta": meta or {},
    }

    # Move all tensors to CPU
    payload = _recursive_to_cpu(payload)

    torch.save(payload, dump_path)
    print(f"\n[ITER {iteration}] Oracle dump saved to {dump_path}")
