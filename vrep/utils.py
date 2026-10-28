"""Small shared helpers: device selection, seeding, image conversion."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    prefer = prefer or os.environ.get("VREP_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rgb_to_tensor(rgb: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """uint8 (...,H,W,3) -> float (...,3,H,W) in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
    t = t.movedim(-1, -3)
    return t.to(device) if device is not None else t


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """float (...,3,H,W) in [0,1] -> uint8 (...,H,W,3)."""
    return (t.detach().clamp(0, 1).movedim(-3, -1).cpu().numpy() * 255).round().astype(np.uint8)


def count_parameters(m: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
