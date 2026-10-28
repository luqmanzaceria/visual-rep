"""DINOv2 patch features: a purely 2D representation of what is visible."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from vrep.repr.base import Representation, TokenBatch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    """Frozen DINOv2 backbone returning normalised patch tokens."""

    def __init__(self, name: str = "dinov2_vits14", input_size: int = 224, device: torch.device | str = "cpu"):
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.patch = 14
        assert input_size % self.patch == 0
        self.grid = input_size // self.patch
        self.model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        self.model.eval().requires_grad_(False)
        self.dim = self.model.embed_dim
        self._freeze_pos_embed_for_input_size()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def _freeze_pos_embed_for_input_size(self) -> None:
        """Interpolate the positional embedding to our patch grid once, on CPU.

        DINOv2 otherwise re-interpolates it (bicubic) on every forward pass, and
        bicubic upsampling is not implemented on the MPS backend.
        """
        m = self.model
        n_tok = self.n_tokens
        if m.pos_embed.shape[1] == 1 + n_tok:
            return
        dummy = torch.zeros(1, 1 + n_tok, self.dim)
        pe = m.interpolate_pos_encoding(dummy, self.input_size, self.input_size)
        m.pos_embed = nn.Parameter(pe.detach().clone(), requires_grad=False)

    @property
    def n_tokens(self) -> int:
        return self.grid * self.grid

    def grid_positions(self) -> torch.Tensor:
        """(N, 2) patch centres in normalised image coordinates (u, v) in [0, 1]."""
        g = self.grid
        c = (torch.arange(g, dtype=torch.float32) + 0.5) / g
        v, u = torch.meshgrid(c, c, indexing="ij")
        return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb float (B,3,H,W) in [0,1] -> patch tokens (B, N, D)."""
        x = rgb.to(self.mean.device)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]

    @torch.no_grad()
    def encode_uint8(self, rgb: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """uint8 (M,H,W,3) -> float16 (M, N, D) encoded in batches."""
        outs = []
        for i in range(0, rgb.shape[0], batch_size):
            x = torch.from_numpy(np.ascontiguousarray(rgb[i : i + batch_size])).float().div_(255).permute(0, 3, 1, 2)
            outs.append(self(x).to(torch.float16).cpu().numpy())
        return np.concatenate(outs, 0)


class FeatureCache:
    """Memory-mapped cache of patch features for a whole dataset: (E, T+1, C, N, D) float16."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path.with_suffix(".json")) as f:
            self.info = json.load(f)
        self.arr = np.load(self.path, mmap_mode="r")

    @property
    def shape(self):
        return self.arr.shape

    def get(self, e: int, t: int, c: int) -> np.ndarray:
        return np.asarray(self.arr[e, t, c])

    @staticmethod
    def build(store, encoder: DinoEncoder, path: Path, batch_size: int = 128, overwrite: bool = False) -> "FeatureCache":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        E, T1, C = len(store), store.T + 1, store.cameras.n
        info = {"encoder": encoder.name, "input_size": encoder.input_size, "shape": [E, T1, C, encoder.n_tokens, encoder.dim]}
        if path.exists() and not overwrite:
            with open(path.with_suffix(".json")) as f:
                if json.load(f) == info:
                    return FeatureCache(path)
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=tuple(info["shape"]))
        for e in tqdm(range(E), desc=f"encoding {encoder.name}"):
            rgb = store[e].rgb  # (T+1, C, H, W, 3)
            flat = rgb.reshape(-1, *rgb.shape[2:])
            feats = encoder.encode_uint8(flat, batch_size=batch_size)
            arr[e] = feats.reshape(T1, C, encoder.n_tokens, encoder.dim)
        arr.flush()
        del arr
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(info, f)
        return FeatureCache(path)


class DinoRepresentation(Representation):
    """Tokens = DINOv2 patch features, positions = patch grid coordinates."""

    name = "dino"
    pos_dim = 2
    residual = True

    def __init__(self, encoder: DinoEncoder, cache: Optional[FeatureCache] = None, feat_scale: float = 1.0):
        self.encoder = encoder
        self.cache = cache
        self.feat_dim = encoder.dim
        self.out_dim = encoder.dim
        self.grid_pos = encoder.grid_positions()
        self.feat_scale = feat_scale

    def tokens_from_cache(self, items: list[tuple[int, int, int]], device=None) -> TokenBatch:
        assert self.cache is not None
        feats = np.stack([self.cache.get(e, t, c) for (e, t, c) in items]).astype(np.float32)
        feat = torch.from_numpy(feats) * self.feat_scale
        pos = self.grid_pos.unsqueeze(0).expand(feat.shape[0], -1, -1).clone()
        tb = TokenBatch(feat, pos)
        return tb.to(device) if device is not None else tb

    @torch.no_grad()
    def tokens_from_rgb(self, rgb: np.ndarray, device=None) -> TokenBatch:
        """uint8 (B,H,W,3) -> TokenBatch (encoded on the fly)."""
        feat = torch.from_numpy(self.encoder.encode_uint8(rgb)).float() * self.feat_scale
        pos = self.grid_pos.unsqueeze(0).expand(feat.shape[0], -1, -1).clone()
        tb = TokenBatch(feat, pos)
        return tb.to(device) if device is not None else tb

    def apply_prediction(self, tokens: TokenBatch, out: torch.Tensor) -> TokenBatch:
        return tokens.with_feat(tokens.feat + out)

    def target(self, next_tokens: TokenBatch, tokens: TokenBatch) -> torch.Tensor:
        return next_tokens.feat - tokens.feat
