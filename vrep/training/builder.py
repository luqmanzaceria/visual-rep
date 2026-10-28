"""Assemble a representation + predictor + probe from a config (and reload them from a run)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vrep.data.dataset import EpisodeStore
from vrep.models import ObjectProbe, Predictor, PredictorConfig, TokenDecoder
from vrep.models.probe import GridProbe
from vrep.repr.base import TokenBatch
from vrep.repr.dino import DinoEncoder, DinoRepresentation, FeatureCache
from vrep.repr.gaussians import GaussianRepresentation
from vrep.repr.geometry import GeometryRepresentation, PointmapCache, build_mono_pointmap_cache
from vrep.training.config import TrainConfig
from vrep.training.samplers import Sampler2D, SamplerGauss
from vrep.utils import get_device


@dataclass
class Experiment:
    cfg: TrainConfig
    store: EpisodeStore
    rep: object
    predictor: Predictor
    probe: ObjectProbe
    probe_heldout: Optional[ObjectProbe]
    device: torch.device
    train_eps: list[int]
    val_eps: list[int]
    step: int = 0
    decoder: Optional[TokenDecoder] = None

    @property
    def is_gauss(self) -> bool:
        return self.rep.name == "gauss"

    def sampler(self, horizon: int, seed: int = 0, need_oracle: bool = False):
        if self.is_gauss:
            return SamplerGauss(
                self.rep, self.store, horizon, self.device, self.cfg.render_size, seed=seed,
                chamfer_points=self.cfg.chamfer_points if "chamfer" in self.cfg.gauss_loss else 0,
                need_oracle=need_oracle or ("oracle" in self.cfg.gauss_loss),
                supervision_cams=list(self.store.train_cams) if self.cfg.multiview_supervision else None,
            )
        return Sampler2D(self.rep, self.store, horizon, self.device)

    def encode_items(self, items: list[tuple[int, int, int]], rng: Optional[np.random.Generator] = None) -> TokenBatch:
        """Ground-truth tokens for (episode, t, cam) triples."""
        if self.is_gauss:
            rng = rng or np.random.default_rng(0)
            scene = self.rep.scene_from_store(self.store, items, rng, self.device)
            return self.rep.tokens(scene)
        if self.rep.name == "dino":
            return self.rep.tokens_from_cache(items, self.device)
        return self.rep.tokens_from_cache(self.store, items, self.device)

    def scene_items(self, items, rng=None):
        assert self.is_gauss
        rng = rng or np.random.default_rng(0)
        return self.rep.scene_from_store(self.store, items, rng, self.device)

    @property
    def K(self) -> torch.Tensor:
        return torch.from_numpy(self.store.cameras.K).float().to(self.device)

    def cam_RT(self, cams: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        R = torch.from_numpy(np.stack([self.store.cameras.R[c] for c in cams])).float().to(self.device)
        t = torch.from_numpy(np.stack([self.store.cameras.t[c] for c in cams])).float().to(self.device)
        return R, t

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (self.cfg.run_dir / "checkpoint.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.to_dict(),
                "predictor": self.predictor.state_dict(),
                "probe": self.probe.state_dict(),
                "probe_heldout": None if self.probe_heldout is None else self.probe_heldout.state_dict(),
                "decoder": None if self.decoder is None else self.decoder.state_dict(),
                "step": self.step,
            },
            path,
        )
        with open(self.cfg.run_dir / "config.json", "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        return path


def _predictor_dims(cfg: TrainConfig, rep, store: EpisodeStore) -> dict:
    th = store.scene.table_half
    if rep.name == "dino":
        return dict(feat_dim=rep.feat_dim, pos_dim=2, out_dim=rep.out_dim, pos_range=1.0)
    if rep.name == "geo":
        return dict(feat_dim=rep.feat_dim, pos_dim=3, out_dim=rep.out_dim, pos_range=cfg.point_scale * th)
    return dict(feat_dim=rep.feat_dim, pos_dim=3, out_dim=rep.out_dim, pos_range=th)


def build_representation(cfg: TrainConfig, store: EpisodeStore, device: torch.device, build_cache: bool = True):
    scene = store.scene
    if cfg.rep in ("dino", "geo"):
        enc = DinoEncoder(cfg.dino_name, cfg.dino_input, device)
        cache = None
        if build_cache:
            cache_path = store.root / "features" / f"{cfg.dino_name}_{cfg.dino_input}.npy"
            cache_path.parent.mkdir(exist_ok=True)
            cache = FeatureCache.build(store, enc, cache_path)
        dino = DinoRepresentation(enc, cache, feat_scale=cfg.feat_scale)
        if cfg.rep == "dino":
            return dino
        pm_cache = None
        if build_cache:
            if cfg.pointmap_source == "oracle":
                pm_cache = PointmapCache.build(store, enc.grid, scene.max_depth, store.root / "features" / f"pointmap_{enc.grid}.npy")
            elif cfg.pointmap_source == "mono":
                pm_cache = build_mono_pointmap_cache(store, enc.grid, store.root / "features" / f"pointmap_{enc.grid}_mono.npy")
            else:
                raise ValueError(cfg.pointmap_source)
        rep = GeometryRepresentation(dino, store.cameras, scene.max_depth, point_scale=cfg.point_scale,
                                     pointmap_source=cfg.pointmap_source, pointmap_cache=pm_cache)
        rep.table_half = scene.table_half
        return rep
    if cfg.rep == "gauss":
        return GaussianRepresentation(store.cameras, scene.table_half, scene.max_depth, n_gaussians=cfg.n_gaussians, disp_scale=cfg.disp_scale)
    raise ValueError(cfg.rep)


def build_experiment(cfg: TrainConfig, store: Optional[EpisodeStore] = None, build_cache: bool = True) -> Experiment:
    device = get_device(cfg.device)
    store = store or EpisodeStore(cfg.data, max_episodes=cfg.max_episodes)
    rep = build_representation(cfg, store, device, build_cache)
    dims = _predictor_dims(cfg, rep, store)
    pcfg = PredictorConfig(
        **dims, action_dim=2, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, n_freq=cfg.n_freq,
        action_scale=store.scene.max_action, n_memory=cfg.n_memory, gated=cfg.gated, dropout=cfg.dropout,
    )
    predictor = Predictor(pcfg).to(device)
    if store.is_granular:
        probe_kwargs = dict(feat_dim=dims["feat_dim"], pos_dim=dims["pos_dim"], grid=cfg.occ_grid, d_model=cfg.probe_d_model,
                            n_layers=cfg.probe_layers, pos_range=dims["pos_range"])
        probe_cls = GridProbe
    else:
        probe_kwargs = dict(
            feat_dim=dims["feat_dim"], pos_dim=dims["pos_dim"], n_objects=store.n_objects, d_model=cfg.probe_d_model,
            n_layers=cfg.probe_layers, pos_range=dims["pos_range"], out_scale=store.scene.table_half,
        )
        probe_cls = ObjectProbe
    probe = probe_cls(**probe_kwargs).to(device)
    probe_heldout = probe_cls(**probe_kwargs).to(device) if store.test_cams else None
    train_eps, val_eps = store.split(cfg.frac_val, seed=cfg.seed)
    decoder = None
    if cfg.rep in ("dino", "geo") and cfg.decoder_steps > 0:
        decoder = TokenDecoder(dims["feat_dim"], grid=rep.encoder.grid if cfg.rep == "dino" else rep.grid, out_size=cfg.decoder_size)  # stays on CPU: conv training diverges on MPS (torch 2.4)
    return Experiment(cfg, store, rep, predictor, probe, probe_heldout, device, train_eps, val_eps, decoder=decoder)


def load_experiment(run: str | Path, device: Optional[str] = None, store: Optional[EpisodeStore] = None, data: Optional[str] = None) -> Experiment:
    run = Path(run)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["cfg"])
    if device is not None:
        cfg.device = device
    if data is not None:
        cfg.data = data
    exp = build_experiment(cfg, store)
    exp.predictor.load_state_dict(ckpt["predictor"])
    exp.probe.load_state_dict(ckpt["probe"])
    if exp.probe_heldout is not None and ckpt.get("probe_heldout") is not None:
        exp.probe_heldout.load_state_dict(ckpt["probe_heldout"])
    if exp.decoder is not None and ckpt.get("decoder") is not None:
        exp.decoder.load_state_dict(ckpt["decoder"])
        exp.decoder.eval()
    exp.step = ckpt.get("step", 0)
    exp.predictor.eval()
    exp.probe.eval()
    if exp.probe_heldout is not None:
        exp.probe_heldout.eval()
    return exp
