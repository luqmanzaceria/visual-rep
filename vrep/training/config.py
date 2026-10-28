"""Experiment configuration (a dataclass, overridable from the command line)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    # what
    rep: str = "dino"  # dino | geo | gauss
    data: str = "data/push"
    run: str = "runs/debug"
    seed: int = 0
    device: Optional[str] = None  # None = auto (cuda > mps > cpu)
    max_episodes: Optional[int] = None  # subsample the dataset (debugging)
    frac_val: float = 0.1

    # encoder (dino / geo)
    dino_name: str = "dinov2_vits14"
    dino_input: int = 224
    feat_scale: float = 1.0
    point_scale: float = 3.0  # geo: world coords multiplied by this inside the feature vector
    pointmap_source: str = "oracle"  # geo: oracle (simulator depth) | mono (Depth Anything V2, table-plane calibrated)

    # gaussians
    n_gaussians: int = 768
    disp_scale: float = 0.05
    render_size: int = 64
    gauss_loss: str = "render+chamfer"  # any '+'-joined subset of {render, chamfer, oracle}
    w_rgb: float = 1.0
    w_depth: float = 10.0  # depth / chamfer are in metres: scale them up to matter next to the RGB term
    w_chamfer: float = 20.0
    w_oracle: float = 1.0
    chamfer_points: int = 2048
    chamfer_color_weight: float = 0.3
    chamfer_visibility: bool = False  # only pull Gaussians the camera can see toward observed points
    multiview_supervision: bool = False  # Gaussians: supervise the next frame from all training cameras (input stays one camera)

    # predictor
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_freq: int = 6
    n_memory: int = 0
    gated: bool = False
    gate_l1: float = 0.0  # locality penalty on gates
    dropout: float = 0.0

    # training
    steps: int = 6000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: int = 300
    horizon: int = 1  # multi-step rollout training horizon
    horizon_schedule: bool = True  # ramp the horizon from 1 to `horizon` over the first half of training
    grad_clip: float = 1.0
    val_every: int = 500
    val_batches: int = 8
    log_every: int = 50

    # probe
    probe_steps: int = 2500
    probe_lr: float = 3e-4
    probe_batch: int = 64
    probe_d_model: int = 192
    probe_layers: int = 2
    occ_grid: int = 8  # granular scenes: occupancy readout resolution

    # visualisation decoder (2D representations only)
    decoder_steps: int = 1500
    decoder_size: int = 64

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.run)


def load_config(argv: list[str] | None = None, base: TrainConfig | None = None) -> TrainConfig:
    """Build a config from defaults, optional ``config=<yaml>`` file and ``key=value`` overrides."""
    cli = OmegaConf.from_cli(argv)
    conf = OmegaConf.structured(base or TrainConfig())
    if "config" in cli:
        conf = OmegaConf.merge(conf, OmegaConf.load(cli.pop("config")))
    conf = OmegaConf.merge(conf, cli)
    return TrainConfig(**OmegaConf.to_container(conf, resolve=True))
