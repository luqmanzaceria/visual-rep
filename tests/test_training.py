"""End-to-end smoke tests: tiny dataset -> train each representation for a few steps -> save/load -> rollout."""

import numpy as np
import pytest
import torch

from vrep.data import EpisodeStore, generate_dataset
from vrep.sim.scene import SceneConfig
from vrep.training import TrainConfig, load_experiment, train
from vrep.training.rollout import rollout_scene, rollout_tokens
from vrep.training.trainer import horizon_at, lr_at


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "tiny"
    generate_dataset(root, n_episodes=6, T=4, cfg=SceneConfig(image_size=56), workers=1)
    return root


def _cfg(tiny_data, tmp_path, rep, **kw):
    base = dict(
        rep=rep, data=str(tiny_data), run=str(tmp_path / f"run_{rep}"), device="cpu", steps=4, batch_size=3, warmup=1,
        val_every=2, val_batches=1, log_every=2, probe_steps=3, probe_batch=4, d_model=32, n_layers=1, n_heads=2,
        probe_d_model=32, probe_layers=1, dino_input=56, n_gaussians=64, render_size=28, chamfer_points=128, frac_val=0.34,
    )
    base.update(kw)
    return TrainConfig(**base)


def test_schedules():
    cfg = TrainConfig(steps=100, warmup=10, lr=1e-3, horizon=3)
    assert lr_at(0, cfg) < lr_at(9, cfg) <= 1e-3
    assert lr_at(99, cfg) < lr_at(10, cfg)
    assert horizon_at(0, cfg) == 1 and horizon_at(99, cfg) == 3


@pytest.mark.slow
@pytest.mark.parametrize("rep", ["dino", "geo", "gauss"])
def test_train_each_representation(tiny_data, tmp_path, rep):
    cfg = _cfg(tiny_data, tmp_path, rep, horizon=2)
    exp = train(cfg)
    assert (cfg.run_dir / "checkpoint.pt").exists() and (cfg.run_dir / "log.jsonl").exists()
    exp2 = load_experiment(cfg.run_dir, device="cpu")
    assert exp2.step == cfg.steps
    # a rollout from ground-truth tokens produces well-formed predictions
    items = [(exp2.val_eps[0], 0, 0), (exp2.val_eps[0], 1, 1)]
    actions = torch.zeros(2, 2, 2)
    if rep == "gauss":
        scene = exp2.scene_items(items)
        scenes, _ = rollout_scene(exp2.predictor, exp2.rep, scene, actions)
        assert len(scenes) == 2 and scenes[-1].mu.shape == scene.mu.shape
        assert torch.isfinite(scenes[-1].mu).all()
    else:
        tokens = exp2.encode_items(items)
        preds, _ = rollout_tokens(exp2.predictor, exp2.rep, tokens, actions)
        assert preds[-1].feat.shape == tokens.feat.shape and torch.isfinite(preds[-1].feat).all()
    xy, vis = exp2.probe(exp2.encode_items(items))
    assert xy.shape == (2, 3, 2) and vis.shape == (2, 3)


@pytest.mark.slow
def test_train_memory_and_gate(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, tmp_path, "dino", n_memory=2, gated=True, gate_l1=0.01, horizon=2, run=str(tmp_path / "run_mem"))
    exp = train(cfg)
    items = [(exp.val_eps[0], 0, 0)]
    tokens = exp.encode_items(items)
    preds, outs = rollout_tokens(exp.predictor, exp.rep, tokens, torch.zeros(1, 2, 2))
    assert outs[0].memory.shape[1] == 2 and outs[0].gate is not None


@pytest.mark.slow
def test_gauss_oracle_loss(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, tmp_path, "gauss", gauss_loss="oracle", run=str(tmp_path / "run_oracle"))
    exp = train(cfg)
    assert exp.step == cfg.steps
