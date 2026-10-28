import numpy as np
import pytest
import torch

from vrep.repr.base import TokenBatch
from vrep.repr.geometry import patch_pointmap, patch_pointmap_torch


def test_patch_pointmap_table_is_flat(push_episode, cameras, small_cfg):
    ep = push_episode
    cam = 2
    depth = ep.depth[0, cam].astype(np.float32)
    pm = patch_pointmap(depth, cameras.K, cameras.R[cam], cameras.t[cam], grid=8, max_depth=small_cfg.max_depth)
    assert pm.shape == (64, 3)
    seg = ep.seg[0, cam]
    # patches that are entirely table should have z ~ 0
    g = 8
    s = depth.shape[0] // g
    for i in range(g):
        for j in range(g):
            blk_seg = seg[i * s : (i + 1) * s, j * s : (j + 1) * s]
            blk_d = depth[i * s : (i + 1) * s, j * s : (j + 1) * s]
            if np.all(blk_seg == -1) and np.all(blk_d < 1.0):
                assert abs(pm[i * g + j, 2]) < 0.02, pm[i * g + j]


def test_patch_pointmap_torch_matches_numpy(push_episode, cameras, small_cfg):
    ep = push_episode
    cams = [0, 1]
    depth = torch.from_numpy(ep.depth[1, cams].astype(np.float32))
    K = torch.from_numpy(cameras.K).float()
    R = torch.from_numpy(cameras.R[cams]).float()
    t = torch.from_numpy(cameras.t[cams]).float()
    pm_t = patch_pointmap_torch(depth, K, R, t, grid=8, max_depth=small_cfg.max_depth)
    for i, c in enumerate(cams):
        pm_n = patch_pointmap(ep.depth[1, c].astype(np.float32), cameras.K, cameras.R[c], cameras.t[c], 8, small_cfg.max_depth)
        assert np.allclose(pm_t[i].numpy(), pm_n, atol=2e-3), np.abs(pm_t[i].numpy() - pm_n).max()


def test_token_batch_ops():
    tb = TokenBatch(torch.zeros(2, 5, 3), torch.ones(2, 5, 2), torch.ones(2, 5, dtype=torch.bool))
    assert tb.B == 2 and tb.N == 5 and tb.D == 3
    t2 = tb.with_feat(torch.ones(2, 5, 3))
    assert t2.feat.sum() == 30 and t2.pos is tb.pos
    assert tb[0:1].B == 1


@pytest.mark.slow
def test_dino_encoder_shapes():
    from vrep.repr.dino import DinoEncoder, DinoRepresentation

    enc = DinoEncoder("dinov2_vits14", input_size=112, device="cpu")
    assert enc.n_tokens == 64 and enc.dim == 384
    rgb = (np.random.default_rng(0).uniform(0, 255, size=(3, 64, 64, 3))).astype(np.uint8)
    feats = enc.encode_uint8(rgb)
    assert feats.shape == (3, 64, 384) and feats.dtype == np.float16
    rep = DinoRepresentation(enc)
    tb = rep.tokens_from_rgb(rgb)
    assert tb.feat.shape == (3, 64, 384) and tb.pos.shape == (3, 64, 2)
    assert tb.pos.min() > 0 and tb.pos.max() < 1
    # residual application
    nxt = rep.apply_prediction(tb, torch.ones_like(tb.feat))
    assert torch.allclose(rep.target(nxt, tb), torch.ones_like(tb.feat))


def test_pointmap_cache_matches_direct(tmp_path, small_cfg):
    from vrep.data import EpisodeStore, generate_dataset
    from vrep.repr.geometry import PointmapCache

    root = generate_dataset(tmp_path / "ds", n_episodes=2, T=2, cfg=small_cfg, workers=1)
    store = EpisodeStore(root)
    cache = PointmapCache.build(store, grid=8, max_depth=small_cfg.max_depth, path=tmp_path / "pm.npy")
    assert cache.arr.shape == (2, 3, small_cfg.n_cameras, 64, 3)
    cams = store.cameras
    direct = patch_pointmap(store[1].depth[2, 3].astype(np.float32), cams.K, cams.R[3], cams.t[3], 8, small_cfg.max_depth)
    assert np.allclose(cache.get(1, 2, 3), direct, atol=2e-3)
    # rebuilding reuses the file
    cache2 = PointmapCache.build(store, grid=8, max_depth=small_cfg.max_depth, path=tmp_path / "pm.npy")
    assert np.array_equal(cache2.arr, cache.arr)
