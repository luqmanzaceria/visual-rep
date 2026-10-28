import numpy as np
import pytest

from vrep.data import generate_episode
from vrep.sim import PushEnv, SceneConfig


@pytest.fixture(scope="session")
def small_cfg():
    return SceneConfig(image_size=64)


@pytest.fixture(scope="session")
def small_env(small_cfg):
    return PushEnv(small_cfg)


@pytest.fixture(scope="session")
def push_episode(small_env):
    """One short 'push' episode at 64x64 (shared across tests)."""
    return generate_episode(small_env, seed=21, T=6, kind="push")


@pytest.fixture(scope="session")
def cameras(small_env):
    from vrep.data.dataset import Cameras

    R, t = small_env.all_extrinsics()
    return Cameras(small_env.K.astype(np.float32), R, t)
