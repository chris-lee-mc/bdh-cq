import numpy as np
import torch

from bdhx.seeding import episode_id, get_rng_states, seed_everything, set_rng_states, task_rng


def test_state_roundtrip_reproduces_torch_rand():
    seed_everything(123)
    states = get_rng_states()
    a = torch.rand(4)
    na = np.random.rand(3)
    set_rng_states(states)
    b = torch.rand(4)
    nb = np.random.rand(3)
    assert torch.equal(a, b)
    assert np.allclose(na, nb)


def test_seed_everything_is_deterministic():
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    assert torch.equal(a, torch.rand(3))


def test_task_rng_pure_function():
    x = task_rng(1000, "train", 5).integers(0, 1_000_000, size=8)
    y = task_rng(1000, "train", 5).integers(0, 1_000_000, size=8)
    z = task_rng(1000, "mild", 5).integers(0, 1_000_000, size=8)
    assert np.array_equal(x, y)
    assert not np.array_equal(x, z)
    assert episode_id(1000, "train", 5) == episode_id(1000, "train", 5)
    assert episode_id(1000, "train", 5) != episode_id(1000, "train", 6)
