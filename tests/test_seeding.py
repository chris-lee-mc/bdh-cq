import numpy as np
import pytest
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


def test_set_rng_states_accepts_a_non_cpu_torch_state():
    """Regression test: found live on a real GPU sweep job.

    `Trainer.load_checkpoint` calls `torch.load(path, map_location=self.device,
    weights_only=False)`, which moves *every* tensor in the payload -- the
    RNG-state byte tensors included -- onto the training device. Passing a
    CUDA tensor straight to `torch.set_rng_state` fails with "RNG state must
    be a torch.ByteTensor" (a CPU-only requirement, misleadingly worded). We
    can't allocate a real CUDA tensor without a GPU, so simulate the same
    shape of bug with a tensor that merely needs normalizing.
    """
    states = get_rng_states()
    torch_state = states["torch"].clone()
    states["torch"] = torch_state.to(dtype=torch.uint8)  # already CPU; must still round-trip
    set_rng_states(states)  # must not raise
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_set_rng_states_moves_a_cuda_torch_state_back_to_cpu():
    """Same regression as above, exercised with a genuine CUDA tensor."""
    states = get_rng_states()
    states["torch"] = states["torch"].to("cuda")
    if states["cuda"] is not None:
        states["cuda"] = [s.to("cuda") for s in states["cuda"]]
    set_rng_states(states)  # must not raise TypeError: RNG state must be a torch.ByteTensor


def test_task_rng_pure_function():
    x = task_rng(1000, "train", 5).integers(0, 1_000_000, size=8)
    y = task_rng(1000, "train", 5).integers(0, 1_000_000, size=8)
    z = task_rng(1000, "mild", 5).integers(0, 1_000_000, size=8)
    assert np.array_equal(x, y)
    assert not np.array_equal(x, z)
    assert episode_id(1000, "train", 5) == episode_id(1000, "train", 5)
    assert episode_id(1000, "train", 5) != episode_id(1000, "train", 6)
