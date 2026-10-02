"""Seeding and RNG state capture (FRAMEWORK_SPEC section 8)."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed python/numpy/torch (and cuda) and return the resulting rng states."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return get_rng_states()


def get_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    # `torch.set_rng_state`/`set_rng_state_all` require a CPU ByteTensor. A
    # checkpoint loaded with `map_location="cuda"` (Trainer.load_checkpoint
    # maps the whole payload to the training device) hands us RNG-state
    # tensors already on the GPU, which torch rejects with the misleading
    # message "RNG state must be a torch.ByteTensor" -- force CPU here so
    # the map_location used for the rest of the checkpoint can't affect it.
    if states.get("python") is not None:
        random.setstate(tuple(states["python"]))
    if states.get("numpy") is not None:
        np.random.set_state(states["numpy"])
    if states.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(states["torch"], dtype=torch.uint8).cpu())
    cuda = states.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8).cpu() for s in cuda])


def task_rng(task_seed: int, split: str, index: int) -> np.random.Generator:
    """Deterministic per-episode generator: a pure function of the three inputs."""
    digest = hashlib.sha256(f"{task_seed}|{split}|{index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def episode_id(task_seed: int, split: str, index: int) -> int:
    """Deterministic episode id from (task_seed, split, index)."""
    digest = hashlib.sha256(f"{task_seed}|{split}|{index}".encode()).digest()
    return int.from_bytes(digest[:6], "little")
