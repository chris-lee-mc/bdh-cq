"""Loader for cached evaluation episodes written by tools/generate_tasks.py
(TASK_SUITE_SPEC section 4). Evaluation always uses fixed, cached episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bdhx.config import PROJECT_ROOT
from bdhx.registry import get_task
from bdhx.tasks.base import Episode, EpisodicTask


def shard_dir(task_name: str, task_seed: int, root: Path | None = None) -> Path:
    root = root or (PROJECT_ROOT / "data")
    return root / f"{task_name}_s{task_seed}"


def load_eval_episodes(
    task: str | EpisodicTask,
    task_seed: int,
    split: str,
    n: int,
    root: Path | None = None,
) -> list[Episode]:
    """Load the first `n` cached episodes for (task, task_seed, split)."""
    task_obj = get_task(task)() if isinstance(task, str) else task
    task_name = task_obj.name
    d = shard_dir(task_name, task_seed, root)
    shard_path = d / f"{split}.npz"
    if not shard_path.exists():
        raise FileNotFoundError(
            f"no cached shard at {shard_path}; run tools/generate_tasks.py --task "
            f"{task_name} --seed {task_seed} first"
        )
    with np.load(shard_path, allow_pickle=True) as npz:
        serialized = npz["serialized"]
        splits = npz["splits"]
        difficulties = npz["difficulties"]
        episode_ids = npz["episode_ids"]
    total = len(serialized)
    if n > total:
        raise ValueError(f"requested {n} episodes but shard {shard_path} only has {total}")
    episodes = []
    for i in range(n):
        tokens = np.asarray(serialized[i], dtype=np.int64)
        difficulty = json.loads(difficulties[i])
        ep = task_obj.parse_serialized(
            torch.from_numpy(tokens),
            difficulty=difficulty,
            split=str(splits[i]),
            episode_id=int(episode_ids[i]),
        )
        episodes.append(ep)
    return episodes
