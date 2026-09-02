"""T1: arbitrary associative binding (TASK_SUITE_SPEC section 2, T1)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import SYMBOL_OFFSET, SYMBOL_POOL, draw_symbols


@register_task("binding")
class BindingTask(EpisodicTask):
    """Draw n key->value pairs; query one key, target is its value."""

    name = "binding"

    def __init__(self, allow_value_collisions: bool = False):
        self.allow_value_collisions = allow_value_collisions

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        n = int(difficulty["n_bindings"])
        keys = draw_symbols(rng, n)
        if self.allow_value_collisions:
            values = rng.integers(0, SYMBOL_POOL, size=n) + SYMBOL_OFFSET
        else:
            values = draw_symbols(rng, n)
        order = rng.permutation(n)
        demos = [(torch.tensor([int(keys[i])]), torch.tensor([int(values[i])])) for i in order]
        qi = int(rng.integers(n))
        query = torch.tensor([int(keys[qi])])
        target = torch.tensor([int(values[qi])])
        return Episode(
            demonstrations=demos,
            query=query,
            target=target,
            difficulty=dict(difficulty),
            split="train",
            episode_id=0,
        )

    def train_difficulties(self) -> list[dict]:
        return [{"n_bindings": n} for n in (1, 2, 4, 8)]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        return {
            "interp": [{"n_bindings": n} for n in (1, 2, 4, 8)],
            "mild": [{"n_bindings": 16}],
            "strong": [{"n_bindings": n} for n in (32, 64)],
        }

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        return self.base_score(prediction, episode)
