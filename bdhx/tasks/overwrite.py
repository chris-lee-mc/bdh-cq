"""T2: binding overwrite (TASK_SUITE_SPEC section 2, T2)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import draw_symbols


@register_task("overwrite")
class OverwriteTask(EpisodicTask):
    """As binding, but `n_overwrites` keys get a later demonstration with a new value."""

    name = "overwrite"

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        n = int(difficulty["n_bindings"])
        n_overwrites = min(int(difficulty.get("n_overwrites", 0)), n)
        gap = int(difficulty.get("gap", 0))
        keys = draw_symbols(rng, n)
        values = draw_symbols(rng, n)
        order = list(rng.permutation(n))
        demos = [(int(keys[i]), int(values[i])) for i in order]

        ow_idx = rng.choice(n, size=n_overwrites, replace=False) if n_overwrites else []
        latest = {int(keys[i]): int(values[i]) for i in range(n)}
        stale = {}
        for idx in ow_idx:
            key = int(keys[idx])
            old_val = latest[key]
            new_val = old_val
            while new_val == old_val:
                new_val = int(draw_symbols(rng, 1)[0])
            stale[key] = old_val
            latest[key] = new_val
            orig_pos = order.index(int(idx))
            insert_pos = min(orig_pos + 1 + gap, len(demos))
            demos.insert(insert_pos, (key, new_val))
            order.insert(insert_pos, int(idx))

        if n_overwrites:
            query_key = int(keys[int(rng.choice(ow_idx))])
        else:
            query_key = int(keys[int(rng.integers(n))])
        target_val = latest[query_key]

        demo_tensors = [(torch.tensor([k]), torch.tensor([v])) for k, v in demos]
        extras = {}
        if query_key in stale:
            extras["stale_target"] = stale[query_key]
        return Episode(
            demonstrations=demo_tensors,
            query=torch.tensor([query_key]),
            target=torch.tensor([target_val]),
            difficulty=dict(difficulty),
            split="train",
            episode_id=0,
            extras=extras,
        )

    def train_difficulties(self) -> list[dict]:
        return [
            {"n_bindings": n, "n_overwrites": k, "gap": g}
            for n in (2, 4, 8)
            for k in (1, 2)
            for g in (0, 4)
            if k <= n
        ]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        return {
            "interp": self.train_difficulties(),
            "mild": [
                {"n_bindings": 8, "n_overwrites": 1, "gap": 8},
                {"n_bindings": 8, "n_overwrites": 2, "gap": 16},
            ],
            "strong": [
                {"n_bindings": 8, "n_overwrites": 8, "gap": 32},
                {"n_bindings": 16, "n_overwrites": 16, "gap": 32},
            ],
        }

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        result = self.base_score(prediction, episode)
        pred = prediction.to(torch.long).flatten()
        stale = episode.extras.get("stale_target")
        stale_hit = 0.0
        if stale is not None:
            stale_t = torch.tensor([int(stale)], dtype=torch.long)
            if pred.numel() == stale_t.numel() and torch.equal(pred, stale_t):
                stale_hit = 1.0
        other_hit = 1.0 if result["exact_match"] < 1.0 and stale_hit < 1.0 else 0.0
        result["stale_rate"] = stale_hit
        result["other_rate"] = other_hit
        return result
