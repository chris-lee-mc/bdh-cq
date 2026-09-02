"""T4: contradictory demonstrations (TASK_SUITE_SPEC section 2, T4)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import draw_symbols

CONVENTIONS = ("recency", "majority")


@register_task("contradict")
class ContradictTask(EpisodicTask):
    """n keys; c of them get conflicting demos. Default convention: recency (last wins)."""

    name = "contradict"

    def __init__(self, convention: str = "recency"):
        if convention not in CONVENTIONS:
            raise ValueError(f"unknown convention: {convention}")
        self.convention = convention

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        n = int(difficulty["n_bindings"])
        c = min(int(difficulty.get("n_conflicts", 0)), n)
        convention = difficulty.get("convention", self.convention)
        keys = draw_symbols(rng, n)
        values = draw_symbols(rng, n)
        conflict_idx = set(rng.choice(n, size=c, replace=False).tolist()) if c else set()

        blocks: list[list[tuple[int, int]]] = []
        first_val: dict[int, int] = {}
        last_val: dict[int, int] = {}
        majority_val: dict[int, int] = {}
        for i in range(n):
            key, val = int(keys[i]), int(values[i])
            if i in conflict_idx:
                alt = int(draw_symbols(rng, 1)[0])
                while alt == val:
                    alt = int(draw_symbols(rng, 1)[0])
                if convention == "majority":
                    blocks.append([(key, val), (key, alt), (key, val)])
                    last_val[key] = val
                else:
                    blocks.append([(key, val), (key, alt)])
                    last_val[key] = alt
                first_val[key] = val
                majority_val[key] = val
            else:
                blocks.append([(key, val)])
                first_val[key] = val
                last_val[key] = val
                majority_val[key] = val

        block_order = rng.permutation(n)
        demos = [pair for bi in block_order for pair in blocks[bi]]

        qi = int(rng.integers(n))
        query_key = int(keys[qi])
        target_val = majority_val[query_key] if convention == "majority" else last_val[query_key]

        demo_tensors = [(torch.tensor([k]), torch.tensor([v])) for k, v in demos]
        return Episode(
            demonstrations=demo_tensors,
            query=torch.tensor([query_key]),
            target=torch.tensor([target_val]),
            difficulty=dict(difficulty),
            split="train",
            episode_id=0,
            extras={
                "first_value": first_val[query_key],
                "last_value": last_val[query_key],
            },
        )

    def train_difficulties(self) -> list[dict]:
        return [{"n_bindings": n, "n_conflicts": c} for n in (2, 4, 8) for c in (0, 1, 2) if c <= n]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        return {
            "interp": self.train_difficulties(),
            "mild": [
                {"n_bindings": 8, "n_conflicts": 3},
                {"n_bindings": 8, "n_conflicts": 4},
            ],
            "strong": [{"n_bindings": 8, "n_conflicts": 8}],
        }

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        result = self.base_score(prediction, episode)
        pred = prediction.to(torch.long).flatten()
        first_v = torch.tensor([int(episode.extras["first_value"])], dtype=torch.long)
        last_v = torch.tensor([int(episode.extras["last_value"])], dtype=torch.long)
        result["first_rate"] = float(pred.numel() == first_v.numel() and torch.equal(pred, first_v))
        result["last_rate"] = float(pred.numel() == last_v.numel() and torch.equal(pred, last_v))
        return result
