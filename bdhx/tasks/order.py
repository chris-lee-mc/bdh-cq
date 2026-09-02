"""T8: ordering (TASK_SUITE_SPEC section 2, T8)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import draw_symbols, reserved_token

LT = reserved_token(0)
GT = reserved_token(1)
PAIR_TAG = reserved_token(2)
SORT_TAG = reserved_token(3)


@register_task("order")
class OrderTask(EpisodicTask):
    """A random total order over n symbols, given by a chain of adjacent comparisons.

    Query is either a pair (x, y) separated by `hops` steps in the chain (target:
    LT/GT token), or a "sort" of the full item set (target: items in ascending order).
    """

    name = "order"

    def train_difficulties(self) -> list[dict[str, int]]:
        out = []
        for h in (1, 2):
            out.append({"n_items": 6, "query_type": "pair", "hops": h})
        out.append({"n_items": 6, "query_type": "sort", "hops": 1})
        return out

    def eval_difficulties(self) -> dict[str, list[dict[str, int]]]:
        return {
            "interp": self.train_difficulties(),
            "mild": [
                {"n_items": 8, "query_type": "pair", "hops": 3},
                {"n_items": 8, "query_type": "pair", "hops": 4},
                {"n_items": 8, "query_type": "sort", "hops": 3},
            ],
            "strong": [
                {"n_items": 14, "query_type": "pair", "hops": 6},
                {"n_items": 14, "query_type": "pair", "hops": 12},
                {"n_items": 14, "query_type": "sort", "hops": 6},
            ],
        }

    def sample(self, rng: np.random.Generator, difficulty: dict[str, int]) -> Episode:
        n_items = int(difficulty["n_items"])
        query_type = difficulty.get("query_type", "pair")
        hops = int(difficulty.get("hops", 1))

        # items[i] < items[i+1] for all i: the total order is the array's own order.
        items = draw_symbols(rng, n_items)

        demonstrations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(n_items - 1):
            demonstrations.append(
                (torch.tensor([int(items[i]), int(items[i + 1])]), torch.tensor([LT]))
            )
        rng.shuffle(demonstrations)

        if query_type == "pair":
            hops = min(hops, n_items - 1)
            i = int(rng.integers(0, n_items - hops))
            j = i + hops
            if bool(rng.integers(0, 2)):
                x, y = int(items[i]), int(items[j])
                target_tok = LT
            else:
                x, y = int(items[j]), int(items[i])
                target_tok = GT
            query = torch.tensor([PAIR_TAG, x, y])
            target = torch.tensor([target_tok])
        elif query_type == "sort":
            shuffled = items.copy()
            rng.shuffle(shuffled)
            query = torch.tensor([SORT_TAG] + [int(t) for t in shuffled])
            target = torch.tensor([int(t) for t in items])
        else:
            raise ValueError(f"unknown query_type: {query_type}")

        return Episode(
            demonstrations=demonstrations,
            query=query,
            target=target,
            difficulty={"n_items": n_items, "query_type": query_type, "hops": hops},
            split="train",
            episode_id=0,
        )

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        return self.base_score(prediction, episode)
