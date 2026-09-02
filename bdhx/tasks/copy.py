"""T7: copy (TASK_SUITE_SPEC section 2, T7)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import SYMBOL_OFFSET, SYMBOL_POOL

DEMO_LEN_RANGE = (1, 4)


def _draw_string(rng: np.random.Generator, length: int) -> np.ndarray:
    """Random symbol string; repeats allowed (copy must handle non-distinct tokens)."""
    return rng.integers(0, SYMBOL_POOL, size=length) + SYMBOL_OFFSET


@register_task("copy")
class CopyTask(EpisodicTask):
    """Copy a random symbol string; demos are short, the query length varies."""

    name = "copy"

    def train_difficulties(self) -> list[dict]:
        return [{"length": length} for length in (2, 4, 6, 8)]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        return {
            "interp": self.train_difficulties(),
            "mild": [{"length": length} for length in (9, 16)],
            "strong": [{"length": length} for length in (17, 32, 64)],
        }

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        length = int(difficulty["length"])
        n_demos = int(rng.integers(2, 4))  # 2 or 3, per spec
        dlo, dhi = DEMO_LEN_RANGE
        demonstrations = []
        for _ in range(n_demos):
            dlen = int(rng.integers(dlo, dhi + 1))
            s = torch.from_numpy(_draw_string(rng, dlen)).clone()
            demonstrations.append((s, s.clone()))

        query = torch.from_numpy(_draw_string(rng, length)).clone()
        target = query.clone()

        return Episode(
            demonstrations=demonstrations,
            query=query,
            target=target,
            difficulty={"length": length},
            split="train",
            episode_id=0,
        )

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        scores = self.base_score(prediction, episode)
        pred = prediction.to(torch.long).flatten()
        tgt = episode.target.to(torch.long).flatten()
        first_error = tgt.numel()
        for i in range(tgt.numel()):
            if i >= pred.numel() or int(pred[i]) != int(tgt[i]):
                first_error = i
                break
        scores["first_error_position"] = float(first_error)
        return scores
