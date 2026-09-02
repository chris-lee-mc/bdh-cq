"""T3: irrelevant-context robustness (TASK_SUITE_SPEC section 2, T3)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import draw_symbols


@register_task("distractors")
class DistractorsTask(EpisodicTask):
    """As binding, plus m distractor demos whose keys never appear in the query."""

    name = "distractors"

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        n = int(difficulty["n_bindings"])
        ratio = float(difficulty.get("distractor_ratio", 0))
        m = round(n * ratio)
        all_keys = draw_symbols(rng, n + m)
        keys, dist_keys = all_keys[:n], all_keys[n:]
        values = draw_symbols(rng, n)
        dist_values = draw_symbols(rng, m) if m else np.array([], dtype=np.int64)

        order = rng.permutation(n)
        demos = [(int(keys[i]), int(values[i])) for i in order]
        for i in range(m):
            pos = int(rng.integers(0, len(demos) + 1))
            demos.insert(pos, (int(dist_keys[i]), int(dist_values[i])))

        qi = int(rng.integers(n))
        query_key = int(keys[qi])
        target_val = int(values[qi])

        demo_tensors = [(torch.tensor([k]), torch.tensor([v])) for k, v in demos]
        return Episode(
            demonstrations=demo_tensors,
            query=torch.tensor([query_key]),
            target=torch.tensor([target_val]),
            difficulty=dict(difficulty),
            split="train",
            episode_id=0,
            extras={"distractor_values": [int(v) for v in dist_values]},
        )

    def train_difficulties(self) -> list[dict]:
        return [{"n_bindings": n, "distractor_ratio": r} for n in (2, 4, 8) for r in (0, 0.5, 1)]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        return {
            "interp": self.train_difficulties(),
            "mild": [{"n_bindings": 8, "distractor_ratio": 2}],
            "strong": [
                {"n_bindings": 8, "distractor_ratio": 4},
                {"n_bindings": 8, "distractor_ratio": 8},
            ],
        }

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        result = self.base_score(prediction, episode)
        pred = prediction.to(torch.long).flatten()
        dist_values = episode.extras.get("distractor_values", [])
        hit = 0.0
        for v in dist_values:
            dv = torch.tensor([int(v)], dtype=torch.long)
            if pred.numel() == dv.numel() and torch.equal(pred, dv):
                hit = 1.0
                break
        result["distractor_answer_rate"] = hit
        return result
