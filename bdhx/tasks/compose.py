"""T5: function composition (TASK_SUITE_SPEC section 2, T5)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import COMPOSE, draw_symbols


@register_task("compose")
class ComposeTask(EpisodicTask):
    """Compose d random bijections over disjoint domains; query applies all d in order."""

    name = "compose"

    def __init__(
        self,
        n_examples_per_fn: int = 4,
        domain_size: int = 8,
    ) -> None:
        self.n_examples_per_fn = n_examples_per_fn
        self.domain_size = domain_size

    def train_difficulties(self) -> list[dict[str, int]]:
        return [{"depth": d} for d in (1, 2)]

    def eval_difficulties(self) -> dict[str, list[dict[str, int]]]:
        return {
            "interp": [{"depth": d} for d in (1, 2)],
            "mild": [{"depth": d} for d in (3, 4)],
            "strong": [{"depth": d} for d in (6, 8)],
        }

    def sample(self, rng: np.random.Generator, difficulty: dict[str, int]) -> Episode:
        d = int(difficulty["depth"])
        n_examples = int(difficulty.get("n_examples_per_fn", self.n_examples_per_fn))
        domain_size = int(difficulty.get("domain_size", self.domain_size))

        # d+1 disjoint domains of size `domain_size`, each drawn fresh from the pool.
        total = draw_symbols(rng, domain_size * (d + 1))
        domains = [total[i * domain_size : (i + 1) * domain_size] for i in range(d + 1)]

        # f_i: domains[i] -> domains[i+1], a random bijection.
        fns: list[np.ndarray] = []
        for i in range(d):
            perm = rng.permutation(domain_size)
            fns.append(domains[i + 1][perm])  # fns[i][j] = f_i(domains[i][j])

        demonstrations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(d):
            idxs = rng.integers(0, domain_size, size=n_examples)
            for j in idxs:
                demonstrations.append(
                    (torch.tensor([int(domains[i][j])]), torch.tensor([int(fns[i][j])]))
                )
        rng.shuffle(demonstrations)  # order does not carry function identity

        x_idx = int(rng.integers(0, domain_size))
        x = int(domains[0][x_idx])
        cur_idx = x_idx
        intermediates = [x]
        for i in range(d):
            cur_val = fns[i][cur_idx]
            intermediates.append(int(cur_val))
            cur_idx = int(np.where(domains[i + 1] == cur_val)[0][0])
        target_val = intermediates[-1]

        query = torch.tensor([COMPOSE, d, x])
        target = torch.tensor([target_val])

        return Episode(
            demonstrations=demonstrations,
            query=query,
            target=target,
            difficulty={"depth": d},
            split="train",
            episode_id=0,
            extras={"intermediates": intermediates, "domains": domains, "fns": fns},
        )

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        scores = self.base_score(prediction, episode)
        scores["partial_depth_acc"] = self._partial_depth_acc(prediction, episode)
        return scores

    @staticmethod
    def _partial_depth_acc(prediction: torch.Tensor, episode: Episode) -> float:
        """Largest prefix depth d' such that intermediates[d'] equals the (1-token) answer.

        Since the target is a single token, this reduces to whether the final answer
        matches some intermediate value; report the fraction of the full depth reached.
        """
        intermediates = episode.extras.get("intermediates")
        depth = int(episode.difficulty.get("depth", 0))
        if not intermediates or depth == 0:
            return 0.0
        pred = prediction.to(torch.long).flatten()
        pred_val = int(pred[0].item()) if pred.numel() else -1
        best = 0
        for k, val in enumerate(intermediates):
            if val == pred_val:
                best = k
        return best / depth
