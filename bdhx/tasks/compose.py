"""T5: function composition (TASK_SUITE_SPEC section 2, T5)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import COMPOSE, draw_symbols


@register_task("compose")
class ComposeTask(EpisodicTask):
    """Compose d random bijections over disjoint domains; query applies all d in order.

    Every episode is solvable from its own demonstrations: the d pairs the
    query's chain passes through are always demonstrated (`guarantee_solvable`),
    and the remaining pairs of each function are drawn without replacement so no
    demonstration is wasted on a duplicate.  Without that guarantee the answer
    is derivable in only `(1 - (1 - 1/domain_size) ** n_examples_per_fn) ** d`
    of episodes -- 0.41 at depth 1 and 0.001 at depth 8 for the defaults -- and
    in the rest the target token does not appear in the episode at all, so no
    model can do better than guess over the whole symbol pool.  See
    `RESULTS.md` section A1 for the measurement that motivated the guarantee.
    """

    name = "compose"

    def __init__(
        self,
        n_examples_per_fn: int = 4,
        domain_size: int = 8,
        guarantee_solvable: bool = True,
    ) -> None:
        self.n_examples_per_fn = n_examples_per_fn
        self.domain_size = domain_size
        # False reproduces the pre-fix distribution of the A1 sweep.
        self.guarantee_solvable = guarantee_solvable

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
        domain_size = int(difficulty.get("domain_size", self.domain_size))
        n_examples = int(difficulty.get("n_examples_per_fn", self.n_examples_per_fn))
        n_examples = max(1, min(n_examples, domain_size))

        # d+1 disjoint domains of size `domain_size`, each drawn fresh from the pool.
        total = draw_symbols(rng, domain_size * (d + 1))
        domains = [total[i * domain_size : (i + 1) * domain_size] for i in range(d + 1)]

        # f_i: domains[i] -> domains[i+1], a random bijection.
        fns: list[np.ndarray] = []
        for i in range(d):
            perm = rng.permutation(domain_size)
            fns.append(domains[i + 1][perm])  # fns[i][j] = f_i(domains[i][j])

        # The query and its chain come first so the demonstrations can be made to
        # cover it.  `chain_idx[i]` is the index into domains[i] that hop i uses.
        x_idx = int(rng.integers(0, domain_size))
        cur_idx = x_idx
        chain_idx: list[int] = []
        intermediates = [int(domains[0][x_idx])]
        for i in range(d):
            chain_idx.append(cur_idx)
            cur_val = fns[i][cur_idx]
            intermediates.append(int(cur_val))
            cur_idx = int(np.where(domains[i + 1] == cur_val)[0][0])
        x, target_val = intermediates[0], intermediates[-1]

        demonstrations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(d):
            idxs = self._demo_indices(rng, domain_size, n_examples, chain_idx[i])
            for j in idxs:
                demonstrations.append(
                    (torch.tensor([int(domains[i][j])]), torch.tensor([int(fns[i][j])]))
                )
        rng.shuffle(demonstrations)  # order does not carry function identity

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

    def _demo_indices(
        self, rng: np.random.Generator, domain_size: int, n_examples: int, chain: int
    ) -> np.ndarray:
        """Which inputs of one function to demonstrate.

        With the guarantee, the chain's own input is always among them and the
        rest are distinct distractors; without it, the legacy draw is with
        replacement and may miss the chain entirely.
        """
        if not self.guarantee_solvable:
            return rng.integers(0, domain_size, size=n_examples)
        others = np.array([j for j in range(domain_size) if j != chain])
        extra = rng.choice(others, size=n_examples - 1, replace=False)
        return np.concatenate(([chain], extra)).astype(int)

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
