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

    Two properties the generator must have for `depth` to isolate chaining, and
    both had to be fixed to get them (`RESULTS.md` section A1a):

    1. Solvable. The d pairs the query's chain passes through are always
       demonstrated. Without that the answer is derivable in only
       `(1 - (1 - 1/domain_size) ** n_examples_per_fn) ** d` of episodes --
       0.41 at depth 1 and 0.001 at depth 8 with 4 examples of a domain of 8 --
       and in the rest the target token appears nowhere in the episode, so no
       model can do better than guess over the whole 4096-symbol pool.
    2. Query-dependent. Demonstrating each function on the chain input plus
       unrelated distractors is solvable but degenerate: the distractors dead
       end, so the chain is the ONLY complete length-d path through the demo
       graph and the answer can be read off without looking at the query at
       all -- in 99.4 percent of depth-8 episodes, measured. Instead each
       function is demonstrated on the IMAGES of the previous function's
       demonstrated inputs, so the demo graph is `n_examples_per_fn` disjoint
       complete paths and the query selects which one. At the default
       `n_examples_per_fn = domain_size` the full bijection is shown, which
       also fixes the guessing floor at `1/domain_size` (0.125) for every
       depth rather than letting it drift with the number of reachable
       endpoints.
    """

    name = "compose"

    def __init__(
        self,
        n_examples_per_fn: int | None = None,
        domain_size: int = 8,
        guarantee_solvable: bool = True,
    ) -> None:
        # None means "the whole domain": every function fully demonstrated.
        self.n_examples_per_fn = domain_size if n_examples_per_fn is None else n_examples_per_fn
        self.domain_size = domain_size
        # False reproduces the pre-fix distribution of the A1 sweep.
        self.guarantee_solvable = guarantee_solvable

    def guess_floor(self, depth: int) -> float:
        """Exact match from guessing uniformly among the reachable endpoints.

        The null this task is scored against. It is NOT 1/vocab: the target is
        always the end of one of the `n_examples_per_fn` demonstrated paths, so
        a model that learns only "emit a path endpoint" already scores this.
        """
        if not self.guarantee_solvable:
            return 0.0
        return 1.0 / max(min(self.n_examples_per_fn, self.domain_size), 1)

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
        # `demoed` is the index set of domains[i] that f_i is demonstrated on.
        # It starts as the chain's input plus distractors, and every later
        # function is demonstrated on the IMAGES of the previous one's, so each
        # demonstrated edge continues and the graph is `n_examples` disjoint
        # complete paths rather than one chain among dead ends.
        demoed = self._demo_indices(rng, domain_size, n_examples, chain_idx[0])
        for i in range(d):
            if not self.guarantee_solvable and i:
                demoed = self._demo_indices(rng, domain_size, n_examples, chain_idx[i])
            for j in demoed:
                demonstrations.append(
                    (torch.tensor([int(domains[i][j])]), torch.tensor([int(fns[i][j])]))
                )
            if i + 1 < d:
                demoed = self._continue(demoed, domains[i + 1], fns[i])
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
        """Which inputs of the FIRST function to demonstrate.

        With the guarantee, the chain's own input is always among them and the
        rest are distinct; without it, the legacy draw is with replacement over
        every function independently and may miss the chain entirely.
        """
        if not self.guarantee_solvable:
            return rng.integers(0, domain_size, size=n_examples)
        others = np.array([j for j in range(domain_size) if j != chain])
        extra = rng.choice(others, size=n_examples - 1, replace=False)
        return np.concatenate(([chain], extra)).astype(int)

    @staticmethod
    def _continue(demoed: np.ndarray, next_domain: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """Indices in `next_domain` of the values `fn` maps `demoed` to.

        Carrying the demonstrated set forward this way is what makes the query
        matter: every demonstrated edge has a demonstrated successor, so there
        are as many complete paths as demonstrations per function and the model
        has to follow the one the query starts.
        """
        position = {int(v): j for j, v in enumerate(next_domain)}
        return np.array([position[int(fn[j])] for j in demoed], dtype=int)

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
