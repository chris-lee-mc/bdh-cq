"""Legacy adapters for the community bdh_cq.tasks ARC-style families
(propagation, copy, order, nesting) -- PAPER_IMPLEMENTATION_GAPS.md row on
"ARC-style task families".

Grids are 2-D numpy arrays over a fixed 10-color palette. We reuse the
community icq token scheme (row separator; colors keep their literal
values so the legacy adapter reproduces the exact community distribution,
not our fresh-symbol-per-episode hygiene) mapped into our own vocabulary:
colors -> SYMBOL_OFFSET + color, rows separated by a single reserved ROW
token. Our own serialize() already supplies the demo/query/answer
structure (MAP/SEP/QUERY/ANSWER/EOS), so no IN/OUT/EOS markers are needed.
"""

from __future__ import annotations

import numpy as np
import torch
from bdh_cq.tasks import Copy, Nesting, Order, Propagation

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import SYMBOL_OFFSET, reserved_token

ROW = reserved_token(11)

N_DEMOS = 2
GRID_SIZE = 4  # keeps grids small; passed as the community `size` kwarg


def _encode_grid(grid: np.ndarray) -> torch.Tensor:
    toks: list[int] = []
    for i, row in enumerate(grid):
        if i:
            toks.append(ROW)
        toks.extend(int(v) + SYMBOL_OFFSET for v in row)
    return torch.tensor(toks, dtype=torch.long)


class _LegacyTask(EpisodicTask):
    """Wraps one community `Task` subclass. demo_levels -> train, test_levels
    split in half -> mild (lower half) / strong (upper half)."""

    community_cls: type = None  # set by subclasses

    def __init__(self, size: int | None = GRID_SIZE):
        self._impl = self.community_cls(size=size)

    def train_difficulties(self) -> list[dict]:
        lo, hi = self._impl.demo_levels
        return [{"level": lv} for lv in range(lo, hi + 1)]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        _demo_lo, demo_hi = self._impl.demo_levels
        lo, hi = self._impl.test_levels
        lo = max(lo, demo_hi + 1)  # keep mild/strong disjoint from the train range
        lo = min(lo, hi)
        mid = lo + (hi - lo) // 2
        mild = list(range(lo, mid + 1))
        strong = list(range(mid + 1, hi + 1)) or [hi]
        return {
            "interp": self.train_difficulties(),
            "mild": [{"level": lv} for lv in mild],
            "strong": [{"level": lv} for lv in strong],
        }

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        level = int(difficulty["level"])
        params = self._impl.sample(rng)
        demo_lo, demo_hi = self._impl.demo_levels

        demonstrations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(N_DEMOS):
            dlevel = int(rng.integers(demo_lo, demo_hi + 1))
            inp, out = self._impl.render(rng, dlevel, params)
            demonstrations.append((_encode_grid(inp), _encode_grid(out)))

        q_inp, q_out = self._impl.render(rng, level, params)
        query = _encode_grid(q_inp)
        target = _encode_grid(q_out)

        return Episode(
            demonstrations=demonstrations,
            query=query,
            target=target,
            difficulty={"level": level},
            split="train",
            episode_id=0,
            extras={"grid_shape": tuple(q_out.shape)},
        )

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        return self.base_score(prediction, episode)


@register_task("legacy_propagation")
class LegacyPropagationTask(_LegacyTask):
    name = "legacy_propagation"
    community_cls = Propagation


@register_task("legacy_copy")
class LegacyCopyTask(_LegacyTask):
    name = "legacy_copy"
    community_cls = Copy


@register_task("legacy_order")
class LegacyOrderTask(_LegacyTask):
    name = "legacy_order"
    community_cls = Order


@register_task("legacy_nesting")
class LegacyNestingTask(_LegacyTask):
    name = "legacy_nesting"
    community_cls = Nesting
