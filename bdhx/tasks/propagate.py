"""T6: propagation (TASK_SUITE_SPEC section 2, T6).

1-D grid: a source cell propagates its fill symbol rightward up to (but not
including) a wall cell. `dim=2` is accepted in the difficulty dict but not
yet implemented.
"""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import draw_symbols

N_DEMOS = 3
DEMO_DISTANCE_RANGE = (1, 3)


def _place_segments(rng: np.random.Generator, length: int, distance: int, n_sources: int):
    """Split `length` into n_sources equal blocks and place one segment (source at
    the block start, wall `distance + 1` cells later) inside each block."""
    seg_len = distance + 2  # source + `distance` fillable cells + wall
    block = length // n_sources
    if seg_len > block:
        raise ValueError(
            f"propagate: length={length} too small for n_sources={n_sources} at "
            f"distance={distance} (need >= {seg_len * n_sources})"
        )
    segments = []
    for b in range(n_sources):
        lo, hi = b * block, (b + 1) * block if b < n_sources - 1 else length
        start = int(rng.integers(lo, hi - seg_len + 1))
        segments.append((start, start + distance + 1))  # (source_pos, wall_pos)
    return segments


def _render(length, segments, empty_tok, fill_tok, wall_tok):
    grid = np.full(length, empty_tok, dtype=np.int64)
    target = grid.copy()
    for source_pos, wall_pos in segments:
        grid[source_pos] = fill_tok
        target[source_pos] = fill_tok
        if wall_pos < length:
            grid[wall_pos] = wall_tok
            target[wall_pos] = wall_tok
        target[source_pos + 1 : min(wall_pos, length)] = fill_tok
    return grid, target


@register_task("propagate")
class PropagateTask(EpisodicTask):
    """Source cell(s) fill rightward up to a wall; demos teach the rule at a small
    distance, the query is drawn at the requested (possibly much larger) distance."""

    name = "propagate"

    def train_difficulties(self) -> list[dict]:
        return [{"length": 12, "distance": d, "n_sources": 1, "dim": 1} for d in (1, 2, 4)]

    def eval_difficulties(self) -> dict[str, list[dict]]:
        return {
            "interp": self.train_difficulties(),
            "mild": [{"length": 12, "distance": d, "n_sources": 1, "dim": 1} for d in (6, 8)],
            "strong": [{"length": 48, "distance": d, "n_sources": 1, "dim": 1} for d in (12, 24)],
        }

    def sample(self, rng: np.random.Generator, difficulty: dict) -> Episode:
        if int(difficulty.get("dim", 1)) != 1:
            raise NotImplementedError("propagate: dim=2 is not yet implemented")
        length = int(difficulty["length"])
        distance = int(difficulty["distance"])
        n_sources = int(difficulty.get("n_sources", 1))

        empty_tok, fill_tok, wall_tok = (int(t) for t in draw_symbols(rng, 3))

        demonstrations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(N_DEMOS):
            dlo, dhi = DEMO_DISTANCE_RANGE
            dhi = min(dhi, max(dlo, (length // n_sources) - 2))
            ddist = int(rng.integers(dlo, dhi + 1))
            segs = _place_segments(rng, length, ddist, n_sources)
            g, t = _render(length, segs, empty_tok, fill_tok, wall_tok)
            demonstrations.append((torch.from_numpy(g).clone(), torch.from_numpy(t).clone()))

        segments = _place_segments(rng, length, distance, n_sources)
        query_grid, target_grid = _render(length, segments, empty_tok, fill_tok, wall_tok)

        return Episode(
            demonstrations=demonstrations,
            query=torch.from_numpy(query_grid).clone(),
            target=torch.from_numpy(target_grid).clone(),
            difficulty={"length": length, "distance": distance, "n_sources": n_sources, "dim": 1},
            split="train",
            episode_id=0,
            extras={"segments": segments},
        )

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        scores = self.base_score(prediction, episode)
        pred = prediction.to(torch.long).flatten()
        tgt = episode.target.to(torch.long).flatten()
        length = tgt.numel()
        if pred.numel() < length:
            pred = torch.cat([pred, torch.full((length - pred.numel(),), -1)])
        segments = episode.extras.get("segments", [])
        distances = []
        for source_pos, _wall_pos in segments:
            d, i = 0, source_pos + 1
            while i < length and int(pred[i]) == int(tgt[i]):
                d += 1
                i += 1
            distances.append(d)
        scores["cell_acc"] = scores["token_acc"]
        scores["max_correct_distance"] = float(np.mean(distances)) if distances else 0.0
        return scores
