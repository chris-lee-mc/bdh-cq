"""Measure how much of a task is answerable at all, before any model is trained.

Two checks, both cheap and both purely properties of the generator:

1. Target in context. Symbol tokens are drawn fresh per episode, so a target
   SYMBOL that appears nowhere in the episode cannot be produced except by
   guessing over the whole symbol pool. The fraction of episodes whose target
   symbols are all in context is a hard upper bound on exact match for those
   episodes. Structural tokens (LT/GT and the like) are excluded: they are
   fixed, so a model can always emit them.

2. Oracle solvable. For tasks with a registered oracle, the fraction of
   episodes whose answer can actually be DERIVED from the demonstrations. This
   is stricter than (1): a target can be visible without being identifiable.

3. Guess floor, where the task defines one: the exact match a model scores by
   learning the shape of the answer and nothing else. Quoting `1/vocab` as the
   null is wrong whenever the target is drawn from a small in-context set --
   `compose` hands back one of 8 demonstrated path endpoints, so its null is
   0.125, not 0.00024. A result is only interesting above this line.

The `AT_CHANCE` flag (`bdhx/results/aggregate.py`) catches a run that learned
nothing. This catches the other half of the same problem: a task that could not
have been learned. `compose` shipped with an oracle-solvable rate of 0.41 at
depth 1 and 0.001 at depth 8, which capped every A1 number on it; see
`RESULTS.md` section A1.

    python tools/task_ceiling.py --task compose
    python tools/task_ceiling.py --all --min-train-solvable 0.9
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import numpy as np

import bdhx.tasks as _tasks  # noqa: F401  (registers the task names)
from bdhx.registry import get_task, list_tasks
from bdhx.tasks.base import Episode
from bdhx.tasks.vocab import is_symbol

DEFAULT_EPISODES = 500


def _compose_oracle(episode: Episode) -> int | None:
    """Walk the query forward through the demonstrated pairs; None if it breaks."""
    demo = {int(inp[0]): int(out[0]) for inp, out in episode.demonstrations}
    cur = int(episode.query[-1])
    for _ in range(int(episode.difficulty["depth"])):
        if cur not in demo:
            return None
        cur = demo[cur]
    return cur


def _binding_oracle(episode: Episode) -> int | None:
    """The value demonstrated for the queried key, taking the latest one."""
    key = int(episode.query[-1])
    answer = None
    for inp, out in episode.demonstrations:
        if int(inp[0]) == key:
            answer = int(out[0])
    return answer


# Oracles answer from the demonstrations ONLY -- never from `episode.extras`,
# which holds ground truth no model sees.
ORACLES: dict[str, Callable[[Episode], int | None]] = {
    "compose": _compose_oracle,
    "binding": _binding_oracle,
    "overwrite": _binding_oracle,
}


def episode_tokens(episode: Episode) -> set[int]:
    tokens = {int(t) for inp, out in episode.demonstrations for t in list(inp) + list(out)}
    tokens.update(int(t) for t in episode.query)
    return tokens


def measure(task, difficulty: dict, n: int, seed: int) -> dict[str, float | None]:
    rng = np.random.default_rng(seed)
    oracle = ORACLES.get(task.name)
    in_context = 0
    solvable = 0
    for _ in range(n):
        episode = task.sample(rng, difficulty)
        target = [int(t) for t in episode.target]
        tokens = episode_tokens(episode)
        in_context += all(t in tokens for t in target if is_symbol(t))
        if oracle is not None:
            solvable += oracle(episode) == target[0]
    floor = getattr(task, "guess_floor", None)
    return {
        "target_in_context": in_context / n,
        "oracle_solvable": (solvable / n) if oracle is not None else None,
        "guess_floor": floor(int(difficulty.get("depth", 0))) if callable(floor) else None,
    }


def difficulties_for(task) -> list[tuple[str, dict]]:
    rows = [("train", d) for d in task.train_difficulties()]
    for split, diffs in task.eval_difficulties().items():
        rows.extend((split, d) for d in diffs)
    return rows


def report(name: str, n: int, seed: int, min_train_solvable: float) -> bool:
    """Prints one task's table; returns False if a train difficulty is below the bar."""
    task = get_task(name)()
    has_oracle = name in ORACLES
    print(
        f"\n=== {name} ({n} episodes per difficulty, "
        f"{'oracle registered' if has_oracle else 'no oracle: in-context bound only'}) ==="
    )
    print(
        f"{'split':8s} {'difficulty':46s} {'target in ctx':>13s} "
        f"{'oracle solvable':>16s} {'guess floor':>12s}"
    )
    ok = True
    for split, difficulty in difficulties_for(task):
        stats = measure(task, difficulty, n, seed)
        solvable = stats["oracle_solvable"]
        shown = "-" if solvable is None else f"{solvable:.3f}"
        floor = stats["guess_floor"]
        floor_shown = "-" if floor is None else f"{floor:.3f}"
        print(
            f"{split:8s} {difficulty!s:46s} {stats['target_in_context']:13.3f} "
            f"{shown:>16s} {floor_shown:>12s}"
        )
        bound = stats["target_in_context"] if solvable is None else solvable
        if split == "train" and bound < min_train_solvable:
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--all", action="store_true", help="every registered task")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--min-train-solvable",
        type=float,
        default=0.0,
        help="exit non-zero if any train difficulty falls below this; use as a pre-sweep gate",
    )
    args = parser.parse_args(argv)

    names = list_tasks() if args.all else (args.task or ["compose"])
    failures = [
        name
        for name in names
        if not report(name, args.episodes, args.seed, args.min_train_solvable)
    ]
    if failures:
        print(
            f"\nFAIL: train difficulties below {args.min_train_solvable} for {', '.join(failures)}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
