"""T9: nested transformations (TASK_SUITE_SPEC section 2, T9)."""

from __future__ import annotations

import numpy as np
import torch

from bdhx.registry import register_task
from bdhx.tasks.base import Episode, EpisodicTask
from bdhx.tasks.vocab import draw_symbols, reserved_token

REVERSE = reserved_token(4)
ROTATE_LEFT = reserved_token(5)
SWAP_FIRST_LAST = reserved_token(6)
DUPLICATE_FIRST = reserved_token(7)
DROP_LAST = reserved_token(8)
LBRACKET = reserved_token(9)
RBRACKET = reserved_token(10)

PRIMITIVE_TAGS = (REVERSE, ROTATE_LEFT, SWAP_FIRST_LAST, DUPLICATE_FIRST, DROP_LAST)


def _reverse(s: list[int]) -> list[int]:
    return s[::-1]


def _rotate_left(s: list[int]) -> list[int]:
    return s[1:] + s[:1] if s else s[:]


def _swap_first_last(s: list[int]) -> list[int]:
    if len(s) < 2:
        return s[:]
    t = s[:]
    t[0], t[-1] = t[-1], t[0]
    return t


def _duplicate_first(s: list[int]) -> list[int]:
    if not s:
        return s[:]
    return [s[0]] + s


def _drop_last(s: list[int]) -> list[int]:
    return s[:-1] if s else s[:]


PRIMITIVES = {
    REVERSE: _reverse,
    ROTATE_LEFT: _rotate_left,
    SWAP_FIRST_LAST: _swap_first_last,
    DUPLICATE_FIRST: _duplicate_first,
    DROP_LAST: _drop_last,
}


def apply_program(program: list[int], s: list[int]) -> list[int]:
    """Apply each tag in `program` in order (leftmost tag applied first)."""
    out = s
    for tag in program:
        out = PRIMITIVES[tag](out)
    return out


@register_task("nested")
class NestedTask(EpisodicTask):
    """Five primitive string transformations, composed into a depth-k nested program.

    Demonstrations show each primitive individually (as in compose). The query is a
    bracketed program `[LBRACKET] tag_1 ... tag_k [RBRACKET] input`; target is the
    program applied to the input, tags applied left-to-right.
    """

    name = "nested"

    def __init__(self, string_len: int = 4, n_examples_per_fn: int = 3) -> None:
        self.string_len = string_len
        self.n_examples_per_fn = n_examples_per_fn

    def train_difficulties(self) -> list[dict[str, int]]:
        return [{"depth": k} for k in (1, 2)]

    def eval_difficulties(self) -> dict[str, list[dict[str, int]]]:
        return {
            "interp": [{"depth": k} for k in (1, 2)],
            "mild": [{"depth": k} for k in (3, 4)],
            "strong": [{"depth": k} for k in (5, 6, 8)],
        }

    def sample(self, rng: np.random.Generator, difficulty: dict[str, int]) -> Episode:
        depth = int(difficulty["depth"])
        string_len = int(difficulty.get("string_len", self.string_len))
        n_examples = int(difficulty.get("n_examples_per_fn", self.n_examples_per_fn))

        demonstrations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for tag in PRIMITIVE_TAGS:
            for _ in range(n_examples):
                s = [int(t) for t in draw_symbols(rng, string_len)]
                out = PRIMITIVES[tag](s)
                demonstrations.append(
                    (torch.tensor([tag] + s), torch.tensor(out, dtype=torch.long))
                )
        rng.shuffle(demonstrations)

        program = [int(PRIMITIVE_TAGS[i]) for i in rng.integers(0, len(PRIMITIVE_TAGS), size=depth)]
        x = [int(t) for t in draw_symbols(rng, string_len)]
        target_list = apply_program(program, x)

        query = torch.tensor([LBRACKET] + program + [RBRACKET] + x)
        target = torch.tensor(target_list, dtype=torch.long)

        return Episode(
            demonstrations=demonstrations,
            query=query,
            target=target,
            difficulty={"depth": depth},
            split="train",
            episode_id=0,
            extras={"program": program, "input": x},
        )

    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        return self.base_score(prediction, episode)
