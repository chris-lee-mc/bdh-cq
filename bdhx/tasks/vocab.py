"""Structural tokens and the symbol pool (TASK_SUITE_SPEC section 1)."""

from __future__ import annotations

import numpy as np
import torch

PAD = 0
BOS = 1
EOS = 2
SEP = 3
MAP = 4
QUERY = 5
ANSWER = 6
COMPOSE = 7

# Tokens 8..31 are reserved for per-task role tokens (declare them here or via
# reserved_token(i) so no task collides with another).
RESERVED_START = 8
N_RESERVED = 32
SYMBOL_OFFSET = N_RESERVED
SYMBOL_POOL = 4096
VOCAB_SIZE = SYMBOL_OFFSET + SYMBOL_POOL

STRUCTURAL_TOKENS = {
    "PAD": PAD,
    "BOS": BOS,
    "EOS": EOS,
    "SEP": SEP,
    "MAP": MAP,
    "QUERY": QUERY,
    "ANSWER": ANSWER,
    "COMPOSE": COMPOSE,
}


def reserved_token(index: int) -> int:
    """Per-task role token; index 0..(N_RESERVED - RESERVED_START - 1)."""
    if not 0 <= index < N_RESERVED - RESERVED_START:
        raise ValueError(f"reserved token index out of range: {index}")
    return RESERVED_START + index


def is_symbol(token: int) -> bool:
    return SYMBOL_OFFSET <= int(token) < VOCAB_SIZE


def symbol_permutation(rng: np.random.Generator, pool: int = SYMBOL_POOL) -> np.ndarray:
    """Fresh per-episode permutation of the symbol pool (vocabulary hygiene)."""
    return rng.permutation(pool)


def draw_symbols(rng: np.random.Generator, n: int, pool: int = SYMBOL_POOL) -> np.ndarray:
    """n distinct symbol *tokens* drawn uniformly from the pool."""
    if n > pool:
        raise ValueError(f"cannot draw {n} distinct symbols from a pool of {pool}")
    return rng.choice(pool, size=n, replace=False) + SYMBOL_OFFSET


def apply_permutation(tokens: torch.Tensor, perm: np.ndarray) -> torch.Tensor:
    """Remap symbol tokens through `perm`; structural tokens pass through."""
    out = tokens.clone()
    mask = out >= SYMBOL_OFFSET
    idx = (out[mask] - SYMBOL_OFFSET).to(torch.long)
    out[mask] = torch.as_tensor(perm, dtype=out.dtype)[idx] + SYMBOL_OFFSET
    return out
