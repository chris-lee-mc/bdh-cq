"""Episode / EpisodicTask / batching (TASK_SUITE_SPEC section 1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from bdhx.tasks.vocab import ANSWER, BOS, EOS, MAP, PAD, QUERY, SEP, VOCAB_SIZE

SPLITS = ("train", "interp", "mild", "strong")


@dataclass(frozen=True)
class Episode:
    demonstrations: list[tuple[torch.Tensor, torch.Tensor]]
    query: torch.Tensor
    target: torch.Tensor
    difficulty: dict[str, int]
    split: str
    episode_id: int
    extras: dict = field(default_factory=dict, compare=False)


@dataclass
class EpisodeBatch:
    """Padded tensors for a list of episodes. Batch dim carries independent episodes."""

    demonstrations: torch.Tensor  # (B, Ld) flat demo sequence, BOS ... last demo
    demonstrations_mask: torch.Tensor  # (B, Ld) bool
    query: torch.Tensor  # (B, Lq)
    query_mask: torch.Tensor
    target: torch.Tensor  # (B, Lt)
    target_mask: torch.Tensor
    serialized: torch.Tensor  # (B, Ls) full serialized episode
    serialized_mask: torch.Tensor
    answer_start: torch.Tensor  # (B,) index of the first target token in `serialized`
    splits: list[str]
    difficulties: list[dict[str, int]]
    episode_ids: torch.Tensor  # (B,)

    def __len__(self) -> int:
        return self.query.shape[0]

    def to(self, device) -> EpisodeBatch:
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return EpisodeBatch(
            *[mv(getattr(self, f)) for f in ("demonstrations", "demonstrations_mask")],
            *[mv(getattr(self, f)) for f in ("query", "query_mask", "target", "target_mask")],
            *[mv(getattr(self, f)) for f in ("serialized", "serialized_mask", "answer_start")],
            self.splits,
            self.difficulties,
            mv(self.episode_ids),
        )


class EpisodicTask(ABC):
    """Base class for every task. Generators are pure functions of (rng, difficulty)."""

    name: str = "task"
    vocab_size: int = VOCAB_SIZE

    @abstractmethod
    def sample(self, rng: np.random.Generator, difficulty: dict[str, int]) -> Episode: ...

    @abstractmethod
    def train_difficulties(self) -> list[dict[str, int]]: ...

    @abstractmethod
    def eval_difficulties(self) -> dict[str, list[dict[str, int]]]:
        """Returns {"interp": [...], "mild": [...], "strong": [...]}"""

    @abstractmethod
    def score(self, prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        """Returns at least {"exact_match": 0/1, "token_acc": float}"""

    # -- serialization -----------------------------------------------------
    def serialize(self, episode: Episode, include_target: bool = True) -> torch.Tensor:
        """[BOS] i1 [MAP] o1 [SEP] i2 [MAP] o2 [QUERY] q [ANSWER] t [EOS]."""
        toks: list[int] = [BOS]
        for i, (inp, out) in enumerate(episode.demonstrations):
            if i:
                toks.append(SEP)
            toks.extend(int(t) for t in inp.tolist())
            toks.append(MAP)
            toks.extend(int(t) for t in out.tolist())
        toks.append(QUERY)
        toks.extend(int(t) for t in episode.query.tolist())
        toks.append(ANSWER)
        if include_target:
            toks.extend(int(t) for t in episode.target.tolist())
            toks.append(EOS)
        return torch.tensor(toks, dtype=torch.long)

    def parse_serialized(
        self,
        tokens: torch.Tensor,
        difficulty: dict[str, int] | None = None,
        split: str = "train",
        episode_id: int = 0,
    ) -> Episode:
        """Inverse of `serialize`; the two forms carry exactly the same information."""
        seq = [int(t) for t in tokens.tolist()]
        if seq and seq[0] == BOS:
            seq = seq[1:]
        if seq and seq[-1] == EOS:
            seq = seq[:-1]
        if ANSWER not in seq or QUERY not in seq:
            raise ValueError("serialized episode lacks QUERY/ANSWER markers")
        qi, ai = seq.index(QUERY), seq.index(ANSWER)
        demo_part, query_part, target_part = seq[:qi], seq[qi + 1 : ai], seq[ai + 1 :]
        demos: list[tuple[torch.Tensor, torch.Tensor]] = []
        if demo_part:
            for chunk in _split_on(demo_part, SEP):
                if MAP not in chunk:
                    raise ValueError("demonstration lacks a MAP token")
                m = chunk.index(MAP)
                demos.append(
                    (
                        torch.tensor(chunk[:m], dtype=torch.long),
                        torch.tensor(chunk[m + 1 :], dtype=torch.long),
                    )
                )
        return Episode(
            demonstrations=demos,
            query=torch.tensor(query_part, dtype=torch.long),
            target=torch.tensor(target_part, dtype=torch.long),
            difficulty=dict(difficulty or {}),
            split=split,
            episode_id=episode_id,
        )

    # -- default metric helper --------------------------------------------
    @staticmethod
    def base_score(prediction: torch.Tensor, episode: Episode) -> dict[str, float]:
        tgt = episode.target.to(torch.long).flatten()
        pred = prediction.to(torch.long).flatten()[: tgt.numel()]
        if pred.numel() < tgt.numel():
            pred = torch.cat([pred, torch.full((tgt.numel() - pred.numel(),), PAD)])
        correct = (pred == tgt).float()
        return {
            "exact_match": float(correct.min().item()) if tgt.numel() else 0.0,
            "token_acc": float(correct.mean().item()) if tgt.numel() else 0.0,
        }


def _split_on(seq: list[int], token: int) -> list[list[int]]:
    out: list[list[int]] = [[]]
    for t in seq:
        if t == token:
            out.append([])
        else:
            out[-1].append(t)
    return out


def _pad(rows: list[list[int]], pad: int = PAD) -> tuple[torch.Tensor, torch.Tensor]:
    n = max((len(r) for r in rows), default=0)
    n = max(n, 1)
    data = torch.full((len(rows), n), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.bool)
    for i, r in enumerate(rows):
        if r:
            data[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = True
    return data, mask


def demo_tokens(episode: Episode) -> list[int]:
    """Flat demonstration sequence: BOS i1 MAP o1 SEP i2 MAP o2 (no QUERY)."""
    toks: list[int] = [BOS]
    for i, (inp, out) in enumerate(episode.demonstrations):
        if i:
            toks.append(SEP)
        toks.extend(int(t) for t in inp.tolist())
        toks.append(MAP)
        toks.extend(int(t) for t in out.tolist())
    return toks


def pad_and_batch(episodes: Sequence[Episode], task: EpisodicTask | None = None) -> EpisodeBatch:
    """Pad a list of episodes into an EpisodeBatch (PAD = 0, masks are True on real tokens)."""
    if not episodes:
        raise ValueError("pad_and_batch got an empty episode list")
    serializer = task.serialize if task is not None else _default_serialize
    demos = [demo_tokens(e) for e in episodes]
    queries = [[int(t) for t in e.query.tolist()] for e in episodes]
    targets = [[int(t) for t in e.target.tolist()] for e in episodes]
    ser = [[int(t) for t in serializer(e).tolist()] for e in episodes]
    answer_start = [s.index(ANSWER) + 1 for s in ser]
    d, dm = _pad(demos)
    q, qm = _pad(queries)
    t, tm = _pad(targets)
    s, sm = _pad(ser)
    return EpisodeBatch(
        demonstrations=d,
        demonstrations_mask=dm,
        query=q,
        query_mask=qm,
        target=t,
        target_mask=tm,
        serialized=s,
        serialized_mask=sm,
        answer_start=torch.tensor(answer_start, dtype=torch.long),
        splits=[e.split for e in episodes],
        difficulties=[dict(e.difficulty) for e in episodes],
        episode_ids=torch.tensor([e.episode_id for e in episodes], dtype=torch.long),
    )


class _Serializer(EpisodicTask):
    name = "_serializer"

    def sample(self, rng, difficulty):  # pragma: no cover - helper only
        raise NotImplementedError

    def train_difficulties(self):  # pragma: no cover
        raise NotImplementedError

    def eval_difficulties(self):  # pragma: no cover
        raise NotImplementedError

    def score(self, prediction, episode):  # pragma: no cover
        return self.base_score(prediction, episode)


_default_serialize = _Serializer().serialize
