"""Common model interface (FRAMEWORK_SPEC section 3)."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from bdhx.tasks.base import EpisodeBatch


@dataclass
class SolveOutput:
    predictions: Tensor  # (B, Lt) argmax token ids
    logits: Tensor  # (B, Lt, V)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    block_applications: int = 0


@dataclass
class ParamReport:
    total: int
    trainable: int
    breakdown: dict[str, int] = field(default_factory=dict)
    serialized_bytes: int = 0


@dataclass
class FlopsReport:
    total: float
    per_episode: float
    breakdown: dict[str, float] = field(default_factory=dict)


class ReasoningModel(nn.Module, ABC):
    """Abstract base every model in `bdhx.models` implements.

    `reasoning_steps` is always a runtime argument; no model bakes it into
    __init__. `reset_context()` must zero every episodic buffer.
    """

    # Sequence-native baselines set this True: they consume the serialized
    # episode (TASK_SUITE_SPEC section 1) instead of structured demonstrations.
    requires_serialized: bool = False

    @abstractmethod
    def reset_context(self) -> None: ...

    @abstractmethod
    def ingest_context(self, demonstrations: list[tuple[Tensor, Tensor]]) -> None: ...

    @abstractmethod
    def solve(
        self, query: Tensor, reasoning_steps: int, collect_diagnostics: bool = False
    ) -> SolveOutput: ...

    @abstractmethod
    def forward_episode(self, batch: EpisodeBatch, reasoning_steps: int) -> Tensor:
        """Training path: returns logits for the target positions, (B, Lt, V)."""

    def param_report(self) -> ParamReport:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        breakdown: dict[str, int] = {}
        for name, mod in self.named_children():
            breakdown[name] = sum(p.numel() for p in mod.parameters())
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        return ParamReport(
            total=total,
            trainable=trainable,
            breakdown=breakdown,
            serialized_bytes=buf.getbuffer().nbytes,
        )

    @abstractmethod
    def flops_estimate(self, batch: EpisodeBatch, reasoning_steps: int) -> FlopsReport: ...


class BlockCounter:
    """Forward-hook counter over a set of modules.

    Used by `test_reasoning_steps_runtime`: solve(q, R) must apply the shared
    block exactly R times.

        with BlockCounter(model.block) as c:
            model.solve(q, 4)
        assert c.count == 4
    """

    def __init__(self, modules: nn.Module | Iterable[nn.Module]):
        self.modules = [modules] if isinstance(modules, nn.Module) else list(modules)
        self.count = 0
        self._handles: list[Any] = []

    def _hook(self, *_args, **_kwargs) -> None:
        self.count += 1

    def __enter__(self):
        self.count = 0
        self._handles = [m.register_forward_hook(self._hook) for m in self.modules]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
