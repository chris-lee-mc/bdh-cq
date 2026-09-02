"""Unified block family for the H3 2x2 (FRAMEWORK_SPEC section 3, H3).

One block shell (pre-norm, residual, SwiGLU feed-forward) whose context mixer
is selected by `model.memory.kind`:

- `kv`   -> the causal RoPE softmax attention of `transformer.py`
- `bdh`  -> the community `bdh_cq.BDHBlock` mixer (linear attention on the
            tokens as values plus its ReLU gated projection), imported from the
            pinned `bdh_cq` package; no reimplementation.

Recurrence depth is handled by the same `RecurrenceRunner` every other model
uses, so memory kind and recurrence are the only variables across the 2x2.
"""

from __future__ import annotations

from torch import Tensor, nn

from bdhx.models.looped_transformer import DEFAULT_R_MAX, recurrence_params
from bdhx.models.recurrence import RecurrenceRunner
from bdhx.models.transformer import (
    Attention,
    SeqReasoner,
    SwiGLU,
    ff_dim,
    pick_heads,
    resolve_width,
)
from bdhx.registry import register_model

MIXER_KINDS = ("kv", "bdh")


def _bdh_block(width: int, heads: int) -> nn.Module:
    try:
        from bdh_cq import BDHBlock
    except ImportError as exc:  # pragma: no cover - depends on the pinned package
        raise NotImplementedError(
            "memory.kind='bdh' needs the community bdh_cq package (BDHBlock); "
            "install the pinned git dependency or use memory.kind='kv'"
        ) from exc
    return BDHBlock(width, heads=heads, dim_queries_keys=max(width // heads, 1))


class UnifiedBlock(nn.Module):
    """Pre-norm block with a swappable context mixer."""

    def __init__(self, width: int, memory_kind: str, sandwich_norm: bool = False):
        super().__init__()
        if memory_kind not in MIXER_KINDS:
            raise NotImplementedError(
                f"unified_block supports memory.kind in {MIXER_KINDS}, got '{memory_kind}'"
            )
        heads = pick_heads(width)
        self.memory_kind = memory_kind
        self.norm1 = nn.RMSNorm(width)
        self.mixer = Attention(width, heads) if memory_kind == "kv" else _bdh_block(width, heads)
        self.norm2 = nn.RMSNorm(width)
        self.ff = SwiGLU(width, ff_dim(width))
        self.sandwich_norm = sandwich_norm
        if sandwich_norm:
            self.post1 = nn.RMSNorm(width)
            self.post2 = nn.RMSNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        h = self.mixer(self.norm1(x))
        x = x + (self.post1(h) if self.sandwich_norm else h)
        h = self.ff(self.norm2(x))
        return x + (self.post2(h) if self.sandwich_norm else h)


def unified_params(width: int, vocab_size: int, memory_kind: str, sandwich_norm: bool) -> int:
    blk = UnifiedBlock(width, memory_kind, sandwich_norm)
    return sum(p.numel() for p in blk.parameters()) + vocab_size * width + width


@register_model("unified_block")
class UnifiedBlockModel(SeqReasoner):
    """Shared unified block run for `reasoning_steps` steps (H3 2x2 cells)."""

    def __init__(
        self,
        model_cfg,
        vocab_size: int,
        target_length: int = 1,
        r_max: int = DEFAULT_R_MAX,
        sandwich_norm: bool = False,
        gate_extrapolation: str = "hold_last",
    ):
        rec = model_cfg.recurrence
        kind = model_cfg.memory.kind
        r_max = max(int(r_max), 1)
        width = resolve_width(
            model_cfg,
            lambda w: (
                unified_params(w, vocab_size, kind, sandwich_norm)
                + recurrence_params(w, rec, r_max)
            ),
        )
        super().__init__(vocab_size, width, target_length)
        self.block = UnifiedBlock(width, kind, sandwich_norm)
        self.input_injection = bool(rec.input_injection)
        self.recurrence = RecurrenceRunner(
            self._apply_block,
            kind=rec.kind,
            r_max=r_max,
            width=width,
            adapter_rank=rec.adapter_rank,
            gate_extrapolation=gate_extrapolation,
        )
        self.solved_width = width

    def _apply_block(self, h: Tensor, s: Tensor | None) -> Tensor:
        return self.block(h + s if (self.input_injection and s is not None) else h)

    def _applications(self, reasoning_steps: int) -> int:
        return max(int(reasoning_steps), 0)

    def _run(
        self, tokens: Tensor, reasoning_steps: int, collect_diagnostics: bool
    ) -> tuple[Tensor, dict[str, list]]:
        s = self.embed(tokens.to(self.embed.weight.device))
        return self.recurrence(s, s, reasoning_steps, collect_diagnostics)
