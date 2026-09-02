"""Looped Transformer (EXPERIMENT_PLAN section 4 item 3).

One shared pre-norm block (RoPE + SwiGLU) applied `reasoning_steps` times
through `RecurrenceRunner`, with input injection of the embedded serialized
sequence at every step. `sandwich_norm` is a constructor option (the Ouro
reimplementation reports it as critical for recurrent stability).
"""

from __future__ import annotations

from torch import Tensor

from bdhx.models.recurrence import RecurrenceRunner
from bdhx.models.transformer import (
    SeqReasoner,
    TransformerBlock,
    block_stack_params,
    resolve_width,
)
from bdhx.registry import register_model

DEFAULT_R_MAX = 8


def recurrence_params(width: int, rec_cfg, r_max: int) -> int:
    runner = RecurrenceRunner(
        lambda h, s: h,
        kind=rec_cfg.kind,
        r_max=r_max,
        width=width,
        adapter_rank=rec_cfg.adapter_rank,
    )
    return sum(p.numel() for p in runner.parameters())


@register_model("looped_transformer")
class LoopedTransformer(SeqReasoner):
    """Shared-block recurrent Transformer; `reasoning_steps` is a runtime arg."""

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
        r_max = max(int(r_max), 1)
        width = resolve_width(
            model_cfg,
            lambda w: (
                block_stack_params(w, vocab_size, 1, sandwich_norm)
                + recurrence_params(w, rec, r_max)
            ),
        )
        super().__init__(vocab_size, width, target_length)
        self.block = TransformerBlock(width, sandwich_norm=sandwich_norm)
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
