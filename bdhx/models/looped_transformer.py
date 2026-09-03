"""Looped Transformer (EXPERIMENT_PLAN section 4 item 3).

One shared block applied `reasoning_steps` times through `RecurrenceRunner`,
with input injection of the embedded serialized sequence at every step.
`sandwich_norm` is a constructor option (the Ouro reimplementation reports it
as critical for recurrent stability).

`model.depth` counts the pre-norm layers *inside* the shared block
(FRAMEWORK_SPEC section 2): the loop repeats a `depth`-layer stack, so a step
at depth 1 is a one-layer network and cannot do induction-style copying. The
optional `prelude` and `coda` layers run once before and after the loop
(Huginn's prelude/coda; both 0 by default, so the default model is loop-only).
"""

from __future__ import annotations

from torch import Tensor, nn

from bdhx.models.recurrence import RecurrenceRunner
from bdhx.models.transformer import (
    SeqReasoner,
    SharedStack,
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
        prelude: int = 0,
        coda: int = 0,
    ):
        rec = model_cfg.recurrence
        r_max = max(int(r_max), 1)
        depth = max(int(model_cfg.depth), 1)
        prelude, coda = max(int(prelude), 0), max(int(coda), 0)
        n_layers = depth + prelude + coda
        width = resolve_width(
            model_cfg,
            lambda w: (
                block_stack_params(w, vocab_size, n_layers, sandwich_norm)
                + recurrence_params(w, rec, r_max)
            ),
        )
        super().__init__(vocab_size, width, target_length)
        self.depth = depth
        self.block = SharedStack(width, depth, sandwich_norm=sandwich_norm)
        self.prelude = nn.ModuleList(
            [TransformerBlock(width, sandwich_norm=sandwich_norm) for _ in range(prelude)]
        )
        self.coda = nn.ModuleList(
            [TransformerBlock(width, sandwich_norm=sandwich_norm) for _ in range(coda)]
        )
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
        """Layer applications: `depth` per reasoning step, plus prelude and coda."""
        return max(int(reasoning_steps), 0) * self.depth + len(self.prelude) + len(self.coda)

    def _run(
        self, tokens: Tensor, reasoning_steps: int, collect_diagnostics: bool
    ) -> tuple[Tensor, dict[str, list]]:
        s = self.embed_tokens(tokens)
        h = s
        for layer in self.prelude:
            h = layer(h)
        h, diag = self.recurrence(h, s, reasoning_steps, collect_diagnostics)
        for layer in self.coda:
            h = layer(h)
        return h, diag
