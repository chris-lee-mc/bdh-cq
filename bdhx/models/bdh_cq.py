"""BDH + recurrent latent reasoning (community BDH-CQ reconstruction).

FRAMEWORK_SPEC sections 3, 3.1 and 13; PAPER_IMPLEMENTATION_GAPS sections 1-3.
Labelled "BDH-CQ (community)" in every table: the latent transition function
is not public, so this module reproduces the community guess and treats the
alternatives of spec 3.1 as hypotheses about it.

The int stage of `BDHReasoningWrapper.forward` (`bdh_cq.py:512-535`) is
reimplemented here so that the update rule between latent steps can be
swapped. Mirrored lines, in order:

- `:513`   `assert exists(memories)` -- must ingest before latent reasoning.
- `:517`   `latent = memories.embeds[..., -1:, :]` -- the seed `H_0`.
- `:521-522` seed `all_block_outputs` with that latent, once per int stage.
- `:526`   loop `item` times.
- `:527-528` add the optional community `latent_step_embed`.
- `:530`   `self.bdh(latent, memories = memories, return_memory = True,
           return_logits = False, update_memory = update,
           all_block_outputs = ..., total_reasoning_iterations = ...)`.
- `:532`   `latent = memories.embeds` -- the state fed back.
- `:502`   `total_reasoning_iterations` is the sum of the int stages (here R).

Not mirrored: `:534-535` (the per-latent logits of the community loss). Those
belong to `training.loss: legacy`, which goes through the wrapper's own
`return_loss` path instead of this loop.

Deviations from spec 3.1, documented:

- `recurrence.input_injection` has no meaning here: the query conditions the
  latent loop through the Hebbian memory, not through an additive term, so
  `S` is always `None`. Injecting the query hidden additively would be a new
  mechanism, not a community one.
- `precision_state` casts the fed-back latent to the configured dtype and
  back to the parameter dtype, so the carried state is rounded while the
  community block still computes in its own dtype (the community block mixes
  the latent with fp32 weights and cannot consume a bf16 activation).
- `kind: attn_residual` is the community mechanism, `BDH(attn_residual=True,
  attn_residual_depth_bias_distance=1)` (`bdh_cq.py:94-146, 390-405`), not
  the generic `recurrence.AttnResidual`; the update rule is the community
  loop, with the residual applied inside `BDH.forward`.
"""

from __future__ import annotations

from bdh_cq.bdh_cq import Memory
from torch import Tensor

from bdhx.models.bdh import BDHAdapter
from bdhx.models.recurrence import RecurrenceRunner
from bdhx.registry import register_model
from bdhx.tasks.base import EpisodeBatch

DEFAULT_R_MAX = 8
COMMUNITY_KINDS = ("plain", "attn_residual")


def recurrence_param_count(width: int, rec_cfg, r_max: int) -> int:
    """Params added by the framework update rules (0 for the community kinds)."""
    runner = RecurrenceRunner(
        lambda h, s: h,
        kind=runner_kind(rec_cfg.kind),
        r_max=r_max,
        width=width,
        adapter_rank=rec_cfg.adapter_rank,
    )
    return sum(p.numel() for p in runner.parameters())


def runner_kind(kind: str) -> str:
    """`attn_residual` is handled by the community `BDH`, so the runner is plain."""
    return "plain" if kind in COMMUNITY_KINDS else kind


@register_model("bdh_cq")
class BDHCQModel(BDHAdapter):
    """BDH with an int stage of `reasoning_steps` latent iterations.

    `solve` ingests the query as a tensor stage, runs the latent loop, then
    decodes the answer greedily from the last latent (batched `generate`).
    `block_applications` is `reasoning_steps * depth`: one `BDH.forward` per
    latent step, `depth` block applications each.
    """

    def __init__(
        self,
        model_cfg,
        vocab_size: int,
        target_length: int = 1,
        *,
        r_max: int = DEFAULT_R_MAX,
        gate_extrapolation: str = "hold_last",
        **kwargs,
    ):
        rec = model_cfg.recurrence
        self.kind = rec.kind
        r_max = max(int(r_max), 1)
        self._rec_cfg, self._r_max, self._gate_extrapolation = rec, r_max, gate_extrapolation
        super().__init__(
            model_cfg,
            vocab_size,
            target_length,
            attn_residual=rec.kind == "attn_residual",
            attn_residual_depth_bias_distance=1 if rec.kind == "attn_residual" else 0,
            **kwargs,
        )
        if self.loss == "legacy" and self.kind not in COMMUNITY_KINDS:
            raise ValueError(
                f"training.loss 'legacy' runs the community latent loop; "
                f"recurrence.kind '{self.kind}' needs loss 'final_answer'"
            )
        self.recurrence = RecurrenceRunner(
            self._latent_block,
            kind=runner_kind(rec.kind),
            r_max=r_max,
            width=self.dim,
            adapter_rank=rec.adapter_rank,
            gate_extrapolation=gate_extrapolation,
        )
        self._mem: Memory | None = None
        self._all_block_outputs: list[Tensor] | None = None
        self._total_iters = 1

    def reset_context(self) -> None:
        """Drops the `Memory` and the latent-loop scratch (stale after an aborted solve)."""
        super().reset_context()
        self._mem = None
        self._all_block_outputs = None
        self._total_iters = 1

    def _extra_param_count(self, width: int) -> int:
        # latent_step_embed is a wrapper parameter (`bdh_cq.py:461`), so the width
        # solver has to count it too.
        step_embed = width if self._rec_cfg.step_embedding else 0
        return recurrence_param_count(width, self._rec_cfg, self._r_max) + step_embed

    # -- latent loop -------------------------------------------------------
    def _cast_state(self, latent: Tensor) -> Tensor:
        """`precision_state`: round the carried state, keep the compute dtype."""
        if self.state_dtype == latent.dtype:
            return latent
        return latent.to(self.state_dtype).to(latent.dtype)

    def _latent_block(self, latent: Tensor, _s: Tensor | None = None) -> Tensor:
        """One latent iteration: `bdh_cq.py:527-532` with the state cast added."""
        latent = self._cast_state(latent)
        if self.wrapper.latent_step_embed is not None:  # bdh_cq.py:527-528
            latent = latent + self.wrapper.latent_step_embed
        _, self._mem = self.bdh(  # bdh_cq.py:530
            latent,
            memories=self._mem,
            return_memory=True,
            return_logits=False,
            update_memory=self.update_latent_memory,
            all_block_outputs=self._all_block_outputs,
            total_reasoning_iterations=self._total_iters,
        )
        return self._mem.embeds  # bdh_cq.py:532

    def _reason(
        self, memories: Memory, reasoning_steps: int, collect_diagnostics: bool
    ) -> tuple[Memory, Tensor, dict[str, list]]:
        steps = int(reasoning_steps)
        if steps < 0:
            raise ValueError("reasoning_steps must be >= 0")
        if memories is None:  # bdh_cq.py:513
            raise ValueError("must ingest tokens before latent reasoning")
        seed = memories.embeds[..., -1:, :]  # bdh_cq.py:517
        if steps == 0:
            return memories, seed, RecurrenceRunner.empty_diagnostics()
        self._mem = memories
        self._total_iters = steps  # bdh_cq.py:502
        # bdh_cq.py:521-522 - seeded once, then mutated by BDH.forward (:391)
        self._all_block_outputs = [seed] if self.bdh.attn_residual is not None else None
        latent, diagnostics = self.recurrence(seed, None, steps, collect_diagnostics)
        memories = self._mem
        self._mem, self._all_block_outputs = None, None
        return memories, latent, diagnostics

    def _applications(self, reasoning_steps: int) -> int:
        return max(int(reasoning_steps), 0) * self.depth

    def _latent_passes(self, reasoning_steps: int) -> int:
        return max(int(reasoning_steps), 0)

    # -- legacy loss -------------------------------------------------------
    def _legacy_stages(self, batch: EpisodeBatch, reasoning_steps: int) -> tuple:
        """Stages for the community loss: context, query, R latent steps, answer."""
        return (
            batch.demonstrations.to(self.device),
            self._query_stage(batch.query),
            max(int(reasoning_steps), 0),
            batch.target.to(self.device),
        )
