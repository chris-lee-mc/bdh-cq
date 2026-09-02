"""Baseline BDH adapter over the community implementation.

FRAMEWORK_SPEC sections 3 and 13; PAPER_IMPLEMENTATION_GAPS sections 1-3.
Every architectural computation is the community code at commit `c246f890`;
this module only stages the episode and adds batched decoding.

Community functions called (all from `bdh_cq.bdh_cq`):

- `BDHReasoningWrapper.forward(tokens, memories=..., update_memory=...,
  return_memory=True)` (`bdh_cq.py:463-473`) for every tensor stage:
  demonstration ingestion, the query stage, teacher-forced answers and each
  decode step.
- `BDHReasoningWrapper.forward(..., return_loss=True)` (`bdh_cq.py:560-602`)
  for `training.loss: legacy`.
- `BDH.forward` (`bdh_cq.py:318-444`) through the wrapper, and directly from
  `bdhx.models.bdh_cq` for latent steps.
- `BDH.to_logits` and `BDH.token_embed` for the decode seed and the token
  feedback, exactly as `BDHReasoningWrapper.generate` (`bdh_cq.py:640-664`).
- `Memory` (`bdh_cq.py:16`), an immutable namedtuple; `reset_context()` drops
  it, which is the community reset (pass `memories=None`).

Deliberate deviations, all documented:

- `generate` is single sequence only (`bdh_cq.py:620`); `_decode` reproduces
  it for a batch at temperature 0 (greedy). `test_batched_generate_parity`
  pins batched output to per-sequence `generate`.
- The wrapper is never called with a trailing int stage: such a call returns
  stale logits (`PAPER_IMPLEMENTATION_GAPS` section 2 item 4). Latent stages
  are run by `bdhx.models.bdh_cq`, which always reseeds from `Memory.embeds`.
- `solve(query, R)` for plain BDH ignores `R > 1`: the baseline has no latent
  loop, so one pass over the query is all it does. `block_applications`
  reports `depth` (the shared block applied once per depth slot), independent
  of R; `bdh_cq` reports `R * depth`.
- No attention mask exists in the community block, so padded batch positions
  are ingested as PAD tokens (batch episodes of one difficulty to avoid it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from bdh_cq.bdh_cq import BDH, BDHBlock, BDHReasoningWrapper, Memory
from torch import Tensor, nn

from bdhx.models.base import FlopsReport, ReasoningModel, SolveOutput
from bdhx.models.param_budget import solve_width
from bdhx.models.recurrence import RecurrenceRunner
from bdhx.registry import register_model
from bdhx.tasks.base import EpisodeBatch
from bdhx.tasks.vocab import ANSWER, BOS, MAP, QUERY, SEP

DEFAULT_HEADS = 4
DEFAULT_NEURON_RATIO = 4  # dim_qk_heads = ratio * dim (figure7.py uses 4x)
MAX_ROTARY_DIM = 64  # community default (`bdh_cq.py:270`)

PRECISION_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def state_dtype(name: str) -> torch.dtype:
    if name not in PRECISION_DTYPES:
        raise ValueError(f"unknown precision_state '{name}'; expected {sorted(PRECISION_DTYPES)}")
    return PRECISION_DTYPES[name]


def rotary_dim_for(dim_qk: int) -> int:
    """Largest even rotary slice <= min(64, dim_qk) (`bdh_cq.py:282-283` asserts)."""
    return min(MAX_ROTARY_DIM, dim_qk) // 2 * 2


def bdh_param_count(
    dim: int,
    vocab_size: int,
    *,
    heads: int = DEFAULT_HEADS,
    neuron_ratio: int = DEFAULT_NEURON_RATIO,
    depth: int = 1,
    share_weights: bool = True,
    attn_residual: bool = False,
    attn_residual_depth_bias_distance: int = 0,
) -> int:
    """Analytic `sum(p.numel() for p in BDH(...).parameters())`.

    Avoids materialising a 4096-wide BDH inside the width solver. Verified
    against the community module by `test_param_report_matches_community`.
    """
    dim_qk_heads = neuron_ratio * dim
    blocks = 1 if share_weights else max(depth, 1)
    total = 2 * vocab_size * dim  # token_embed + to_logits
    total += 3 * dim * dim_qk_heads * blocks  # to_qk + proj_up + proj_out
    total += rotary_dim_for(dim_qk_heads // heads) // 2  # rope.freqs
    if attn_residual:
        total += dim + dim  # tied pseudo-query + key_rmsnorm
        total += max(attn_residual_depth_bias_distance, 0)
    return total


class UnsharedBDH(BDH):
    """`recurrence.share_weights: false`: `depth` distinct `BDHBlock`s.

    The community `BDH` reuses one block for every depth slot
    (`bdh_cq.py:299-305, 381`) and cannot express an unshared stack
    (PAPER_IMPLEMENTATION_GAPS section 2 item 1). `forward` below reproduces
    `BDH.forward` (`bdh_cq.py:318-444`) line for line, indexing
    `self.blocks[layer_index]` where the community indexes `self.block`.
    """

    def __init__(self, *, dim, num_tokens, depth=1, heads=4, dim_qk_heads=32768, **kwargs):
        super().__init__(
            dim=dim,
            num_tokens=num_tokens,
            depth=depth,
            heads=heads,
            dim_qk_heads=dim_qk_heads,
            **kwargs,
        )
        block_cls = kwargs.get("block_cls", BDHBlock)
        act = {k: kwargs[k] for k in ("qk_activation", "ff_activation") if k in kwargs}
        del self.block
        self.blocks = nn.ModuleList(
            [
                block_cls(dim, heads=heads, dim_queries_keys=dim_qk_heads // heads, **act)
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        tokens_or_ids,
        memories=None,
        return_memory=False,
        return_logits=True,
        update_memory=True,
        return_per_pass_hiddens=False,
        all_block_outputs=None,
        total_reasoning_iterations=1,
    ):
        device = tokens_or_ids.device
        attention_residual = self.attn_residual

        tokens = tokens_or_ids if tokens_or_ids.is_floating_point() else None
        if tokens is None:
            tokens = self.post_embed_norm(self.token_embed(tokens_or_ids))

        seq_len, depth = tokens.shape[-2], self.depth

        if attention_residual is not None:
            if torch.is_tensor(all_block_outputs):
                all_block_outputs = [all_block_outputs]
            if all_block_outputs is None:
                all_block_outputs = [tokens]

        tokens_seen = 0
        if memories is not None:
            tokens_seen, _, memories = memories

        seq = torch.arange(seq_len, device=device) + tokens_seen
        pos_emb = self.rope(seq) if self.rope is not None else None

        memories = iter(memories if memories is not None else (None,) * depth)
        next_memories = []
        per_pass_hiddens = []

        for layer_index in range(depth):
            prev_memory = next(memories, None)
            block = self.blocks[layer_index]
            block_out, layer_memory = block(
                tokens, memories=prev_memory, rotary_emb=pos_emb, return_memories=True
            )
            if attention_residual is not None:
                all_block_outputs.append(block_out)
                query_index = 0 if self.attn_residual_tied else layer_index
                tokens = attention_residual(
                    tokens,
                    all_block_outputs,
                    layer_index=query_index,
                    depth=depth,
                    total_reasoning_iterations=total_reasoning_iterations,
                )
            else:
                tokens = tokens + block_out
            if return_per_pass_hiddens:
                per_pass_hiddens.append(tokens)
            next_memories.append(
                block.combine_memories(layer_memory, prev_memory) if update_memory else prev_memory
            )

        tokens = self.post_norm(tokens)
        logits = self.to_logits(tokens) if return_logits else None

        returns = (logits,)
        if return_memory:
            returns += (Memory(tokens_seen + seq_len, tokens, next_memories),)
        if return_per_pass_hiddens:
            returns += (per_pass_hiddens,)
        return returns[0] if len(returns) == 1 else returns


def make_bdh(
    dim: int,
    vocab_size: int,
    *,
    heads: int = DEFAULT_HEADS,
    neuron_ratio: int = DEFAULT_NEURON_RATIO,
    depth: int = 1,
    share_weights: bool = True,
    attn_residual: bool = False,
    attn_residual_depth_bias_distance: int = 0,
) -> BDH:
    """Community `BDH` (or `UnsharedBDH`) with the framework's width policy."""
    dim_qk_heads = neuron_ratio * dim
    if dim_qk_heads % heads:
        raise ValueError(f"dim_qk_heads {dim_qk_heads} not divisible by heads {heads}")
    kwargs = {
        "dim": dim,
        "num_tokens": vocab_size,
        "depth": max(depth, 1),
        "heads": heads,
        "dim_qk_heads": dim_qk_heads,
        "rotary_dim": rotary_dim_for(dim_qk_heads // heads),
        "attn_residual": attn_residual,
        "attn_residual_depth_bias_distance": attn_residual_depth_bias_distance,
    }
    return BDH(**kwargs) if share_weights else UnsharedBDH(**kwargs)


class BDHAdapter(ReasoningModel):
    """Shared staging for the BDH family; `bdh_cq` adds the latent loop.

    Episode staging (one tensor stage per phase, `TASK_SUITE_SPEC` section 1):
    context `[BOS] i1 [MAP] o1 [SEP] ...`, query `[QUERY] q [ANSWER]`, then
    the answer. Latent reasoning sits between the query and the answer, so a
    latent step predicts the first answer token under the community loss
    (`bdh_cq.py:517-522, 543-550`).
    """

    requires_serialized = False

    def __init__(
        self,
        model_cfg,
        vocab_size: int,
        target_length: int = 1,
        *,
        heads: int = DEFAULT_HEADS,
        neuron_ratio: int = DEFAULT_NEURON_RATIO,
        loss: str = "final_answer",
        attn_residual: bool = False,
        attn_residual_depth_bias_distance: int = 0,
        extra_params: int = 0,
    ):
        super().__init__()
        rec, mem = model_cfg.recurrence, model_cfg.memory
        if mem.kind not in ("bdh", "none"):
            raise ValueError(f"memory.kind '{mem.kind}' is not a BDH memory; use models/{mem.kind}")
        if loss not in ("final_answer", "legacy"):
            raise ValueError(f"unknown loss '{loss}'")

        self.vocab_size = int(vocab_size)
        self.target_length = int(target_length)
        self.depth = max(int(model_cfg.depth), 1)
        self.heads = int(heads)
        self.neuron_ratio = int(neuron_ratio)
        self.share_weights = bool(rec.share_weights)
        self.loss = loss
        self.state_dtype = state_dtype(model_cfg.precision_state)
        # memory.kind: none freezes the Hebbian update everywhere (ablation).
        self.update_memory = mem.kind != "none"
        self.update_latent_memory = self.update_memory

        width = self._resolve_width(
            model_cfg, vocab_size, attn_residual, attn_residual_depth_bias_distance, extra_params
        )
        self.dim = width
        self.solved_width = width
        self.dim_qk = neuron_ratio * width // heads
        bdh = make_bdh(
            width,
            vocab_size,
            heads=heads,
            neuron_ratio=neuron_ratio,
            depth=self.depth,
            share_weights=self.share_weights,
            attn_residual=attn_residual,
            attn_residual_depth_bias_distance=attn_residual_depth_bias_distance,
        )
        # The wrapper is the only owner of the BDH module, so `state_dict()` holds
        # one copy of every weight; `self.bdh` is a view on it.
        # `latent_step_embed` is the community per-step vector (`bdh_cq.py:461`).
        self.wrapper = BDHReasoningWrapper(bdh, latent_step_embed=bool(rec.step_embedding))
        self._memories: Memory | None = None

    @property
    def bdh(self) -> BDH:
        return self.wrapper.bdh

    def _resolve_width(
        self, model_cfg, vocab_size, attn_residual, bias_distance, extra_params
    ) -> int:
        self._count_kwargs = {
            "vocab_size": int(vocab_size),
            "attn_residual": bool(attn_residual),
            "bias_distance": int(bias_distance),
            "extra_params": int(extra_params),
        }
        if model_cfg.width:
            return int(model_cfg.width)
        if not model_cfg.params_target:
            raise ValueError("model config needs either `width` or `params_target`")
        return solve_width(self._count_at, int(model_cfg.params_target), step=8)[0]

    def _count_at(self, width: int) -> int:
        """Analytic parameter count at `width`; what the width solver optimises.

        `param_report().total` of the built model must equal this at the solved
        width (`test_width_solver_target_includes_every_added_param`).
        """
        kw = self._count_kwargs
        return (
            bdh_param_count(
                width,
                kw["vocab_size"],
                heads=self.heads,
                neuron_ratio=self.neuron_ratio,
                depth=self.depth,
                share_weights=self.share_weights,
                attn_residual=kw["attn_residual"],
                attn_residual_depth_bias_distance=kw["bias_distance"],
            )
            + self._extra_param_count(width)
            + kw["extra_params"]
        )

    def _extra_param_count(self, width: int) -> int:
        """Params the subclass adds on top of the community module (0 here)."""
        return 0

    # -- blocks ------------------------------------------------------------
    @property
    def block_modules(self) -> list[nn.Module]:
        """Modules a `BlockCounter` hooks (one shared block, or `depth` of them)."""
        return [self.bdh.block] if self.share_weights else list(self.bdh.blocks)

    @property
    def device(self) -> torch.device:
        return self.bdh.token_embed.weight.device

    # -- context -----------------------------------------------------------
    def reset_context(self) -> None:
        """Drops the ingested `Memory`; the community reset is `memories=None`."""
        self._memories = None

    def ingest_context(self, demonstrations: list[tuple[Tensor, Tensor]]) -> None:
        """Feeds the serialized demonstrations through the wrapper as one tensor stage.

        Calls `BDHReasoningWrapper.forward(tokens, memories=None,
        update_memory=..., return_memory=True)` (`bdh_cq.py:463-473`, the
        binding point of FRAMEWORK_SPEC section 13) and stores the returned
        `Memory`.
        """
        self._memories = self._ingest(self._demo_tokens(demonstrations), None)

    def _ingest(self, tokens: Tensor, memories: Memory | None) -> Memory:
        _, memories = self.wrapper(
            tokens.to(self.device),
            memories=memories,
            update_memory=self.update_memory,
            return_memory=True,
        )
        return memories

    def _col(self, value: int, b: int) -> Tensor:
        return torch.full((b, 1), value, dtype=torch.long, device=self.device)

    def _demo_tokens(self, demonstrations: list[tuple[Tensor, Tensor]]) -> Tensor:
        """`[BOS] i1 [MAP] o1 [SEP] i2 ...`, same order as `tasks.base.demo_tokens`."""
        pairs = [
            (i if i.dim() == 2 else i.unsqueeze(0), o if o.dim() == 2 else o.unsqueeze(0))
            for i, o in demonstrations
        ]
        b = pairs[0][0].shape[0] if pairs else 1
        parts = [self._col(BOS, b)]
        for index, (inp, out) in enumerate(pairs):
            if index:
                parts.append(self._col(SEP, b))
            parts += [inp.to(self.device), self._col(MAP, b), out.to(self.device)]
        return torch.cat(parts, dim=1)

    def _query_stage(self, query: Tensor) -> Tensor:
        q = (query if query.dim() == 2 else query.unsqueeze(0)).to(self.device)
        return torch.cat([self._col(QUERY, q.shape[0]), q, self._col(ANSWER, q.shape[0])], dim=1)

    # -- reasoning ---------------------------------------------------------
    def _reason(
        self, memories: Memory, reasoning_steps: int, collect_diagnostics: bool
    ) -> tuple[Memory, Tensor, dict[str, list]]:
        """Baseline BDH has no latent loop; `reasoning_steps > 1` is ignored."""
        return memories, memories.embeds[..., -1:, :], RecurrenceRunner.empty_diagnostics()

    def _applications(self, reasoning_steps: int) -> int:
        return self.depth

    # -- decoding ----------------------------------------------------------
    def _decode(self, memories: Memory, seed: Tensor, num_tokens: int) -> tuple[Tensor, Tensor]:
        """Batched greedy decoding; the batched form of `generate` (`bdh_cq.py:604-680`).

        Mirrors `generate` at `temperature = 0`: seed logits from
        `BDH.to_logits` of the last latent (`:640-641`), argmax (`:650`), then
        feed `BDH.token_embed` of the token back as a float stage (`:664-671`)
        so `BDH.forward` skips the embedding norm exactly as it does there.
        The only change is that the argmax is taken over the batch instead of
        `.item()`, so no `stop_token` early exit exists; callers decode a fixed
        `num_tokens` and truncate.
        """
        logits = self.bdh.to_logits(seed)
        tokens, step_logits = [], []
        for index in range(int(num_tokens)):
            step = logits[:, -1]
            token = step.argmax(dim=-1)
            tokens.append(token)
            step_logits.append(step)
            if index + 1 == int(num_tokens):
                break
            embeds = self.bdh.token_embed(token.unsqueeze(1))
            logits, memories = self.wrapper(
                embeds,
                memories=memories,
                return_memory=True,
                update_memory=self.update_memory,
                update_latent_memory=self.update_latent_memory,
            )
        return torch.stack(tokens, dim=1), torch.stack(step_logits, dim=1)

    # -- interface ---------------------------------------------------------
    def solve(
        self,
        query: Tensor,
        reasoning_steps: int,
        collect_diagnostics: bool = False,
        target_length: int | None = None,
    ) -> SolveOutput:
        stage = self._query_stage(query)
        _, memories = self.wrapper(
            stage,
            memories=self._memories,
            update_memory=self.update_memory,
            return_memory=True,
        )
        memories, seed, diagnostics = self._reason(memories, reasoning_steps, collect_diagnostics)
        predictions, logits = self._decode(memories, seed, int(target_length or self.target_length))
        return SolveOutput(
            predictions=predictions,
            logits=logits,
            diagnostics=diagnostics,
            block_applications=self._applications(reasoning_steps),
        )

    def forward_episode(self, batch: EpisodeBatch, reasoning_steps: int) -> Tensor:
        """Teacher-forced logits over the target positions, (B, Lt, V).

        Position 0 is the projection of the state after the last latent stage
        (what `generate` seeds from); positions 1.. come from feeding the
        target back as a tensor stage, the community answer segment.
        """
        memories = self._ingest(batch.demonstrations.to(self.device), None)
        _, memories = self.wrapper(
            self._query_stage(batch.query),
            memories=memories,
            update_memory=self.update_memory,
            return_memory=True,
        )
        memories, seed, _ = self._reason(memories, reasoning_steps, False)
        logits = self.bdh.to_logits(seed)
        target = batch.target.to(self.device)
        if target.shape[1] > 1:
            answer_logits, _ = self.wrapper(
                target,
                memories=memories,
                update_memory=self.update_memory,
                return_memory=True,
            )
            logits = torch.cat([logits, answer_logits[:, :-1]], dim=1)
        return logits

    def episode_loss(self, batch: EpisodeBatch, reasoning_steps: int) -> Tensor:
        """`final_answer`: CE on the answer tokens only. `legacy`: the community path."""
        if self.loss == "legacy":
            return self._legacy_loss(batch, reasoning_steps)
        logits = self.forward_episode(batch, reasoning_steps)
        target = batch.target.to(self.device)
        mask = batch.target_mask.to(self.device)
        labels = target.masked_fill(~mask, -1)
        return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-1)

    def _legacy_stages(self, batch: EpisodeBatch, reasoning_steps: int) -> tuple:
        return (
            batch.demonstrations.to(self.device),
            self._query_stage(batch.query),
            batch.target.to(self.device),
        )

    def _legacy_loss(self, batch: EpisodeBatch, reasoning_steps: int) -> Tensor:
        """`BDHReasoningWrapper.forward(..., return_loss=True)` (`bdh_cq.py:560-602`).

        Latent positions predict the first token of the next segment and each
        answer position the next answer token; the stage list always ends on a
        tensor, so the `bdh_cq.py:572` assertion holds.
        """
        return self.wrapper(
            *self._legacy_stages(batch, reasoning_steps),
            memories=None,
            return_loss=True,
            update_memory=self.update_memory,
            update_latent_memory=self.update_latent_memory,
        )

    # -- flops -------------------------------------------------------------
    def _block_flops(self, n: int) -> dict[str, float]:
        d, h, q = self.dim, self.heads, self.dim_qk
        attention = 2 * n * d * (h * q) + 2 * h * n * n * q + 2 * h * n * n * d
        memory = 2 * 2 * h * n * q * d
        feedforward = 2 * h * n * d * q + 2 * n * (h * q) * d
        return {"attention": float(attention), "memory": float(memory), "ff": float(feedforward)}

    def _pass_flops(self, n: int, out: dict[str, float]) -> None:
        for key, value in self._block_flops(n).items():
            out[key] = out.get(key, 0.0) + value * self.depth

    def flops_estimate(self, batch: EpisodeBatch, reasoning_steps: int) -> FlopsReport:
        n_demo = batch.demonstrations.shape[1]
        n_query = batch.query.shape[1] + 2
        n_target = batch.target.shape[1]
        parts: dict[str, float] = {}
        self._pass_flops(n_demo, parts)
        self._pass_flops(n_query, parts)
        for _ in range(self._latent_passes(reasoning_steps) + max(n_target - 1, 0)):
            self._pass_flops(1, parts)
        parts["head"] = float(2 * (n_demo + n_query + n_target) * self.dim * self.vocab_size)
        per_episode = float(sum(parts.values()))
        return FlopsReport(total=per_episode * len(batch), per_episode=per_episode, breakdown=parts)

    def _latent_passes(self, reasoning_steps: int) -> int:
        return 0


@register_model("bdh")
class BDHModel(BDHAdapter):
    """Baseline BDH (PAPER_IMPLEMENTATION_GAPS section 3).

    The community `BDHBlock` with Hebbian memory, shared across `depth` (or
    `depth` distinct blocks when `recurrence.share_weights: false`), run
    without latent reasoning: `solve(query, R)` ignores `R > 1` and reports
    `depth` block applications.
    """

    def __init__(self, model_cfg, vocab_size: int, target_length: int = 1, **kwargs):
        if model_cfg.recurrence.kind != "plain":
            raise ValueError(
                f"model 'bdh' has no latent loop; recurrence.kind "
                f"'{model_cfg.recurrence.kind}' belongs to 'bdh_cq'"
            )
        super().__init__(model_cfg, vocab_size, target_length, **kwargs)
