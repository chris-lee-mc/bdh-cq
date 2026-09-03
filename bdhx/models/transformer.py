"""Fixed-depth Transformer baseline (EXPERIMENT_PLAN section 4 item 5).

Pre-norm block with RoPE attention and a SwiGLU feed-forward, distinct
weights per layer, R fixed at 1. Also hosts the shared sequence-model
machinery reused by `looped_transformer.py` and `unified_block.py`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bdhx.models.base import FlopsReport, ReasoningModel, SolveOutput
from bdhx.models.param_budget import solve_width
from bdhx.registry import register_model
from bdhx.tasks.base import EpisodeBatch
from bdhx.tasks.vocab import ANSWER, BOS, PAD, QUERY

EMBED_INIT_STD = 0.02


def pick_heads(width: int) -> int:
    """Largest head count in {8,4,2,1} giving an even head dim (RoPE needs pairs)."""
    for h in (8, 4, 2, 1):
        if width % h == 0 and (width // h) % 2 == 0:
            return h
    return 1


def ff_dim(width: int) -> int:
    """SwiGLU hidden size: 8/3 * width rounded to a multiple of 8."""
    return max(8, round(width * 8 / 3 / 8) * 8)


def rope_tables(seq_len: int, head_dim: int, device, dtype) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, L, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    """Causal multi-head softmax attention with RoPE."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        cos, sin = rope_tables(n, self.head_dim, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Pre-norm block; `sandwich_norm` adds the post-sublayer norms (Ouro)."""

    def __init__(self, width: int, heads: int | None = None, sandwich_norm: bool = False):
        super().__init__()
        heads = heads or pick_heads(width)
        self.norm1 = nn.RMSNorm(width)
        self.mixer = Attention(width, heads)
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


class SharedStack(nn.Module):
    """`depth` pre-norm blocks applied in order: the unit a looped model repeats.

    FRAMEWORK_SPEC section 2: for a looped model `model.depth` counts the layers
    *inside* the shared block, not the number of distinct blocks. One layer per
    step cannot express induction-style copying (that needs two attention
    layers), which is why the looped default is 2.
    """

    def __init__(
        self, width: int, depth: int, heads: int | None = None, sandwich_norm: bool = False
    ):
        super().__init__()
        self.depth = max(int(depth), 1)
        self.layers = nn.ModuleList(
            [TransformerBlock(width, heads, sandwich_norm) for _ in range(self.depth)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SeqReasoner(ReasoningModel):
    """Sequence-native ReasoningModel: consumes the serialized episode.

    Subclasses implement `_run(tokens, reasoning_steps, collect_diagnostics)`
    returning `(hidden (B, L, D), diagnostics)` and `_applications(R)`.
    The unembedding is tied to the token embedding (no separate parameter, so
    `param_report().total` equals the state_dict numel sum). Because the head is
    tied, the embedding init sets the logit scale: `nn.Embedding`'s default
    N(0, 1) makes the init logits O(sqrt(width)) and the init cross-entropy tens
    of nats instead of ln(vocab). The embedding is therefore initialized at
    `EMBED_INIT_STD`, the usual transformer value, and `embed_tokens` passes it
    through a parameter-free RMSNorm (the community BDH's `post_embed_norm`,
    `bdh_cq.py:288`) so that the residual stream starts at unit RMS whatever the
    init std. Without it a 0.02 embedding makes content-based key matching start
    from attention logits of order 1e-3 and the match circuit never forms.
    """

    requires_serialized = True

    def __init__(self, vocab_size: int, width: int, target_length: int = 1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.width = int(width)
        self.target_length = int(target_length)
        self.embed = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embed.weight, mean=0.0, std=EMBED_INIT_STD)
        self.embed_norm = nn.RMSNorm(width, elementwise_affine=False)
        self.norm_out = nn.RMSNorm(width)
        self._prefix: Tensor | None = None

    # -- context -----------------------------------------------------------
    def reset_context(self) -> None:
        self._prefix = None

    def ingest_context(self, demonstrations: list[tuple[Tensor, Tensor]]) -> None:
        """Stores the serialized demonstration prefix (no hidden reformatting)."""
        from bdhx.tasks.vocab import MAP, SEP

        toks: list[int] = [BOS]
        for i, (inp, out) in enumerate(demonstrations):
            if i:
                toks.append(SEP)
            toks.extend(int(t) for t in inp.flatten().tolist())
            toks.append(MAP)
            toks.extend(int(t) for t in out.flatten().tolist())
        self._prefix = torch.tensor(toks, dtype=torch.long).unsqueeze(0)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """Token embedding at unit RMS; the input of every `_run`."""
        return self.embed_norm(self.embed(tokens.to(self.embed.weight.device)))

    # -- readout -----------------------------------------------------------
    def logits_from_hidden(self, h: Tensor) -> Tensor:
        return F.linear(self.norm_out(h), self.embed.weight)

    def solve(
        self,
        query: Tensor,
        reasoning_steps: int,
        collect_diagnostics: bool = False,
        target_length: int | None = None,
    ) -> SolveOutput:
        q = query if query.dim() == 2 else query.unsqueeze(0)
        b = q.shape[0]
        dev = self.embed.weight.device
        q = q.to(dev)
        prefix = self._prefix
        if prefix is None:
            prefix = torch.tensor([[BOS]], dtype=torch.long, device=dev)
        prefix = prefix.to(dev).expand(b, -1)
        tl = int(target_length or self.target_length)

        def col(t: int) -> Tensor:
            return torch.full((b, 1), t, dtype=torch.long, device=dev)

        parts = [prefix, col(QUERY), q, col(ANSWER)]
        if tl > 1:
            parts.append(torch.full((b, tl - 1), PAD, dtype=torch.long, device=dev))
        seq = torch.cat(parts, dim=1)
        start = prefix.shape[1] + 1 + q.shape[1]  # index of the ANSWER token
        h, diag = self._run(seq, reasoning_steps, collect_diagnostics)
        logits = self.logits_from_hidden(h[:, start : start + tl])
        return SolveOutput(
            predictions=logits.argmax(dim=-1),
            logits=logits,
            diagnostics=diag,
            block_applications=self._applications(reasoning_steps),
        )

    def forward_episode(self, batch: EpisodeBatch, reasoning_steps: int) -> Tensor:
        h, _ = self._run(batch.serialized, reasoning_steps, False)
        lt = batch.target.shape[1]
        idx = batch.answer_start.unsqueeze(1) - 1 + torch.arange(lt, device=h.device)
        idx = idx.clamp(0, h.shape[1] - 1)
        gathered = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        return self.logits_from_hidden(gathered)

    # -- flops -------------------------------------------------------------
    def flops_estimate(self, batch: EpisodeBatch, reasoning_steps: int) -> FlopsReport:
        b, n = batch.serialized.shape
        d, f = self.width, ff_dim(self.width)
        applications = self._applications(reasoning_steps)
        attn = applications * (2 * n * (4 * d * d) + 4 * n * n * d)
        ff = applications * (2 * n * 3 * d * f)
        head = 2 * n * d * self.vocab_size
        per_episode = float(attn + ff + head)
        return FlopsReport(
            total=per_episode * b,
            per_episode=per_episode,
            breakdown={"attention": float(attn), "feedforward": float(ff), "head": float(head)},
        )

    # -- subclass hooks ----------------------------------------------------
    def _applications(self, reasoning_steps: int) -> int:
        raise NotImplementedError

    def _run(
        self, tokens: Tensor, reasoning_steps: int, collect_diagnostics: bool
    ) -> tuple[Tensor, dict[str, list]]:
        raise NotImplementedError


def block_stack_params(
    width: int, vocab_size: int, n_blocks: int, sandwich_norm: bool = False
) -> int:
    """Trainable params of an embedding + `n_blocks` blocks + final norm (tied head)."""
    blocks = nn.ModuleList(
        [TransformerBlock(width, sandwich_norm=sandwich_norm) for _ in range(n_blocks)]
    )
    return sum(p.numel() for p in blocks.parameters()) + vocab_size * width + width


def resolve_width(model_cfg, ctor) -> int:
    """cfg.width if given, else the width solver against cfg.params_target."""
    if model_cfg.width:
        return int(model_cfg.width)
    if not model_cfg.params_target:
        raise ValueError("model config needs either `width` or `params_target`")
    # step=2 is the finest grid RoPE allows (head_dim must stay even); coarser
    # grids miss the matched-control budget by more than the 3 percent tolerance
    # once the embedding dominates.
    width, _ = solve_width(ctor, int(model_cfg.params_target), step=2)
    return width


@register_model("transformer")
class TransformerModel(SeqReasoner):
    """Fixed-depth Transformer: `depth` distinct blocks, R fixed at 1."""

    def __init__(self, model_cfg, vocab_size: int, target_length: int = 1):
        depth = max(int(model_cfg.depth), 1)
        width = resolve_width(model_cfg, lambda w: block_stack_params(w, vocab_size, depth))
        super().__init__(vocab_size, width, target_length)
        self.depth = depth
        self.blocks = nn.ModuleList([TransformerBlock(width) for _ in range(depth)])
        self.solved_width = width

    def _applications(self, reasoning_steps: int) -> int:
        return self.depth  # R is fixed at 1 for this baseline

    def _run(
        self, tokens: Tensor, reasoning_steps: int, collect_diagnostics: bool
    ) -> tuple[Tensor, dict[str, list]]:
        h = self.embed_tokens(tokens)
        diag = {k: [] for k in ("state_norm", "update_norm", "cos_consecutive", "nan_count")}
        for blk in self.blocks:
            prev = h
            h = blk(h)
            if collect_diagnostics and blk is self.blocks[-1]:
                from bdhx.models.recurrence import RecurrenceRunner

                RecurrenceRunner._record(diag, prev, h)
        return h, diag
