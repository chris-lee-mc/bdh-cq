"""Gated DeltaNet baseline (EXPERIMENT_PLAN section 4 item 4).

Pure-PyTorch O(T) recurrent reference for CPU:

    S_t = S_{t-1} * alpha_t * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

with sigmoid `beta_t` (write strength), a data-dependent decay
`alpha_t` in (0, 1) (sigmoid), L2-normalized keys `k_t`, and an output
gate. Implemented per-step via the algebraically equivalent, cheaper
delta-rule form (avoids the O(dim^3) matrix-matrix product):

    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T

Short convolution (pre-mixing token conv) is a documented option that is
off in this reference (`short_conv=False`; enabling it raises, since only
the no-conv path is validated here).

If `fla` is importable and CUDA is available, `fla.layers.GatedDeltaNet`
is used instead with matching hyperparameters (hidden_size, num_heads);
otherwise every block falls back to the reference above. The chosen
backend is recorded in `param_report().breakdown` as `backend:<name>`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bdhx.models.base import FlopsReport, ParamReport
from bdhx.models.transformer import SeqReasoner, SwiGLU, ff_dim, pick_heads, resolve_width
from bdhx.registry import register_model


def _fla_gated_deltanet_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import fla.layers  # noqa: F401
    except ImportError:
        return False
    return True


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class GatedDeltaNetMixer(nn.Module):
    """Reference recurrent Gated DeltaNet mixer (see module docstring)."""

    def __init__(self, width: int, heads: int, short_conv: bool = False):
        super().__init__()
        if short_conv:
            raise NotImplementedError("short_conv is off in this CPU reference")
        self.heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.alpha_proj = nn.Linear(width, heads)
        self.beta_proj = nn.Linear(width, heads)
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        h, dh = self.heads, self.head_dim
        q = self.q_proj(x).view(b, n, h, dh)
        k = _l2norm(self.k_proj(x).view(b, n, h, dh))
        v = self.v_proj(x).view(b, n, h, dh)
        alpha = torch.sigmoid(self.alpha_proj(x))  # (b, n, h), decay in (0, 1)
        beta = torch.sigmoid(self.beta_proj(x))  # (b, n, h), write strength

        state = x.new_zeros(b, h, dh, dh)  # (Dv, Dk) per head
        outs = []
        for t in range(n):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            at = alpha[:, t].unsqueeze(-1)  # (b, h, 1)
            bt = beta[:, t].unsqueeze(-1)  # (b, h, 1)
            sk = torch.einsum("bhvk,bhk->bhv", state, kt)  # S_{t-1} k_t
            delta = bt * (vt - at * sk)  # beta_t * (v_t - alpha_t S_{t-1} k_t)
            state = at.unsqueeze(-1) * state + torch.einsum("bhv,bhk->bhvk", delta, kt)
            outs.append(torch.einsum("bhvk,bhk->bhv", state, qt))
        o = torch.stack(outs, dim=1)  # (b, n, h, dh)
        o = self.out_norm(o)
        gate = torch.sigmoid(self.gate_proj(x).view(b, n, h, dh))
        o = (gate * o).reshape(b, n, d)
        return self.out_proj(o)


class FLAGatedDeltaNetMixer(nn.Module):  # pragma: no cover - requires CUDA + fla
    """Thin wrapper around `fla.layers.GatedDeltaNet` with matching hyperparameters."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        from fla.layers import GatedDeltaNet

        self.layer = GatedDeltaNet(hidden_size=width, num_heads=heads)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


class GatedDeltaNetBlock(nn.Module):
    """Pre-norm block: gated deltanet mixer + SwiGLU feed-forward."""

    def __init__(self, width: int, heads: int | None = None):
        super().__init__()
        heads = heads or pick_heads(width)
        self.norm1 = nn.RMSNorm(width)
        self.mixer = (
            FLAGatedDeltaNetMixer(width, heads)
            if _fla_gated_deltanet_available()
            else GatedDeltaNetMixer(width, heads)
        )
        self.norm2 = nn.RMSNorm(width)
        self.ff = SwiGLU(width, ff_dim(width))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


def gdn_block_stack_params(width: int, vocab_size: int, n_blocks: int) -> int:
    """Trainable params of an embedding + `n_blocks` GDN blocks + final norm (tied head)."""
    blocks = nn.ModuleList([GatedDeltaNetBlock(width) for _ in range(n_blocks)])
    return sum(p.numel() for p in blocks.parameters()) + vocab_size * width + width


@register_model("gated_deltanet")
class GatedDeltaNetModel(SeqReasoner):
    """Fixed-depth Gated DeltaNet baseline: `depth` distinct blocks, R fixed at 1."""

    def __init__(self, model_cfg, vocab_size: int, target_length: int = 1):
        depth = max(int(model_cfg.depth), 1)
        width = resolve_width(model_cfg, lambda w: gdn_block_stack_params(w, vocab_size, depth))
        super().__init__(vocab_size, width, target_length)
        self.depth = depth
        self.blocks = nn.ModuleList([GatedDeltaNetBlock(width) for _ in range(depth)])
        self.solved_width = width
        self.backend = "fla" if _fla_gated_deltanet_available() else "reference"
        self._resolved_cfg = model_cfg.model_copy(update={"width": width, "depth": depth})

    def param_report(self) -> ParamReport:
        rep = super().param_report()
        rep.breakdown[f"backend:{self.backend}"] = 0
        return rep

    def _applications(self, reasoning_steps: int) -> int:
        return self.depth  # R fixed at 1 per layer for this baseline

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

    def flops_estimate(self, batch, reasoning_steps: int) -> FlopsReport:
        from bdhx.training.flops import gated_deltanet_flops

        n = batch.serialized.shape[1]
        b = batch.serialized.shape[0]
        return gated_deltanet_flops(self._resolved_cfg, n, reasoning_steps, self.vocab_size, b)
