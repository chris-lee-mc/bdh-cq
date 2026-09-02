"""Analytic per-model FLOP estimates (FRAMEWORK_SPEC sections 9, 12).

Every function takes a resolved `ModelCfg` (`.width` must be set -- the
value the param-budget solver picked), the serialized sequence length, and
the reasoning-depth `R`, and returns a `FlopsReport` whose `.per_episode`
is the forward (inference) cost and whose `.total` is the training cost
for `batch_size` episodes (forward + backward, backward taken as 2x
forward per Kaplan et al., so training = 3x forward).

Shared building blocks (all models stack a mixer + SwiGLU feed-forward
block, tied unembedding):

- projections (q/k/v/o or equivalent): `4 * n * d^2` multiply-adds -> `2 *`
  that many FLOPs, applied per block application.
- SwiGLU feed-forward: `3 * n * d * ff_dim(d)` multiply-adds (gate, up,
  down), `ff_dim(d) = round(8/3 * d)`.
- softmax attention mixer (transformer family, `kv` memory): QK^T + AV,
  `4 * n^2 * d` FLOPs.
- linear/recurrent mixer (BDH-style linear attention, Gated DeltaNet delta
  rule): `O(n * d * state_dim)` FLOPs, no quadratic-in-n term; the constant
  below (6) matches the ~6 matrix-vector/outer-product ops per token of the
  delta-rule update. `state_dim` is the per-head state width `d // heads`:
  a multi-head recurrent mixer carries `heads` states of `(d/heads)^2`, so
  the head-agnostic `6 * n * d^2` overcounts by `heads`. Gated DeltaNet
  passes its real head count (`pick_heads`); the single-state BDH-family
  estimates keep `heads=1`.
- tied unembedding head: `2 * n * d * vocab_size`.
"""

from __future__ import annotations

from bdhx.models.base import FlopsReport
from bdhx.models.transformer import ff_dim, pick_heads

LINEAR_MIXER_CONST = 6
TRAIN_FORWARD_BACKWARD_MULT = 3  # forward + ~2x backward


def _require_width(cfg) -> int:
    if not cfg.width:
        raise ValueError("flops estimate needs a resolved model_cfg.width")
    return int(cfg.width)


def _dense_flops(n: int, d: int) -> tuple[float, float]:
    """(projection, feed-forward) FLOPs for one block application, mixer-agnostic."""
    proj = 2.0 * n * 4 * d * d
    ff = 2.0 * n * 3 * d * ff_dim(d)
    return proj, ff


def _softmax_attn_flops(n: int, d: int) -> float:
    return 4.0 * n * n * d


def _linear_mixer_flops(n: int, d: int, heads: int = 1) -> float:
    return float(LINEAR_MIXER_CONST) * n * d * (d // max(heads, 1))


def train_flops(report: FlopsReport) -> float:
    """Section-9 train-FLOPs column: forward + backward over `report.total`."""
    return report.total * TRAIN_FORWARD_BACKWARD_MULT


def _assemble(
    applications: int,
    mixer: float,
    n: int,
    d: int,
    vocab_size: int,
    batch_size: int,
) -> FlopsReport:
    proj, ff = _dense_flops(n, d)
    mixer_total = applications * mixer
    proj_total = applications * proj
    ff_total = applications * ff
    head = 2.0 * n * d * vocab_size
    per_episode = mixer_total + proj_total + ff_total + head
    return FlopsReport(
        total=per_episode * batch_size,
        per_episode=per_episode,
        breakdown={
            "mixer": mixer_total,
            "projection": proj_total,
            "feedforward": ff_total,
            "head": head,
        },
    )


def transformer_flops(
    cfg, seq_len: int, r: int, vocab_size: int, batch_size: int = 1
) -> FlopsReport:
    """Fixed-depth Transformer: `depth` distinct softmax-attention blocks, R ignored."""
    d = _require_width(cfg)
    applications = max(int(cfg.depth), 1)
    return _assemble(
        applications, _softmax_attn_flops(seq_len, d), seq_len, d, vocab_size, batch_size
    )


def looped_transformer_flops(
    cfg, seq_len: int, r: int, vocab_size: int, batch_size: int = 1
) -> FlopsReport:
    """Shared softmax-attention block applied `r` times."""
    d = _require_width(cfg)
    applications = max(int(r), 0)
    return _assemble(
        applications, _softmax_attn_flops(seq_len, d), seq_len, d, vocab_size, batch_size
    )


def bdh_flops(cfg, seq_len: int, r: int, vocab_size: int, batch_size: int = 1) -> FlopsReport:
    """BDH: `depth` (or 1, shared) linear-attention blocks."""
    d = _require_width(cfg)
    applications = max(int(cfg.depth), 1)
    return _assemble(
        applications, _linear_mixer_flops(seq_len, d), seq_len, d, vocab_size, batch_size
    )


def bdh_cq_flops(cfg, seq_len: int, r: int, vocab_size: int, batch_size: int = 1) -> FlopsReport:
    """BDH-CQ: one linear-attention block, `r` reasoning (latent) steps."""
    d = _require_width(cfg)
    applications = max(int(r), 0)
    return _assemble(
        applications, _linear_mixer_flops(seq_len, d), seq_len, d, vocab_size, batch_size
    )


def gated_deltanet_flops(
    cfg, seq_len: int, r: int, vocab_size: int, batch_size: int = 1
) -> FlopsReport:
    """Gated DeltaNet: `depth` distinct delta-rule blocks, R fixed at 1 per layer."""
    d = _require_width(cfg)
    applications = max(int(cfg.depth), 1)
    mixer = _linear_mixer_flops(seq_len, d, pick_heads(d))
    return _assemble(applications, mixer, seq_len, d, vocab_size, batch_size)


def unified_block_flops(
    cfg, seq_len: int, r: int, vocab_size: int, batch_size: int = 1
) -> FlopsReport:
    """H3 shared block: mixer selected by `cfg.memory.kind` ('kv' or 'bdh'), R = r steps."""
    d = _require_width(cfg)
    applications = max(int(r), 0)
    mixer_flops = (
        _softmax_attn_flops(seq_len, d)
        if cfg.memory.kind == "kv"
        else _linear_mixer_flops(seq_len, d)
    )
    return _assemble(applications, mixer_flops, seq_len, d, vocab_size, batch_size)
