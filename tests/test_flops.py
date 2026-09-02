"""Analytic FLOP estimate tests (FRAMEWORK_SPEC section 12: test_flops_estimate_monotone)."""

from __future__ import annotations

import pytest
import torch

from bdhx.config import MemoryCfg, ModelCfg, RecurrenceCfg
from bdhx.models.transformer import TransformerModel
from bdhx.training.flops import (
    TRAIN_FORWARD_BACKWARD_MULT,
    bdh_cq_flops,
    bdh_flops,
    gated_deltanet_flops,
    looped_transformer_flops,
    train_flops,
    transformer_flops,
    unified_block_flops,
)

V = 4128


def mcfg(width=32, depth=2, memory_kind="bdh"):
    return ModelCfg(
        name="x",
        width=width,
        depth=depth,
        recurrence=RecurrenceCfg(),
        memory=MemoryCfg(kind=memory_kind),
    )


R_FUNCS = [looped_transformer_flops, bdh_cq_flops, unified_block_flops]
DEPTH_FUNCS = [transformer_flops, bdh_flops, gated_deltanet_flops]


@pytest.mark.parametrize("fn", R_FUNCS)
def test_monotone_in_r(fn):
    cfg = mcfg()
    lo = fn(cfg, seq_len=16, r=1, vocab_size=V).per_episode
    hi = fn(cfg, seq_len=16, r=8, vocab_size=V).per_episode
    assert hi > lo > 0


@pytest.mark.parametrize("fn", R_FUNCS + DEPTH_FUNCS)
def test_monotone_in_width(fn):
    small = mcfg(width=16)
    big = mcfg(width=64)
    lo = fn(small, seq_len=16, r=4, vocab_size=V).per_episode
    hi = fn(big, seq_len=16, r=4, vocab_size=V).per_episode
    assert hi > lo > 0


def test_monotone_in_depth():
    for fn in DEPTH_FUNCS:
        shallow = mcfg(depth=1)
        deep = mcfg(depth=4)
        lo = fn(shallow, seq_len=16, r=1, vocab_size=V).per_episode
        hi = fn(deep, seq_len=16, r=1, vocab_size=V).per_episode
        assert hi > lo > 0


def test_total_is_batch_forward_cost():
    """`.total` is the batch forward cost, same convention as the model-level
    `flops_estimate` methods; the fwd+bwd multiplier lives in `train_flops`."""
    cfg = mcfg()
    rep = transformer_flops(cfg, seq_len=16, r=1, vocab_size=V, batch_size=5)
    assert rep.total == pytest.approx(rep.per_episode * 5)
    assert train_flops(rep) == pytest.approx(rep.total * TRAIN_FORWARD_BACKWARD_MULT)


class _Batch:
    def __init__(self, b, n):
        self.serialized = torch.zeros(b, n, dtype=torch.long)

    def __len__(self):
        return self.serialized.shape[0]


def test_matches_model_level_transformer_estimate():
    """flops.py and TransformerModel.flops_estimate must agree, else a
    comparison group mixes units and the section-9 15% flag is meaningless."""
    cfg = mcfg(width=32, depth=2)
    torch.manual_seed(0)
    m = TransformerModel(cfg, V)
    batch = _Batch(3, 12)
    a = m.flops_estimate(batch, reasoning_steps=1)
    b = transformer_flops(cfg, seq_len=12, r=1, vocab_size=V, batch_size=3)
    assert a.per_episode == pytest.approx(b.per_episode)
    assert a.total == pytest.approx(b.total)


def test_gdn_mixer_is_head_aware():
    """Multi-head recurrent state is heads x (d/heads)^2, not d^2."""
    from bdhx.models.transformer import pick_heads
    from bdhx.training.flops import LINEAR_MIXER_CONST

    d = 64
    rep = gated_deltanet_flops(mcfg(width=d, depth=1), seq_len=16, r=1, vocab_size=V)
    assert rep.breakdown["mixer"] == pytest.approx(
        LINEAR_MIXER_CONST * 16 * d * (d // pick_heads(d))
    )


def test_unified_block_mixer_selects_by_memory_kind():
    kv = unified_block_flops(mcfg(memory_kind="kv"), seq_len=16, r=4, vocab_size=V)
    bdh = unified_block_flops(mcfg(memory_kind="bdh"), seq_len=16, r=4, vocab_size=V)
    assert kv.breakdown["mixer"] != bdh.breakdown["mixer"]


def test_requires_resolved_width():
    cfg = ModelCfg(name="x", recurrence=RecurrenceCfg(), memory=MemoryCfg())
    with pytest.raises(ValueError):
        transformer_flops(cfg, seq_len=8, r=1, vocab_size=V)
