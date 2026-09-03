"""Transformer family tests (FRAMEWORK_SPEC sections 3, 3.1, 12)."""

from __future__ import annotations

import pytest
import torch

from bdhx.config import MemoryCfg, ModelCfg, RecurrenceCfg
from bdhx.models.base import BlockCounter
from bdhx.models.looped_transformer import LoopedTransformer
from bdhx.models.param_budget import solve_width
from bdhx.models.recurrence import KINDS, RecurrenceRunner
from bdhx.models.transformer import TransformerBlock, TransformerModel, block_stack_params
from bdhx.models.unified_block import UnifiedBlockModel
from bdhx.registry import get_model
from bdhx.tasks.base import pad_and_batch
from bdhx.tasks.vocab import ANSWER, BOS, MAP, QUERY, SEP

V = 64
W = 16


def mcfg(name="looped_transformer", width=W, **rec):
    return ModelCfg(
        name=name,
        width=width,
        depth=rec.pop("depth", 1),
        memory=MemoryCfg(kind=rec.pop("memory_kind", "bdh")),
        recurrence=RecurrenceCfg(**rec),
    )


def looped(**rec):
    torch.manual_seed(0)
    return LoopedTransformer(mcfg(**rec), V, target_length=2, r_max=4)


def tokens(b=2, n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(8, V, (b, n), generator=g)


def test_registry_names():
    assert get_model("transformer") is TransformerModel
    assert get_model("looped_transformer") is LoopedTransformer
    assert get_model("unified_block") is UnifiedBlockModel


@pytest.mark.parametrize("r", [1, 4, 16])
def test_reasoning_steps_runtime(r):
    m = looped()
    q = tokens()
    with BlockCounter(m.block) as c:
        out = m.solve(q, r)
    assert c.count == r
    assert out.block_applications == r
    assert out.predictions.shape == (2, 2)


@pytest.mark.parametrize("kind", KINDS)
def test_recurrence_variants_shapes(kind):
    m = looped(kind=kind, adapter_rank=2)
    q = tokens()
    out = m.solve(q, 6, collect_diagnostics=True)
    assert out.logits.shape == (2, 2, V)
    for key in ("state_norm", "update_norm", "cos_consecutive", "nan_count"):
        assert len(out.diagnostics[key]) == 6
    assert out.diagnostics["nan_count"] == [0] * 6


def test_recurrence_gate_semantics():
    """step_gate equals residual at alpha=1 and plain-plus-skip at alpha=0."""
    torch.manual_seed(0)
    blk = TransformerBlock(W)
    h = torch.randn(2, 5, W)

    def f(x, _s):
        return blk(x)

    plain = RecurrenceRunner(f, "plain", r_max=2, width=W)(h, None, 1)[0]
    residual = RecurrenceRunner(f, "residual", r_max=2, width=W)(h, None, 1)[0]
    gate = RecurrenceRunner(f, "step_gate", r_max=2, width=W)
    assert torch.allclose(gate(h, None, 1)[0], residual, atol=1e-6)
    with torch.no_grad():
        gate.alpha.zero_()
    assert torch.allclose(gate(h, None, 1)[0], h, atol=1e-6)
    assert not torch.allclose(plain, residual)


def test_gate_extrapolation_hold_last():
    def f(x, _s):
        return x * 0.5

    r = RecurrenceRunner(f, "step_gate", r_max=2, width=W)
    with torch.no_grad():
        r.alpha.copy_(torch.tensor([1.0, 2.0]))
    assert torch.allclose(r._gate(r.alpha, 5, 8), torch.tensor(2.0))
    ri = RecurrenceRunner(f, "step_gate", r_max=2, width=W, gate_extrapolation="interpolate")
    with torch.no_grad():
        ri.alpha.copy_(torch.tensor([1.0, 2.0]))
    assert 1.0 <= float(ri._gate(ri.alpha, 3, 8).detach()) <= 2.0


def test_context_isolation():
    demo_a = [(torch.tensor([10, 11]), torch.tensor([12]))]
    demo_b = [(torch.tensor([20]), torch.tensor([21, 22]))]
    q = tokens(b=1, n=3, seed=3)
    m = looped()
    m.eval()
    m.ingest_context(demo_a)
    m.solve(q, 3)
    m.reset_context()
    m.ingest_context(demo_b)
    with torch.no_grad():
        got = m.solve(q, 3).logits
    fresh = looped()
    fresh.eval()
    fresh.ingest_context(demo_b)
    with torch.no_grad():
        want = fresh.solve(q, 3).logits
    assert torch.equal(got, want)


def test_batch_independence():
    m = looped()
    m.eval()
    q = tokens(b=4, n=6, seed=7)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = m.solve(q, 3).logits[perm]
        b = m.solve(q[perm], 3).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_memory_reset_zeroes_state():
    m = looped()
    m.ingest_context([(torch.tensor([10]), torch.tensor([11]))])
    assert m._prefix is not None
    m.reset_context()
    assert m._prefix is None


def test_param_count_matches_state_dict():
    for m in (
        looped(kind="attn_residual"),
        TransformerModel(mcfg("transformer", depth=2), V),
        UnifiedBlockModel(mcfg("unified_block", memory_kind="kv"), V),
    ):
        rep = m.param_report()
        assert rep.total == sum(t.numel() for t in m.state_dict().values())
        assert rep.trainable == rep.total


def test_unified_block_bdh_mixer():
    m = UnifiedBlockModel(mcfg("unified_block", memory_kind="bdh"), V, target_length=2, r_max=4)
    out = m.solve(tokens(), 3)
    assert out.logits.shape == (2, 2, V)
    assert type(m.block.layers[0].mixer).__name__ == "BDHBlock"


def test_forward_episode_and_flops():
    from bdhx.tasks.base import Episode

    eps = [
        Episode(
            demonstrations=[(torch.tensor([10, 11]), torch.tensor([12]))],
            query=torch.tensor([10, 11]),
            target=torch.tensor([12, 13]),
            difficulty={"depth": 1},
            split="train",
            episode_id=i,
        )
        for i in range(2)
    ]
    batch = pad_and_batch(eps)
    m = looped()
    logits = m.forward_episode(batch, 3)
    assert logits.shape == (2, 2, V)
    f1 = m.flops_estimate(batch, 1).per_episode
    f4 = m.flops_estimate(batch, 4).per_episode
    assert f4 > f1 > 0


def test_serialized_prefix_tokens():
    m = looped()
    m.ingest_context(
        [(torch.tensor([10]), torch.tensor([11])), (torch.tensor([12]), torch.tensor([13]))]
    )
    assert m._prefix.flatten().tolist() == [BOS, 10, MAP, 11, SEP, 12, MAP, 13]
    assert QUERY != ANSWER


@pytest.mark.parametrize("target", [200_000, 2_000_000])
@pytest.mark.parametrize("name", ["transformer", "looped_transformer"])
def test_solve_width_hits_target(name, target):
    if name == "transformer":

        def ctor(w):
            return block_stack_params(w, 4128, 2)
    else:

        def ctor(w):
            cfg = ModelCfg(name=name, width=w, recurrence=RecurrenceCfg(kind="residual"))
            return LoopedTransformer(cfg, 4128)

    width, realized = solve_width(ctor, target, step=4)
    assert abs(realized - target) / target <= 0.03
    assert width % 4 == 0


# --- adversarial verification (FRAMEWORK_SPEC section 3 guarantees) ----------


def unified(memory_kind="bdh", **rec):
    torch.manual_seed(0)
    return UnifiedBlockModel(
        mcfg("unified_block", memory_kind=memory_kind, **rec), V, target_length=2, r_max=4
    )


def plain_transformer(depth=2):
    torch.manual_seed(0)
    return TransformerModel(mcfg("transformer", depth=depth), V, target_length=2)


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("build", [looped, unified, plain_transformer])
def test_context_isolation_bitwise_deterministic(deterministic, build):
    """Spec: reset + episode B must match a fresh instance bit-for-bit."""
    demo_a = [(torch.tensor([10, 11]), torch.tensor([12]))]
    demo_b = [(torch.tensor([20]), torch.tensor([21, 22]))]
    q = tokens(b=1, n=3, seed=3)
    m = build()
    m.eval()
    m.ingest_context(demo_a)
    m.solve(q, 5)
    m.reset_context()
    m.ingest_context(demo_b)
    with torch.no_grad():
        got = m.solve(q, 3).logits
    fresh = build()
    fresh.eval()
    fresh.ingest_context(demo_b)
    with torch.no_grad():
        want = fresh.solve(q, 3).logits
    assert torch.equal(got, want)


@pytest.mark.parametrize("build", [looped, unified, plain_transformer])
def test_no_hidden_state_survives_reset(build):
    """No buffer, cache or attribute picks up episodic state."""
    m = build()
    m.eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    attrs = set(m.__dict__)
    m.ingest_context([(torch.tensor([10, 11]), torch.tensor([12]))])
    with torch.no_grad():
        m.solve(tokens(), 4, collect_diagnostics=True)
    m.reset_context()
    after = m.state_dict()
    assert set(after) == set(before)
    for k, v in after.items():
        assert torch.equal(v, before[k]), k
    assert set(m.__dict__) == attrs
    assert all(v is None for k, v in m.__dict__.items() if k == "_prefix")


@pytest.mark.parametrize(
    "build", [looped, lambda: unified("bdh"), lambda: unified("kv"), plain_transformer]
)
def test_batch_permutation_and_row_isolation(build):
    """Batch dim carries independent episodes: no cross-episode communication."""
    m = build()
    m.eval()
    q = tokens(b=4, n=6, seed=7)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = m.solve(q, 3).logits[perm]
        b = m.solve(q[perm], 3).logits
        base = m.solve(q, 3).logits
        q2 = q.clone()
        q2[0, 0] = int(q2[0, 0]) + 1
        changed = m.solve(q2, 3).logits
    assert torch.equal(a, b)
    assert torch.equal(base[1:], changed[1:])


@pytest.mark.parametrize("kind", KINDS)
def test_block_applications_equal_reasoning_steps(kind):
    """Every recurrence kind applies the shared block exactly R times."""
    m = looped(kind=kind, adapter_rank=2)
    q = tokens()
    for r in (1, 3, 7):
        with BlockCounter(m.block) as c:
            out = m.solve(q, r)
        assert c.count == r
        assert out.block_applications == r


def test_reasoning_steps_not_baked_in():
    """R is a runtime arg only: repeat calls are stateless and R changes output."""
    m = looped(kind="residual")
    m.eval()
    q = tokens(b=2, n=5, seed=1)
    with torch.no_grad():
        a1 = m.solve(q, 1).logits
        a8 = m.solve(q, 8).logits
        a1_again = m.solve(q, 1).logits
    assert torch.equal(a1, a1_again)
    assert not torch.allclose(a1, a8)
    u = unified()
    with BlockCounter(u.block) as c:
        u.solve(q, 5)
    assert c.count == 5
    t = plain_transformer(depth=3)
    with BlockCounter(list(t.blocks)) as c:
        t.solve(q, 9)
    assert c.count == 3  # fixed-depth baseline ignores R by design


def _ref_step(kind, runner, f, h, h0, r):
    if kind == "plain":
        return f(h, None)
    if kind == "residual":
        return h + f(h, None)
    if kind == "step_gate":
        return h + runner.alpha[r] * f(h, None)
    if kind == "init_skip":
        return f(h, None) + runner.skip_gate[r] * h0
    if kind == "step_emb":
        return f(h + runner.step_emb[r], None)
    if kind == "adapter":
        return f(h, None) + (h @ runner.adapter_b[r].T) @ runner.adapter_a[r].T
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind", ["plain", "residual", "step_gate", "init_skip", "step_emb", "adapter"]
)
def test_recurrence_update_rules_exact(kind):
    """Each kind reproduces the section 3.1 update rule for two steps."""
    torch.manual_seed(0)
    h0 = torch.randn(2, 5, W)

    def f(x, _s):
        return x * 0.5 + 1.0

    r = RecurrenceRunner(f, kind, r_max=4, width=W, adapter_rank=3)
    with torch.no_grad():
        for p in r.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    got = r(h0, None, 2)[0]
    h = _ref_step(kind, r, f, h0, h0, 0)
    want = _ref_step(kind, r, f, h, h0, 1)
    assert torch.allclose(got, want, atol=1e-6)


def test_zero_gate_variants_reduce_to_plain():
    """init_skip(g=0), step_emb(e=0) and adapter(A=0) must equal plain."""
    torch.manual_seed(0)
    h = torch.randn(2, 4, W)

    def f(x, _s):
        return torch.tanh(x) * 1.3

    plain = RecurrenceRunner(f, "plain", r_max=3, width=W)(h, None, 3)[0]
    for kind, zeros in (("init_skip", "skip_gate"), ("step_emb", "step_emb"), ("adapter", None)):
        r = RecurrenceRunner(f, kind, r_max=3, width=W, adapter_rank=2)
        if zeros:
            with torch.no_grad():
                getattr(r, zeros).zero_()
        assert torch.allclose(r(h, None, 3)[0], plain, atol=1e-6), kind


def test_attn_residual_is_convex_mixture():
    """attn_residual output is a softmax mixture of the seed and block outputs."""
    torch.manual_seed(0)
    r = RecurrenceRunner(lambda x, _s: x * 0.0 + 2.0, "attn_residual", r_max=3, width=W)
    h = torch.full((1, 3, W), 2.0)
    out = r(h, None, 3)[0]
    assert torch.allclose(out, h, atol=1e-6)  # all keys equal -> mixture is the same value
    assert sum(p.numel() for p in r.parameters()) == 2 * W + 1


def test_module_block_is_not_double_counted():
    """An nn.Module block passed to the runner must not become a submodule."""
    blk = TransformerBlock(W)
    r = RecurrenceRunner(blk, "step_gate", r_max=2, width=W)
    assert r.block is blk
    assert not any("mixer" in k for k in r.state_dict())
    assert sum(p.numel() for p in r.parameters()) == 2


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("build", [looped, unified])
def test_param_report_matches_state_dict_all_kinds(build, kind):
    m = build(kind=kind, adapter_rank=2)
    rep = m.param_report()
    assert rep.total == sum(t.numel() for t in m.state_dict().values())
    assert rep.total == sum(rep.breakdown.values())
    assert rep.serialized_bytes > 0


def test_recurrence_params_matches_constructed_runner():
    """The width solver's recurrence term equals the runner it will build."""
    from bdhx.models.looped_transformer import recurrence_params

    for kind in KINDS:
        cfg = mcfg(kind=kind, adapter_rank=3)
        torch.manual_seed(0)
        m = LoopedTransformer(cfg, V, r_max=5)
        got = sum(p.numel() for p in m.recurrence.parameters())
        assert got == recurrence_params(m.width, cfg.recurrence, 5), kind


@pytest.mark.parametrize("kind", ["plain", "adapter", "step_emb"])
def test_params_target_matched_control_within_tolerance(kind):
    """Matched-control rule: solved widths land within the 3 percent tolerance."""
    cfg = ModelCfg(
        name="looped_transformer",
        width=None,
        params_target=200_000,
        depth=1,
        memory=MemoryCfg(kind="bdh"),
        recurrence=RecurrenceCfg(kind=kind, adapter_rank=4),
    )
    m = LoopedTransformer(cfg, 4128)
    total = m.param_report().total
    assert abs(total - 200_000) / 200_000 <= 0.03
    assert m.width % 2 == 0


def test_solve_matches_forward_episode_readout():
    """solve() reads the same position forward_episode() trains on."""
    from bdhx.tasks.base import Episode

    ep = Episode(
        demonstrations=[(torch.tensor([10, 11]), torch.tensor([12]))],
        query=torch.tensor([14, 15]),
        target=torch.tensor([16]),
        difficulty={"depth": 1},
        split="train",
        episode_id=0,
    )
    torch.manual_seed(0)
    m = LoopedTransformer(mcfg(), V, target_length=1, r_max=4)
    m.eval()
    with torch.no_grad():
        trained = m.forward_episode(pad_and_batch([ep]), 3)
        m.ingest_context(ep.demonstrations)
        solved = m.solve(ep.query.unsqueeze(0), 3).logits
    assert torch.equal(trained, solved)
