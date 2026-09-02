"""BDH family tests (FRAMEWORK_SPEC sections 3, 12, 13)."""

from __future__ import annotations

import copy

import pytest
import torch
from bdh_cq.bdh_cq import BDH

from bdhx.config import MemoryCfg, ModelCfg, RecurrenceCfg
from bdhx.models.base import BlockCounter
from bdhx.models.bdh import BDHModel, bdh_param_count
from bdhx.models.bdh_cq import BDHCQModel
from bdhx.models.recurrence import KINDS
from bdhx.registry import get_model
from bdhx.tasks.base import Episode, pad_and_batch
from bdhx.tasks.vocab import VOCAB_SIZE

V = 64
W = 16
HEADS = 2
DEPTH = 2
TL = 3


def mcfg(name="bdh", width=W, depth=DEPTH, memory_kind="bdh", precision_state="bf16", **rec):
    return ModelCfg(
        name=name,
        width=width,
        depth=depth,
        precision_state=precision_state,
        memory=MemoryCfg(kind=memory_kind),
        recurrence=RecurrenceCfg(**rec),
    )


def build(cls=BDHModel, seed=0, target_length=TL, loss="final_answer", **kwargs):
    torch.manual_seed(seed)
    cfg = kwargs.pop("cfg", None) or mcfg(name="bdh_cq" if cls is BDHCQModel else "bdh", **kwargs)
    model = cls(cfg, V, target_length=target_length, heads=HEADS, loss=loss)
    model.eval()
    return model


def episode_tensors(b=4, n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    demos = [(torch.randint(8, V, (b, n), generator=g), torch.randint(8, V, (b, n), generator=g))]
    return demos, torch.randint(8, V, (b, n), generator=g)


def batch_of(b=3, n=3, lt=2):
    eps = [
        Episode(
            demonstrations=[(torch.arange(8, 8 + n), torch.arange(12, 12 + n))],
            query=torch.arange(16, 16 + n),
            target=torch.arange(20 + i, 20 + i + lt),
            difficulty={"n": n},
            split="train",
            episode_id=i,
        )
        for i in range(b)
    ]
    return pad_and_batch(eps)


def test_registry_names():
    assert get_model("bdh") is BDHModel
    assert get_model("bdh_cq") is BDHCQModel


def test_batched_generate_parity():
    """Batched greedy decoding equals per-sequence community `generate`."""
    model = build()
    demos, query = episode_tensors(b=8)
    with torch.no_grad():
        model.ingest_context(demos)
        out = model.solve(query, 1)
        demo_stage, query_stage = model._demo_tokens(demos), model._query_stage(query)
        for i in range(8):
            tokens = model.wrapper.generate(
                demo_stage[i : i + 1],
                query_stage[i : i + 1],
                memories=None,
                num_tokens=TL,
                temperature=0.0,
            )
            assert tokens == out.predictions[i].tolist()


@pytest.mark.parametrize("r", [1, 4, 16])
def test_reasoning_steps_runtime(r):
    """R latent steps apply the block exactly R * depth extra times."""
    model = build(BDHCQModel)
    demos, query = episode_tensors()
    with torch.no_grad():
        model.ingest_context(demos)
        with BlockCounter(model.block_modules) as counter:
            model.solve(query, 0)
        baseline = counter.count
        with BlockCounter(model.block_modules) as counter:
            out = model.solve(query, r)
    assert counter.count - baseline == r * DEPTH
    assert out.block_applications == r * DEPTH
    assert out.predictions.shape == (4, TL)


@pytest.mark.parametrize("kind", KINDS)
def test_recurrence_kind_shapes(kind):
    model = build(BDHCQModel, kind=kind, adapter_rank=2)
    demos, query = episode_tensors(b=2)
    with torch.no_grad():
        model.ingest_context(demos)
        out = model.solve(query, 3, collect_diagnostics=True)
    assert out.logits.shape == (2, TL, V)
    assert len(out.diagnostics["state_norm"]) == 3
    assert out.diagnostics["nan_count"] == [0, 0, 0]


@pytest.mark.parametrize("cls", [BDHModel, BDHCQModel])
def test_context_isolation(cls):
    """Episode A, reset, episode B equals episode B in a fresh instance."""
    model = build(cls)
    fresh = copy.deepcopy(model)
    demos_a, query_a = episode_tensors(b=2, seed=1)
    demos_b, query_b = episode_tensors(b=2, seed=2)
    with torch.no_grad():
        model.ingest_context(demos_a)
        model.solve(query_a, 3)
        model.reset_context()
        model.ingest_context(demos_b)
        after = model.solve(query_b, 3)
        fresh.ingest_context(demos_b)
        expected = fresh.solve(query_b, 3)
    assert torch.equal(after.logits, expected.logits)
    assert torch.equal(after.predictions, expected.predictions)


@pytest.mark.parametrize("cls", [BDHModel, BDHCQModel])
def test_memory_reset_zeroes_state(cls):
    model = build(cls)
    demos, _ = episode_tensors(b=2)
    with torch.no_grad():
        model.ingest_context(demos)
    assert model._memories is not None
    model.reset_context()
    assert model._memories is None
    if cls is BDHCQModel:
        assert model._mem is None and model._all_block_outputs is None


def test_batch_independence():
    """Permuting the batch permutes the outputs; no cross-episode communication."""
    model = build(BDHCQModel)
    demos, query = episode_tensors(b=4)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        model.ingest_context(demos)
        out = model.solve(query, 2)
        model.reset_context()
        model.ingest_context([(i[perm], o[perm]) for i, o in demos])
        permuted = model.solve(query[perm], 2)
    assert torch.equal(permuted.predictions, out.predictions[perm])
    assert torch.allclose(permuted.logits, out.logits[perm], atol=1e-5)


def test_bare_wrapper_call_ending_on_int_stage_is_stale():
    """The community trap (PAPER_IMPLEMENTATION_GAPS section 2 item 4).

    A bare forward whose last stage is an int returns the previous tensor
    stage's logits; the adapter therefore never calls the wrapper that way.
    """
    model = build(BDHCQModel)
    tokens = torch.randint(8, V, (1, 4))
    with torch.no_grad():
        tensor_only = model.wrapper(tokens, memories=None)
        trailing_int = model.wrapper(tokens, 3, memories=None)
    assert torch.equal(tensor_only, trailing_int)


@pytest.mark.parametrize("cls", [BDHModel, BDHCQModel])
@pytest.mark.parametrize("loss", ["final_answer", "legacy"])
def test_adapter_never_ends_a_wrapper_call_on_an_int_stage(cls, loss):
    model = build(cls, loss=loss)
    calls: list[tuple] = []
    original = model.wrapper.forward

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    model.wrapper.forward = spy
    demos, query = episode_tensors(b=2)
    batch = batch_of()
    model.ingest_context(demos)
    model.solve(query, 3)
    model.forward_episode(batch, 3)
    model.episode_loss(batch, 3)
    assert calls
    for args in calls:
        stages = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        assert not isinstance(stages[-1], int)


def test_legacy_loss_rejects_non_community_recurrence():
    with pytest.raises(ValueError, match="legacy"):
        build(BDHCQModel, kind="step_gate", loss="legacy")


def test_bdh_rejects_recurrence_kinds():
    with pytest.raises(ValueError, match="no latent loop"):
        build(BDHModel, kind="residual")


@pytest.mark.parametrize("share_weights", [True, False])
def test_param_report_matches_community(share_weights):
    """param_report().total equals the community BDH built with the same kwargs."""
    model = build(share_weights=share_weights)
    reference = BDH(
        dim=W,
        num_tokens=V,
        depth=DEPTH,
        heads=HEADS,
        dim_qk_heads=4 * W,
        rotary_dim=2 * W,
    )
    community = sum(p.numel() for p in reference.parameters())
    blocks = 1 if share_weights else DEPTH
    expected = community + (blocks - 1) * 3 * W * 4 * W
    assert model.param_report().total == expected
    assert bdh_param_count(W, V, heads=HEADS, depth=DEPTH, share_weights=share_weights) == expected
    assert model.param_report().total == sum(p.numel() for p in model.state_dict().values())


@pytest.mark.parametrize("target", [200_000, 2_000_000])
def test_solve_width_hits_target(target):
    cfg = ModelCfg(
        name="bdh",
        params_target=target,
        depth=1,
        memory=MemoryCfg(),
        recurrence=RecurrenceCfg(),
    )
    torch.manual_seed(0)
    model = BDHModel(cfg, VOCAB_SIZE)
    realized = model.param_report().total
    assert abs(realized - target) / target <= 0.03


def test_forward_episode_and_losses():
    model = build(BDHCQModel)
    batch = batch_of()
    logits = model.forward_episode(batch, 2)
    assert logits.shape == (len(batch), batch.target.shape[1], V)
    assert model.episode_loss(batch, 2).ndim == 0
    model.loss = "legacy"
    assert model.episode_loss(batch, 2).ndim == 0


def test_precision_state_rounds_the_carried_latent():
    """`precision_state` changes the fed-back latent, and only that."""
    bf16 = build(BDHCQModel)
    fp32 = build(BDHCQModel, cfg=mcfg(name="bdh_cq", precision_state="fp32"))
    demos, query = episode_tensors(b=2)
    with torch.no_grad():
        bf16.ingest_context(demos)
        fp32.ingest_context(demos)
        a, b = bf16.solve(query, 4), fp32.solve(query, 4)
    assert bf16.state_dtype is torch.bfloat16
    assert not torch.equal(a.logits, b.logits)
    assert torch.allclose(a.logits, b.logits, atol=5e-2)


def test_memory_kind_none_freezes_the_hebbian_update():
    model = build(memory_kind="none")
    demos, query = episode_tensors(b=2)
    with torch.no_grad():
        model.ingest_context(demos)
    assert all(m is None for m in model._memories.fast_weight_memories)
    with torch.no_grad():
        assert model.solve(query, 1).predictions.shape == (2, TL)


# -- adversarial verification (FRAMEWORK_SPEC sections 3, 3.1) ---------------


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("kind", KINDS)
def test_context_isolation_bitwise_all_kinds(kind, deterministic):
    """Every recurrence kind: reset then B equals a fresh instance on B, bit-for-bit."""
    model = build(BDHCQModel, kind=kind, adapter_rank=2)
    fresh = copy.deepcopy(model)
    demos_a, query_a = episode_tensors(b=2, seed=1)
    demos_b, query_b = episode_tensors(b=2, seed=2)
    with torch.no_grad():
        model.ingest_context(demos_a)
        model.solve(query_a, 5, collect_diagnostics=True)
        model.forward_episode(batch_of(), 5)
        model.reset_context()
        model.ingest_context(demos_b)
        after = model.solve(query_b, 3)
        fresh.ingest_context(demos_b)
        expected = fresh.solve(query_b, 3)
    assert torch.equal(after.logits, expected.logits)


@pytest.mark.parametrize("cls", [BDHModel, BDHCQModel])
def test_solve_does_not_consume_the_context(cls, deterministic):
    """Decoding must not write back into the ingested Memory."""
    model = build(cls)
    demos, query = episode_tensors(b=2)
    with torch.no_grad():
        model.ingest_context(demos)
        first = model.solve(query, 3)
        second = model.solve(query, 3)
    assert torch.equal(first.logits, second.logits)


def test_reset_clears_latent_scratch_after_an_aborted_solve():
    """A failure inside the latent loop must not leave episode A state behind."""
    model = build(BDHCQModel)
    demos, query = episode_tensors(b=2)
    model.ingest_context(demos)
    original = model.bdh.forward
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("aborted")
        return original(*args, **kwargs)

    model.bdh.forward = boom
    with pytest.raises(RuntimeError, match="aborted"), torch.no_grad():
        model.solve(query, 4)
    model.bdh.forward = original
    assert model._mem is not None  # scratch survives the abort
    model.reset_context()
    assert model._memories is None
    assert model._mem is None and model._all_block_outputs is None


def test_no_cross_episode_communication(deterministic):
    """Changing one batch row leaves every other row bit-for-bit identical."""
    model = build(BDHCQModel)
    demos, query = episode_tensors(b=4, seed=3)
    with torch.no_grad():
        model.ingest_context(demos)
        base = model.solve(query, 3)
        other = [(i.clone(), o.clone()) for i, o in demos]
        other[0][0][3] = 9
        changed_query = query.clone()
        changed_query[3] = torch.full((query.shape[1],), 11)
        model.reset_context()
        model.ingest_context(other)
        changed = model.solve(changed_query, 3)
    assert torch.equal(base.logits[:3], changed.logits[:3])
    assert not torch.equal(base.logits[3], changed.logits[3])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("r", [0, 1, 4])
def test_block_applications_are_runtime_for_every_kind(kind, r):
    model = build(BDHCQModel, kind=kind, adapter_rank=2)
    demos, query = episode_tensors(b=2)
    with torch.no_grad():
        model.ingest_context(demos)
        with BlockCounter(model.block_modules) as counter:
            out = model.solve(query, r)
        with BlockCounter(model.block_modules) as zero:
            model.solve(query, 0)
    assert counter.count - zero.count == r * DEPTH
    assert out.block_applications == r * DEPTH


def test_baseline_bdh_has_no_latent_loop():
    """Documented deviation: `bdh` ignores R (no latent loop to iterate)."""
    model = build(BDHModel)
    demos, query = episode_tensors(b=2)
    with torch.no_grad():
        model.ingest_context(demos)
        with BlockCounter(model.block_modules) as one:
            a = model.solve(query, 1)
        with BlockCounter(model.block_modules) as many:
            b = model.solve(query, 16)
    assert one.count == many.count
    assert a.block_applications == b.block_applications == DEPTH
    assert torch.equal(a.logits, b.logits)


# -- section 3.1 update rules ------------------------------------------------


def _runner(kind, block, **kwargs):
    from bdhx.models.recurrence import RecurrenceRunner

    return RecurrenceRunner(block, kind=kind, r_max=4, width=W, **kwargs)


def _lin_block():
    torch.manual_seed(0)
    w = torch.randn(W, W) * 0.1
    return lambda h, _s: torch.tanh(h @ w)


def test_update_rules_match_spec_3_1():
    f = _lin_block()
    h0 = torch.randn(2, 1, W)
    plain, _ = _runner("plain", f)(h0, None, 3)
    residual, _ = _runner("residual", f)(h0, None, 3)
    # plain: H[r+1] = F(H[r])
    expect = h0
    for _ in range(3):
        expect = f(expect, None)
    assert torch.equal(plain, expect)
    # residual: H[r+1] = H[r] + F(H[r])
    expect = h0
    for _ in range(3):
        expect = expect + f(expect, None)
    assert torch.equal(residual, expect)

    gate = _runner("step_gate", f)
    with torch.no_grad():
        gate.alpha.zero_()
    zero, _ = gate(h0, None, 3)
    assert torch.equal(zero, h0)  # alpha = 0 -> identity
    with torch.no_grad():
        gate.alpha.fill_(1.0)
    one, _ = gate(h0, None, 3)
    assert torch.equal(one, residual)  # alpha = 1 -> residual

    skip = _runner("init_skip", f)  # g initialised to 0 -> plain
    assert torch.equal(skip(h0, None, 3)[0], plain)
    with torch.no_grad():
        skip.skip_gate.fill_(1.0)
    expect = h0
    for _ in range(3):
        expect = f(expect, None) + h0
    assert torch.equal(skip(h0, None, 3)[0], expect)

    emb = _runner("step_emb", f)  # e initialised to 0 -> plain
    assert torch.equal(emb(h0, None, 3)[0], plain)
    with torch.no_grad():
        emb.step_emb.fill_(0.5)
    expect = h0
    for _ in range(3):
        expect = f(expect + 0.5, None)
    assert torch.equal(emb(h0, None, 3)[0], expect)

    adapt = _runner("adapter", f, adapter_rank=2)  # A initialised to 0 -> plain
    assert torch.equal(adapt(h0, None, 3)[0], plain)
    with torch.no_grad():
        adapt.adapter_a.normal_()
    expect = h0
    for r in range(3):
        expect = f(expect, None) + (expect @ adapt.adapter_b[r].T) @ adapt.adapter_a[r].T
    assert torch.allclose(adapt(h0, None, 3)[0], expect, atol=1e-6)


def test_gate_extrapolation_beyond_r_max():
    def f(h, _s):
        return h

    h0 = torch.randn(1, 1, W)
    hold = _runner("step_gate", f)
    with torch.no_grad():
        hold.alpha.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # r_max = 4, hold_last: alpha for r >= 4 is alpha[3]
    out, _ = hold(h0, None, 6)
    scale = 2.0 * 3.0 * 4.0 * 5.0 * 5.0 * 5.0  # prod(1 + alpha[r]), alpha held at 4
    assert torch.allclose(out, h0 * scale)
    interp = _runner("step_gate", f, gate_extrapolation="interpolate")
    with torch.no_grad():
        interp.alpha.copy_(hold.alpha)
    assert not torch.equal(interp(h0, None, 6)[0], out)


def test_recurrence_params_are_used_not_dead_weight():
    """Every added parameter must receive gradient (matched-control rule)."""
    for kind in ("step_gate", "init_skip", "step_emb", "adapter", "combo"):
        model = build(BDHCQModel, kind=kind, adapter_rank=2)
        model.train()
        model.episode_loss(batch_of(), 3).backward()
        grads = {n: p.grad for n, p in model.recurrence.named_parameters()}
        assert grads, kind
        assert all(g is not None for g in grads.values()), kind
        # adapter_b is reached only through the zero-initialised adapter_a (LoRA
        # init), so it is the one parameter with no gradient at step 0.
        assert any(float(g.abs().sum()) > 0 for n, g in grads.items() if n != "adapter_b"), kind


@pytest.mark.parametrize("kind", KINDS)
def test_param_report_equals_state_dict_for_every_kind(kind):
    model = build(BDHCQModel, kind=kind, adapter_rank=2, step_embedding=True)
    report = model.param_report()
    assert report.total == sum(p.numel() for p in model.state_dict().values())
    assert sum(report.breakdown.values()) == report.total
    # the only frozen tensor is the community rotary frequency table
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"wrapper.bdh.rope.freqs"}
    assert report.trainable == report.total - model.bdh.rope.freqs.numel()


def test_bdh_param_count_covers_attn_residual_and_step_embed():
    model = build(BDHCQModel, kind="attn_residual", step_embedding=True)
    analytic = (
        bdh_param_count(
            W,
            V,
            heads=HEADS,
            depth=DEPTH,
            attn_residual=True,
            attn_residual_depth_bias_distance=1,
        )
        + W  # latent_step_embed
    )
    assert model.param_report().total == analytic


@pytest.mark.parametrize("step_embedding", [False, True])
def test_width_solver_target_includes_every_added_param(step_embedding):
    cfg = ModelCfg(
        name="bdh_cq",
        params_target=500_000,
        depth=1,
        memory=MemoryCfg(),
        recurrence=RecurrenceCfg(kind="step_gate", step_embedding=step_embedding),
    )
    torch.manual_seed(0)
    model = BDHCQModel(cfg, VOCAB_SIZE)
    assert model.param_report().total == model._count_at(model.dim)
    assert abs(model.param_report().total - 500_000) / 500_000 <= 0.03
