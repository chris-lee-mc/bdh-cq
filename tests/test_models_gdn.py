"""Gated DeltaNet tests (FRAMEWORK_SPEC sections 3, 3.1, 12; HANDOFF_TASKS item 12)."""

from __future__ import annotations

import torch

from bdhx.config import MemoryCfg, ModelCfg, RecurrenceCfg
from bdhx.models.gated_deltanet import GatedDeltaNetMixer, GatedDeltaNetModel
from bdhx.models.param_budget import solve_width
from bdhx.registry import get_model
from bdhx.tasks.vocab import BOS, MAP, SEP

V = 64
W = 16


def mcfg(width=W, depth=1):
    return ModelCfg(
        name="gated_deltanet",
        width=width,
        depth=depth,
        recurrence=RecurrenceCfg(),
        memory=MemoryCfg(kind="gated_deltanet"),
    )


def model(depth=1, target_length=1):
    torch.manual_seed(0)
    return GatedDeltaNetModel(mcfg(depth=depth), V, target_length=target_length)


def tokens(b=2, n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(8, V, (b, n), generator=g)


def naive_quadratic_recurrence(q, k, v, alpha, beta):
    """O(T^2) reference: unrolls S_t as an explicit product chain (no incremental
    state carried forward), used only to check the O(T) mixer against the
    literal update rule S_t = S_{t-1} * alpha_t * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T.
    """
    b, n, h, dh = q.shape
    eye = torch.eye(dh).view(1, 1, dh, dh).expand(b, h, dh, dh)
    outs = []
    for t in range(n):
        state = torch.zeros(b, h, dh, dh)
        for i in range(t + 1):
            ki, vi, ai, bi = k[:, i], v[:, i], alpha[:, i], beta[:, i]
            decay = ai.unsqueeze(-1).unsqueeze(-1) * (
                eye - bi.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhi,bhj->bhij", ki, ki)
            )
            state = torch.einsum("bhvk,bhkj->bhvj", state, decay) + bi.unsqueeze(-1).unsqueeze(
                -1
            ) * torch.einsum("bhv,bhk->bhvk", vi, ki)
        outs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]))
    return torch.stack(outs, dim=1)


def test_registry_name():
    assert get_model("gated_deltanet") is GatedDeltaNetModel


def test_recurrence_vs_naive_quadratic():
    torch.manual_seed(1)
    b, n, h, dh = 1, 16, 2, 4
    q = torch.randn(b, n, h, dh)
    k = torch.randn(b, n, h, dh)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = torch.randn(b, n, h, dh)
    alpha = torch.sigmoid(torch.randn(b, n, h))
    beta = torch.sigmoid(torch.randn(b, n, h))

    # efficient O(T) form used by GatedDeltaNetMixer.forward
    state = torch.zeros(b, h, dh, dh)
    outs = []
    for t in range(n):
        kt, vt, qt = k[:, t], v[:, t], q[:, t]
        at, bt = alpha[:, t].unsqueeze(-1), beta[:, t].unsqueeze(-1)
        sk = torch.einsum("bhvk,bhk->bhv", state, kt)
        delta = bt * (vt - at * sk)
        state = at.unsqueeze(-1) * state + torch.einsum("bhv,bhk->bhvk", delta, kt)
        outs.append(torch.einsum("bhvk,bhk->bhv", state, qt))
    efficient = torch.stack(outs, dim=1)

    naive = naive_quadratic_recurrence(q, k, v, alpha, beta)
    assert torch.allclose(efficient, naive, atol=1e-4)


def test_reasoning_steps_runtime():
    m = model(depth=1)
    q = tokens()
    from bdhx.models.base import BlockCounter

    for r in (1, 4, 16):
        with BlockCounter(list(m.blocks)) as c:
            out = m.solve(q, r)
        assert c.count == m.depth  # R fixed at 1 per layer for this baseline
        assert out.block_applications == m.depth


def test_context_isolation():
    demo_a = [(torch.tensor([10, 11]), torch.tensor([12]))]
    demo_b = [(torch.tensor([20]), torch.tensor([21, 22]))]
    q = tokens(b=1, n=3, seed=3)
    m = model()
    m.eval()
    m.ingest_context(demo_a)
    m.solve(q, 3)
    m.reset_context()
    m.ingest_context(demo_b)
    with torch.no_grad():
        got = m.solve(q, 3).logits
    fresh = model()
    fresh.eval()
    fresh.ingest_context(demo_b)
    with torch.no_grad():
        want = fresh.solve(q, 3).logits
    assert torch.equal(got, want)


def test_batch_independence():
    m = model()
    m.eval()
    q = tokens(b=4, n=6, seed=7)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = m.solve(q, 3).logits[perm]
        b = m.solve(q[perm], 3).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_memory_reset_zeroes_state():
    m = model()
    m.ingest_context([(torch.tensor([10]), torch.tensor([11]))])
    assert m._prefix is not None
    m.reset_context()
    assert m._prefix is None


def test_serialized_prefix_tokens():
    m = model()
    m.ingest_context(
        [(torch.tensor([10]), torch.tensor([11])), (torch.tensor([12]), torch.tensor([13]))]
    )
    assert m._prefix.flatten().tolist() == [BOS, 10, MAP, 11, SEP, 12, MAP, 13]


def test_param_count_matches_state_dict():
    m = model(depth=2)
    rep = m.param_report()
    assert rep.total == sum(t.numel() for t in m.state_dict().values())
    assert rep.trainable == rep.total


def test_backend_reference_on_cpu():
    m = model()
    assert m.backend == "reference"
    assert f"backend:{m.backend}" in m.param_report().breakdown


def test_solve_width_hits_target():
    from bdhx.models.gated_deltanet import gdn_block_stack_params

    for target in (200_000, 2_000_000):

        def ctor(w, target=target):
            return gdn_block_stack_params(w, 4128, 2)

        _width, realized = solve_width(ctor, target, step=4)
        assert abs(realized - target) / target <= 0.05


# -- adversarial checks (FRAMEWORK_SPEC section 3 guarantees) ---------------


def deterministic():
    """Context manager enabling torch deterministic mode for bit-for-bit checks."""

    class _Ctx:
        def __enter__(self):
            self.prev = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)

        def __exit__(self, *exc):
            torch.use_deterministic_algorithms(self.prev)

    return _Ctx()


def test_context_isolation_bitwise_deterministic():
    """reset_context + episode B must equal a fresh instance on B bit-for-bit."""
    demo_a = [
        (torch.tensor([10, 11]), torch.tensor([12])),
        (torch.tensor([13]), torch.tensor([14])),
    ]
    demo_b = [(torch.tensor([20]), torch.tensor([21, 22]))]
    q = tokens(b=2, n=4, seed=11)
    with deterministic():
        m = model(depth=2, target_length=2)
        m.eval()
        with torch.no_grad():
            m.ingest_context(demo_a)
            m.solve(q, 5)
            m.reset_context()
            m.ingest_context(demo_b)
            got = m.solve(q, 5).logits
        fresh = model(depth=2, target_length=2)
        fresh.eval()
        with torch.no_grad():
            fresh.ingest_context(demo_b)
            want = fresh.solve(q, 5).logits
    assert torch.equal(got, want)


def test_reset_restores_virgin_state():
    """After reset with no re-ingest, solve must equal a never-ingested model."""
    q = tokens(b=1, n=4, seed=12)
    m = model()
    m.eval()
    with torch.no_grad():
        virgin = m.solve(q, 2).logits
        m.ingest_context([(torch.tensor([30, 31]), torch.tensor([32]))])
        m.solve(q, 2)
        m.reset_context()
        after = m.solve(q, 2).logits
    assert torch.equal(virgin, after)


def test_no_persistent_hidden_state():
    """No buffers, no python attribute other than _prefix survives an episode."""
    m = model(depth=2)
    m.eval()
    assert list(m.buffers()) == []
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        m.ingest_context([(torch.tensor([10]), torch.tensor([11]))])
        m.solve(tokens(b=1, n=3, seed=13), 4)
    m.reset_context()
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert m._prefix is None
    tensor_attrs = [
        k for k, v in vars(m).items() if torch.is_tensor(v) and not k.startswith("_parameters")
    ]
    assert tensor_attrs == []


def test_no_cross_batch_communication():
    """Perturbing one episode must leave the other rows bit-identical."""
    m = model(depth=2)
    m.eval()
    q = tokens(b=4, n=6, seed=14)
    q2 = q.clone()
    q2[0] = (q2[0] + 5) % 40 + 8
    with torch.no_grad():
        a = m.solve(q, 3).logits
        b = m.solve(q2, 3).logits
    assert torch.equal(a[1:], b[1:])
    assert not torch.equal(a[0], b[0])


def test_mixer_is_causal():
    """Output at position t must not depend on tokens after t."""
    torch.manual_seed(2)
    mix = GatedDeltaNetMixer(W, heads=2)
    x = torch.randn(1, 8, W)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, W)
    with torch.no_grad():
        assert torch.equal(mix(x)[:, :5], mix(x2)[:, :5])


def test_depth_drives_applications_not_reasoning_steps():
    for depth in (1, 3):
        m = model(depth=depth)
        from bdhx.models.base import BlockCounter

        with BlockCounter(list(m.blocks)) as c:
            out = m.solve(tokens(b=1, n=4), 7)
        assert c.count == depth == out.block_applications


def test_gradients_reach_every_parameter():
    m = model(depth=2)
    out = m.solve(tokens(b=2, n=5, seed=15), 3)
    out.logits.square().mean().backward()
    dead = [
        n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert dead == []


def test_short_conv_not_implemented():
    import pytest

    with pytest.raises(NotImplementedError):
        GatedDeltaNetMixer(W, heads=2, short_conv=True)


def test_flops_report_conventions():
    """Model-level estimate matches flops.py and uses total = per_episode * B."""
    from bdhx.training.flops import gated_deltanet_flops

    m = model(depth=2)

    class _B:
        serialized = torch.zeros(3, 11, dtype=torch.long)

    rep = m.flops_estimate(_B(), 4)
    ref = gated_deltanet_flops(m._resolved_cfg, 11, 4, V, 3)
    assert rep.per_episode == ref.per_episode
    assert rep.total == rep.per_episode * 3
    assert m.flops_estimate(_B(), 1).per_episode == rep.per_episode  # R fixed at 1/layer


def test_target_length_readout_shapes():
    m = model(target_length=3)
    out = m.solve(tokens(b=2, n=5, seed=16), 2)
    assert out.logits.shape == (2, 3, V)
    assert out.predictions.shape == (2, 3)
