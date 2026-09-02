"""TASK_SUITE_SPEC section 5 tests for T1-T4: binding, overwrite, distractors, contradict."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bdhx.tasks.base import pad_and_batch
from bdhx.tasks.binding import BindingTask
from bdhx.tasks.contradict import ContradictTask
from bdhx.tasks.distractors import DistractorsTask
from bdhx.tasks.overwrite import OverwriteTask

TASKS = {
    "binding": (BindingTask(), {"n_bindings": 6}),
    "overwrite": (OverwriteTask(), {"n_bindings": 6, "n_overwrites": 2, "gap": 2}),
    "distractors": (DistractorsTask(), {"n_bindings": 6, "distractor_ratio": 1}),
    "contradict": (ContradictTask(), {"n_bindings": 6, "n_conflicts": 2}),
}


def _episodes_equal(a, b) -> bool:
    if len(a.demonstrations) != len(b.demonstrations):
        return False
    for (ai, ao), (bi, bo) in zip(a.demonstrations, b.demonstrations):
        if not (torch.equal(ai, bi) and torch.equal(ao, bo)):
            return False
    return torch.equal(a.query, b.query) and torch.equal(a.target, b.target)


@pytest.mark.parametrize("name", TASKS.keys())
def test_determinism(name):
    task, diff = TASKS[name]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    ep1 = task.sample(rng1, diff)
    ep2 = task.sample(rng2, diff)
    assert _episodes_equal(ep1, ep2)


def test_fresh_randomness_binding():
    task = BindingTask()
    rng = np.random.default_rng(0)
    mappings = []
    for _ in range(1000):
        ep = task.sample(rng, {"n_bindings": 4})
        mapping = {int(k.item()): int(v.item()) for k, v in ep.demonstrations}
        mappings.append(mapping)
    # no key maps to the same value across every episode it appears in
    key_to_values: dict[int, set[int]] = {}
    for mapping in mappings:
        for k, v in mapping.items():
            key_to_values.setdefault(k, set()).add(v)
    varying = [k for k, vs in key_to_values.items() if len(vs) > 1]
    assert len(varying) > 0


@pytest.mark.parametrize("name", TASKS.keys())
def test_splits_disjoint(name):
    task, _ = TASKS[name]
    train = task.train_difficulties()
    ev = task.eval_difficulties()
    assert set(ev.keys()) == {"interp", "mild", "strong"}
    train_set = {tuple(sorted(d.items())) for d in train}
    mild_set = {tuple(sorted(d.items())) for d in ev["mild"]}
    strong_set = {tuple(sorted(d.items())) for d in ev["strong"]}
    assert train_set.isdisjoint(mild_set)
    assert train_set.isdisjoint(strong_set)
    assert mild_set.isdisjoint(strong_set)


@pytest.mark.parametrize("name", TASKS.keys())
def test_serialize_roundtrip(name):
    task, diff = TASKS[name]
    rng = np.random.default_rng(1)
    ep = task.sample(rng, diff)
    ser = task.serialize(ep)
    back = task.parse_serialized(
        ser, difficulty=ep.difficulty, split=ep.split, episode_id=ep.episode_id
    )
    assert _episodes_equal(ep, back)


@pytest.mark.parametrize("name", TASKS.keys())
def test_no_state_leak(name):
    task, diff = TASKS[name]
    rng = np.random.default_rng(7)
    _ = task.sample(rng, diff)
    second_from_reused = task.sample(rng, diff)

    fresh_rng = np.random.default_rng(7)
    _ = task.sample(fresh_rng, diff)
    second_from_fresh = task.sample(fresh_rng, diff)

    assert _episodes_equal(second_from_reused, second_from_fresh)


def test_target_correctness_binding():
    task = BindingTask()
    rng = np.random.default_rng(3)
    for _ in range(50):
        ep = task.sample(rng, {"n_bindings": 8})
        mapping = {int(k.item()): int(v.item()) for k, v in ep.demonstrations}
        assert mapping[int(ep.query.item())] == int(ep.target.item())


def test_target_correctness_overwrite():
    task = OverwriteTask()
    rng = np.random.default_rng(4)
    for _ in range(50):
        ep = task.sample(rng, {"n_bindings": 8, "n_overwrites": 3, "gap": 2})
        latest = {}
        for k, v in ep.demonstrations:
            latest[int(k.item())] = int(v.item())
        assert latest[int(ep.query.item())] == int(ep.target.item())


def test_overwrite_target_is_latest():
    task = OverwriteTask()
    rng = np.random.default_rng(5)
    saw_stale = False
    for _ in range(200):
        ep = task.sample(rng, {"n_bindings": 8, "n_overwrites": 4, "gap": 1})
        latest = {}
        for k, v in ep.demonstrations:
            latest[int(k.item())] = int(v.item())
        assert latest[int(ep.query.item())] == int(ep.target.item())
        stale = ep.extras.get("stale_target")
        if stale is not None:
            saw_stale = True
            assert stale != int(ep.target.item())
    assert saw_stale


def test_target_correctness_distractors():
    task = DistractorsTask()
    rng = np.random.default_rng(6)
    for _ in range(50):
        ep = task.sample(rng, {"n_bindings": 6, "distractor_ratio": 2})
        mapping = {int(k.item()): int(v.item()) for k, v in ep.demonstrations}
        assert mapping[int(ep.query.item())] == int(ep.target.item())


def test_distractors_do_not_leak_into_query():
    task = DistractorsTask()
    rng = np.random.default_rng(8)
    for _ in range(50):
        ep = task.sample(rng, {"n_bindings": 6, "distractor_ratio": 1})
        dist_values = set(ep.extras.get("distractor_values", []))
        assert int(ep.target.item()) not in dist_values


def test_target_correctness_contradict_recency():
    task = ContradictTask(convention="recency")
    rng = np.random.default_rng(9)
    for _ in range(50):
        ep = task.sample(rng, {"n_bindings": 8, "n_conflicts": 3})
        latest = {}
        for k, v in ep.demonstrations:
            latest[int(k.item())] = int(v.item())
        assert latest[int(ep.query.item())] == int(ep.target.item())


def test_target_correctness_contradict_majority():
    task = ContradictTask(convention="majority")
    rng = np.random.default_rng(10)
    for _ in range(50):
        ep = task.sample(rng, {"n_bindings": 8, "n_conflicts": 3})
        counts: dict[int, dict[int, int]] = {}
        for k, v in ep.demonstrations:
            ki, vi = int(k.item()), int(v.item())
            counts.setdefault(ki, {}).setdefault(vi, 0)
            counts[ki][vi] += 1
        qk = int(ep.query.item())
        majority = max(counts[qk].items(), key=lambda kv: kv[1])[0]
        assert majority == int(ep.target.item())


def test_batching_smoke():
    for task, diff in TASKS.values():
        rng = np.random.default_rng(11)
        episodes = [task.sample(rng, diff) for _ in range(4)]
        batch = pad_and_batch(episodes, task=task)
        assert len(batch) == 4
