"""TASK_SUITE_SPEC section 5 tests for propagate (T6), copy (T7), legacy
adapters, and tools/generate_tasks.py determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bdhx.tasks.base import Episode
from bdhx.tasks.copy import CopyTask
from bdhx.tasks.legacy import (
    LegacyCopyTask,
    LegacyNestingTask,
    LegacyOrderTask,
    LegacyPropagationTask,
)
from bdhx.tasks.propagate import PropagateTask
from bdhx.tasks.vocab import ANSWER, QUERY

TASKS = {
    "propagate": PropagateTask,
    "copy": CopyTask,
}

LEGACY_TASKS = {
    "legacy_propagation": LegacyPropagationTask,
    "legacy_copy": LegacyCopyTask,
    "legacy_order": LegacyOrderTask,
    "legacy_nesting": LegacyNestingTask,
}


def _all_difficulties(task):
    diffs = list(task.train_difficulties())
    for group in task.eval_difficulties().values():
        diffs.extend(group)
    return diffs


# -- test_determinism ---------------------------------------------------------


@pytest.mark.parametrize("name", TASKS)
def test_determinism(name):
    task = TASKS[name]()
    for diff in _all_difficulties(task):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        ep1 = task.sample(rng1, diff)
        ep2 = task.sample(rng2, diff)
        assert ep1.query.tolist() == ep2.query.tolist()
        assert ep1.target.tolist() == ep2.target.tolist()
        assert len(ep1.demonstrations) == len(ep2.demonstrations)
        for (i1, o1), (i2, o2) in zip(ep1.demonstrations, ep2.demonstrations):
            assert i1.tolist() == i2.tolist()
            assert o1.tolist() == o2.tolist()


@pytest.mark.parametrize("name", LEGACY_TASKS)
def test_determinism_legacy(name):
    task = LEGACY_TASKS[name]()
    for diff in task.train_difficulties():
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        ep1 = task.sample(rng1, diff)
        ep2 = task.sample(rng2, diff)
        assert ep1.query.tolist() == ep2.query.tolist()
        assert ep1.target.tolist() == ep2.target.tolist()


# -- test_fresh_randomness -----------------------------------------------------


def test_fresh_randomness_propagate():
    task = PropagateTask()
    diff = {"length": 12, "distance": 2, "n_sources": 1, "dim": 1}
    rng = np.random.default_rng(0)
    fills = [int(task.sample(rng, diff).query[0]) for _ in range(200)]
    assert len(set(fills)) > 1


def test_fresh_randomness_copy():
    task = CopyTask()
    diff = {"length": 6}
    rng = np.random.default_rng(0)
    firsts = [int(task.sample(rng, diff).query[0]) for _ in range(200)]
    assert len(set(firsts)) > 1


# -- test_splits_disjoint -------------------------------------------------------


def test_splits_disjoint_propagate():
    task = PropagateTask()
    train = {d["distance"] for d in task.train_difficulties()}
    evald = task.eval_difficulties()
    mild = {d["distance"] for d in evald["mild"]}
    strong = {d["distance"] for d in evald["strong"]}
    assert train.isdisjoint(mild)
    assert train.isdisjoint(strong)
    assert mild.isdisjoint(strong)


def test_splits_disjoint_copy():
    task = CopyTask()
    train = {d["length"] for d in task.train_difficulties()}
    evald = task.eval_difficulties()
    mild = {d["length"] for d in evald["mild"]}
    strong = {d["length"] for d in evald["strong"]}
    assert train.isdisjoint(mild)
    assert train.isdisjoint(strong)
    assert mild.isdisjoint(strong)


@pytest.mark.parametrize("name", LEGACY_TASKS)
def test_splits_disjoint_legacy(name):
    task = LEGACY_TASKS[name]()
    train = {d["level"] for d in task.train_difficulties()}
    evald = task.eval_difficulties()
    mild = {d["level"] for d in evald["mild"]}
    strong = {d["level"] for d in evald["strong"]}
    assert train.isdisjoint(mild)
    assert train.isdisjoint(strong)


# -- test_serialize_roundtrip ---------------------------------------------------


@pytest.mark.parametrize("name", TASKS)
def test_serialize_roundtrip(name):
    task = TASKS[name]()
    for diff in task.train_difficulties():
        rng = np.random.default_rng(7)
        ep = task.sample(rng, diff)
        ser = task.serialize(ep)
        back = task.parse_serialized(
            ser, difficulty=ep.difficulty, split=ep.split, episode_id=ep.episode_id
        )
        assert len(back.demonstrations) == len(ep.demonstrations)
        for (a, b), (c, d) in zip(back.demonstrations, ep.demonstrations):
            assert a.tolist() == c.tolist()
            assert b.tolist() == d.tolist()
        assert back.query.tolist() == ep.query.tolist()
        assert back.target.tolist() == ep.target.tolist()
        assert QUERY in ser.tolist()
        assert ANSWER in ser.tolist()


# -- test_no_state_leak ----------------------------------------------------------


@pytest.mark.parametrize("name", TASKS)
def test_no_state_leak(name):
    task = TASKS[name]()
    diff = task.train_difficulties()[0]
    rng_shared = np.random.default_rng(3)
    _ = task.sample(rng_shared, diff)
    ep_second_shared = task.sample(rng_shared, diff)

    rng_b = np.random.default_rng(3)
    _ = task.sample(rng_b, diff)
    ep_second_fresh = task.sample(rng_b, diff)

    assert ep_second_shared.query.tolist() == ep_second_fresh.query.tolist()
    assert ep_second_shared.target.tolist() == ep_second_fresh.target.tolist()


# -- test_target_correctness (brute-force reference solvers) --------------------


def test_target_correctness_propagate():
    task = PropagateTask()
    rng = np.random.default_rng(11)
    for diff in _all_difficulties(task):
        for _ in range(5):
            ep = task.sample(rng, diff)
            target = ep.target.tolist()
            query = ep.query.tolist()
            length = len(target)
            for source_pos, wall_pos in ep.extras["segments"]:
                fill = query[source_pos]
                ref = list(query)
                for i in range(source_pos + 1, min(wall_pos, length)):
                    ref[i] = fill
                for i in range(source_pos, min(wall_pos + 1, length)):
                    assert target[i] == ref[i]


def test_target_correctness_copy():
    task = CopyTask()
    rng = np.random.default_rng(5)
    for diff in _all_difficulties(task):
        for _ in range(5):
            ep = task.sample(rng, diff)
            assert ep.target.tolist() == ep.query.tolist()
            for inp, out in ep.demonstrations:
                assert inp.tolist() == out.tolist()


# -- legacy smoke test ------------------------------------------------------------


@pytest.mark.parametrize("name", LEGACY_TASKS)
def test_legacy_smoke(name):
    task = LEGACY_TASKS[name]()
    rng = np.random.default_rng(0)
    diff = task.train_difficulties()[0]
    ep = task.sample(rng, diff)
    assert isinstance(ep, Episode)
    assert ep.query.numel() > 0
    assert ep.target.numel() > 0
    assert len(ep.demonstrations) > 0
    ser = task.serialize(ep)
    back = task.parse_serialized(
        ser, difficulty=ep.difficulty, split=ep.split, episode_id=ep.episode_id
    )
    assert back.query.tolist() == ep.query.tolist()
    assert back.target.tolist() == ep.target.tolist()
    scores = task.score(ep.target, ep)
    assert scores["exact_match"] == 1.0


# -- generate_tasks determinism --------------------------------------------------


def test_generate_tasks_determinism(tmp_path):
    from tools.generate_tasks import main

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    main(
        ["--task", "binding", "--seed", "17", "--n_train", "6", "--n_eval", "4", "--out", str(out1)]
    )
    main(
        ["--task", "binding", "--seed", "17", "--n_train", "6", "--n_eval", "4", "--out", str(out2)]
    )

    for fname in ("train.npz", "interp.npz", "mild.npz", "strong.npz"):
        b1 = (out1 / fname).read_bytes()
        b2 = (out2 / fname).read_bytes()
        assert b1 == b2

    manifest1 = json.loads((out1 / "manifest.json").read_text())
    manifest2 = json.loads((out2 / "manifest.json").read_text())
    assert manifest1["task"] == "binding"
    assert manifest1["split_counts"] == {"train": 6, "interp": 4, "mild": 4, "strong": 4}
    assert manifest1 == manifest2


def test_generate_tasks_load_eval_episodes(tmp_path):
    from bdhx.tasks.cache import load_eval_episodes
    from tools.generate_tasks import main

    out = tmp_path / "data" / "binding_s3"
    main(["--task", "binding", "--seed", "3", "--n_train", "5", "--n_eval", "5", "--out", str(out)])

    eps = load_eval_episodes("binding", 3, "interp", 3, root=tmp_path / "data")
    assert len(eps) == 3
    for ep in eps:
        assert ep.split == "interp"

    with pytest.raises(FileNotFoundError):
        load_eval_episodes("binding", 999, "interp", 3, root=tmp_path / "data")

    with pytest.raises(ValueError):
        load_eval_episodes("binding", 3, "interp", 100, root=tmp_path / "data")
