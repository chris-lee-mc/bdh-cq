"""TASK_SUITE_SPEC section 5 tests for compose (T5), order (T8), nested (T9)."""

from __future__ import annotations

import numpy as np
import pytest

from bdhx.tasks.compose import ComposeTask
from bdhx.tasks.nested import PRIMITIVES, NestedTask, apply_program
from bdhx.tasks.order import GT, LT, OrderTask
from bdhx.tasks.vocab import ANSWER, QUERY, is_symbol

TASKS = {
    "compose": ComposeTask,
    "order": OrderTask,
    "nested": NestedTask,
}


def _all_difficulties(task):
    diffs = list(task.train_difficulties())
    for group in task.eval_difficulties().values():
        diffs.extend(group)
    return diffs


# -- test_determinism --------------------------------------------------------


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


# -- test_fresh_randomness ---------------------------------------------------


def test_fresh_randomness_compose():
    task = ComposeTask()
    diff = {"depth": 2}
    rng = np.random.default_rng(0)
    xs = [int(task.sample(rng, diff).query[-1]) for _ in range(200)]
    assert len(set(xs)) > 1


def test_fresh_randomness_order():
    task = OrderTask()
    diff = {"n_items": 6, "query_type": "pair", "hops": 1}
    rng = np.random.default_rng(0)
    firsts = [int(task.sample(rng, diff).demonstrations[0][0][0]) for _ in range(200)]
    assert len(set(firsts)) > 1


def test_fresh_randomness_nested():
    task = NestedTask()
    diff = {"depth": 2}
    rng = np.random.default_rng(0)
    programs = [tuple(task.sample(rng, diff).extras["program"]) for _ in range(200)]
    assert len(set(programs)) > 1


# -- test_splits_disjoint ----------------------------------------------------


def _difficulty_keys(diffs, key):
    return {d[key] for d in diffs}


@pytest.mark.parametrize("name,key", [("compose", "depth"), ("nested", "depth")])
def test_splits_disjoint(name, key):
    task = TASKS[name]()
    train = _difficulty_keys(task.train_difficulties(), key)
    evald = task.eval_difficulties()
    mild = _difficulty_keys(evald["mild"], key)
    strong = _difficulty_keys(evald["strong"], key)
    assert train.isdisjoint(mild)
    assert train.isdisjoint(strong)
    assert mild.isdisjoint(strong)


def test_splits_disjoint_order():
    task = OrderTask()
    train = {
        (d["n_items"], d["hops"]) for d in task.train_difficulties() if d["query_type"] == "pair"
    }
    evald = task.eval_difficulties()
    mild = {(d["n_items"], d["hops"]) for d in evald["mild"] if d["query_type"] == "pair"}
    strong = {(d["n_items"], d["hops"]) for d in evald["strong"] if d["query_type"] == "pair"}
    assert train.isdisjoint(mild)
    assert train.isdisjoint(strong)
    assert mild.isdisjoint(strong)


# -- test_serialize_roundtrip -------------------------------------------------


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


# -- test_no_state_leak -------------------------------------------------------


@pytest.mark.parametrize("name", TASKS)
def test_no_state_leak(name):
    task = TASKS[name]()
    diff = _all_difficulties(task)[0]
    rng_shared = np.random.default_rng(3)
    _ = task.sample(rng_shared, diff)
    ep_second_shared = task.sample(rng_shared, diff)

    rng_a = np.random.default_rng(3)
    _ = task.sample(rng_a, diff)
    rng_b = np.random.default_rng(3)  # fresh generator advanced identically
    _ = task.sample(rng_b, diff)
    ep_second_fresh = task.sample(rng_b, diff)

    assert ep_second_shared.query.tolist() == ep_second_fresh.query.tolist()
    assert ep_second_shared.target.tolist() == ep_second_fresh.target.tolist()


# -- test_target_correctness (brute-force reference solvers) -----------------


def _brute_compose(episode):
    """Reference: rebuild the domain chain from demonstrations and walk it by hand."""
    domains = episode.extras["domains"]
    fns = episode.extras["fns"]
    x = int(episode.query[-1])
    depth = int(episode.difficulty["depth"])
    cur = x
    for i in range(depth):
        idx = int(np.where(domains[i] == cur)[0][0])
        cur = int(fns[i][idx])
    return cur


def test_target_correctness_compose():
    task = ComposeTask()
    rng = np.random.default_rng(11)
    for diff in _all_difficulties(task):
        for _ in range(5):
            ep = task.sample(rng, diff)
            assert _brute_compose(ep) == int(ep.target[0])
            # every demo pair must actually be consistent with some f_i
            fns, domains = ep.extras["fns"], ep.extras["domains"]
            for inp, out in ep.demonstrations:
                a, b = int(inp[0]), int(out[0])
                found = False
                for i in range(len(fns)):
                    where = np.where(domains[i] == a)[0]
                    if len(where) and int(fns[i][where[0]]) == b:
                        found = True
                        break
                assert found


# -- test_solvable_from_demonstrations ---------------------------------------
#
# `test_target_correctness_compose` above checks the target against the hidden
# bijections in `extras`, which no model sees.  These check the property a model
# actually needs: that the answer can be derived from the demonstrations.


def _chain_from_demonstrations(episode):
    """Walk the query forward through the demonstrated pairs; None if it breaks."""
    demo = {int(inp[0]): int(out[0]) for inp, out in episode.demonstrations}
    cur = int(episode.query[-1])
    for _ in range(int(episode.difficulty["depth"])):
        if cur not in demo:
            return None
        cur = demo[cur]
    return cur


def test_compose_every_episode_is_solvable_from_its_demonstrations():
    task = ComposeTask()
    rng = np.random.default_rng(17)
    for diff in _all_difficulties(task):
        for _ in range(50):
            ep = task.sample(rng, diff)
            assert _chain_from_demonstrations(ep) == int(ep.target[0]), diff


def test_compose_demonstrations_are_distinct_per_function():
    """Sampling without replacement: no function wastes a demonstration on a repeat."""
    task = ComposeTask()
    rng = np.random.default_rng(19)
    for depth in (1, 2, 4, 8):
        ep = task.sample(rng, {"depth": depth})
        inputs = [int(inp[0]) for inp, _ in ep.demonstrations]
        assert len(inputs) == len(set(inputs))
        assert len(inputs) == depth * task.n_examples_per_fn


def test_compose_legacy_distribution_is_mostly_unsolvable():
    """The pre-fix generator, kept behind a flag, and why the flag defaults to on.

    A1's parameters explicitly: 4 examples of a domain of 8, drawn with
    replacement per function, query drawn independently.
    """
    task = ComposeTask(n_examples_per_fn=4, domain_size=8, guarantee_solvable=False)
    rng = np.random.default_rng(23)
    solved = sum(
        _chain_from_demonstrations(task.sample(rng, {"depth": 2})) is not None for _ in range(500)
    )
    assert solved / 500 < 0.3  # analytic value at depth 2 is 0.171


def test_compose_n_examples_never_exceeds_the_domain():
    task = ComposeTask(n_examples_per_fn=32, domain_size=8)
    rng = np.random.default_rng(29)
    ep = task.sample(rng, {"depth": 2})
    assert len(ep.demonstrations) == 2 * 8


def _brute_order(items, x, y):
    idx = {int(v): i for i, v in enumerate(items)}
    return LT if idx[x] < idx[y] else GT


def test_target_correctness_order():
    task = OrderTask()
    rng = np.random.default_rng(5)
    for diff in [d for d in _all_difficulties(task) if d["query_type"] == "pair"]:
        for _ in range(5):
            ep = task.sample(rng, diff)
            # reconstruct the chain from the (shuffled) adjacency demonstrations and
            # walk it starting from the node with no predecessor
            succ = {int(inp[0]): int(inp[1]) for inp, _ in ep.demonstrations}
            starts = set(succ.keys()) - set(succ.values())
            assert len(starts) == 1
            cur = next(iter(starts))
            chain = [cur]
            while cur in succ:
                cur = succ[cur]
                chain.append(cur)
            x, y = int(ep.query[1]), int(ep.query[2])
            assert _brute_order(chain, x, y) == int(ep.target[0])


def test_target_correctness_order_sort():
    task = OrderTask()
    diff = {"n_items": 6, "query_type": "sort", "hops": 1}
    rng = np.random.default_rng(9)
    for _ in range(5):
        ep = task.sample(rng, diff)
        succ = {int(inp[0]): int(inp[1]) for inp, _ in ep.demonstrations}
        starts = set(succ.keys()) - set(succ.values())
        cur = next(iter(starts))
        chain = [cur]
        while cur in succ:
            cur = succ[cur]
            chain.append(cur)
        assert chain == [int(t) for t in ep.target]


def test_target_correctness_nested():
    task = NestedTask()
    rng = np.random.default_rng(13)
    for diff in _all_difficulties(task):
        for _ in range(5):
            ep = task.sample(rng, diff)
            program, x = ep.extras["program"], ep.extras["input"]
            ref = list(x)
            for tag in program:
                ref = PRIMITIVES[tag](ref)
            assert ref == [int(t) for t in ep.target]
            assert apply_program(program, x) == ref


# -- symbol hygiene / structural sanity ---------------------------------------


def test_compose_query_symbol_is_from_pool():
    task = ComposeTask()
    rng = np.random.default_rng(1)
    ep = task.sample(rng, {"depth": 2})
    assert is_symbol(int(ep.query[-1]))
    assert is_symbol(int(ep.target[0]))


def test_nested_program_uses_five_primitives():
    task = NestedTask()
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(50):
        ep = task.sample(rng, {"depth": 3})
        seen.update(ep.extras["program"])
    assert seen == set(PRIMITIVES.keys())


# -- query-dependence (RESULTS.md A1a, finding 5 of the pre-launch review) ----


def _complete_paths(episode):
    """Every length-d path through the demonstration graph, from any start."""
    demo = {int(inp[0]): int(out[0]) for inp, out in episode.demonstrations}
    depth = int(episode.difficulty["depth"])
    starts = set(demo) - set(demo.values())
    paths = []
    for start in starts:
        cur, ok = start, True
        for _ in range(depth):
            if cur not in demo:
                ok = False
                break
            cur = demo[cur]
        if ok:
            paths.append(cur)
    return paths


def test_compose_answer_cannot_be_read_off_without_the_query():
    """Regression: the first `guarantee_solvable` made the query redundant.

    Demonstrating each function on the chain input plus unrelated distractors
    leaves the distractors dead-ending, so the chain was the only complete
    length-d path -- unique in 99.4 percent of depth-8 episodes, so a model
    could emit the answer without reading x at all. Each function is now
    demonstrated on the images of the previous one's inputs, so there are as
    many complete paths as demonstrations per function.
    """
    task = ComposeTask()
    rng = np.random.default_rng(31)
    for depth in (1, 2, 4, 8):
        for _ in range(20):
            ep = task.sample(rng, {"depth": depth})
            paths = _complete_paths(ep)
            assert len(paths) == task.n_examples_per_fn, (depth, len(paths))
            assert int(ep.target[0]) in paths


def test_compose_guess_floor_is_one_over_the_demonstrated_paths():
    """The null to score against, and it is not 1/vocab."""
    task = ComposeTask()
    assert task.guess_floor(8) == pytest.approx(1.0 / 8)
    assert ComposeTask(n_examples_per_fn=4).guess_floor(8) == pytest.approx(0.25)
    rng = np.random.default_rng(37)
    for depth in (1, 8):
        endpoints = {
            len(set(_complete_paths(task.sample(rng, {"depth": depth})))) for _ in range(50)
        }
        assert endpoints == {task.n_examples_per_fn}
