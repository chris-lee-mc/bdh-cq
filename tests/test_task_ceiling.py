"""Tests for the pre-sweep task-ceiling gate (`tools/task_ceiling.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import task_ceiling

from bdhx.tasks.compose import ComposeTask
from bdhx.tasks.order import OrderTask


def test_compose_oracle_matches_the_target_after_the_fix():
    task = ComposeTask()
    rng = np.random.default_rng(3)
    for depth in (1, 2, 4, 8):
        ep = task.sample(rng, {"depth": depth})
        assert task_ceiling._compose_oracle(ep) == int(ep.target[0])


def test_compose_measure_reports_the_legacy_ceiling():
    """The pre-fix generator is what the gate exists to catch."""
    legacy = ComposeTask(n_examples_per_fn=4, domain_size=8, guarantee_solvable=False)
    stats = task_ceiling.measure(legacy, {"depth": 2}, n=400, seed=5)
    assert stats["oracle_solvable"] < 0.3  # analytic 0.171
    fixed = task_ceiling.measure(ComposeTask(), {"depth": 2}, n=100, seed=5)
    assert fixed["oracle_solvable"] == 1.0


def test_structural_targets_are_not_counted_as_missing():
    """`order`'s pair queries answer with LT/GT, which are fixed tokens."""
    stats = task_ceiling.measure(
        OrderTask(), {"n_items": 6, "query_type": "pair", "hops": 1}, n=100, seed=7
    )
    assert stats["target_in_context"] == 1.0


def test_gate_passes_on_the_fixed_generator(capsys):
    assert (
        task_ceiling.main(["--task", "compose", "--episodes", "50", "--min-train-solvable", "0.9"])
        == 0
    )
    assert "compose" in capsys.readouterr().out


def test_gate_fails_when_a_train_difficulty_is_unanswerable(monkeypatch, capsys):
    monkeypatch.setattr(
        task_ceiling,
        "get_task",
        lambda name: lambda: ComposeTask(n_examples_per_fn=4, guarantee_solvable=False),
    )
    assert (
        task_ceiling.main(["--task", "compose", "--episodes", "200", "--min-train-solvable", "0.9"])
        == 1
    )
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["binding", "propagate", "nested", "contradict"])
def test_every_other_task_answers_its_own_train_difficulties(name):
    assert (
        task_ceiling.main(["--task", name, "--episodes", "50", "--min-train-solvable", "0.99"]) == 0
    )
