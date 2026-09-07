"""Guards on tools/reeval_checkpoint.py (RESULTS.md section A5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.reeval_checkpoint import stored_rows


def _run_dir(tmp_path: Path, evaluations: list[dict]) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"evaluations": evaluations}))
    return d


def _row(step: int, split: str, r: int, em: float, distance: int = 1) -> dict:
    return {
        "step": step,
        "split": split,
        "reasoning_steps": r,
        "exact_match": em,
        "difficulty": {"length": 12, "distance": distance, "n_sources": 1, "dim": 1},
    }


def test_stored_rows_takes_the_last_duplicate_not_the_first(tmp_path):
    """A resumed run APPENDS evaluation rows rather than replacing them.

    This is the bug class that produced wrong numbers earlier in this project
    (`_dedupe_evaluations` in the aggregator exists for it). A re-evaluation
    tool that compared against the FIRST row would silently report a delta
    against a stale pre-resume number, which looks exactly like a real change.
    """
    d = _run_dir(tmp_path, [_row(40000, "interp", 8, 0.11), _row(40000, "interp", 8, 0.99)])
    assert stored_rows(d, 40000, "interp")[
        (json.dumps({"length": 12, "distance": 1, "n_sources": 1, "dim": 1}, sort_keys=True), 8)
    ] == pytest.approx(0.99)


def test_stored_rows_separates_difficulties_and_steps(tmp_path):
    """Keys must carry the difficulty and respect the step/split filter.

    Pooling two difficulties under one key, or letting a mid-training
    checkpoint's row answer for the final one, would make the reproduction
    check compare unlike things and pass when it should fail.
    """
    d = _run_dir(
        tmp_path,
        [
            _row(40000, "interp", 8, 0.99, distance=1),
            _row(40000, "interp", 8, 0.50, distance=2),
            _row(2500, "interp", 8, 0.00, distance=1),
            _row(40000, "strong", 8, 0.10, distance=1),
        ],
    )
    got = stored_rows(d, 40000, "interp")
    assert len(got) == 2
    k1 = json.dumps({"length": 12, "distance": 1, "n_sources": 1, "dim": 1}, sort_keys=True)
    k2 = json.dumps({"length": 12, "distance": 2, "n_sources": 1, "dim": 1}, sort_keys=True)
    assert got[(k1, 8)] == pytest.approx(0.99)
    assert got[(k2, 8)] == pytest.approx(0.50)


def test_stored_rows_is_empty_when_nothing_matches(tmp_path):
    """An unreported R must yield no stored value, not a spurious 0.0.

    The whole point of the tool is evaluating R values the run never reported;
    a default of 0.0 there would print a fake delta against a real result.
    """
    d = _run_dir(tmp_path, [_row(40000, "interp", 8, 0.99)])
    assert (
        json.dumps({"length": 12, "distance": 1, "n_sources": 1, "dim": 1}, sort_keys=True),
        6,
    ) not in stored_rows(d, 40000, "interp")
