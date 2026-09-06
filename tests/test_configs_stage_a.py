"""The cross-sweep claims Stage A configs make about each other.

a4_convergence.yaml omits its own `plain` arm and reuses a1_first_experiment's
three bdh_cq/propagate seeds as the reference arm -- 3 jobs of GPU time, and a
comparison that is only valid while the two configs still describe the same
training run. Nothing else in the pipeline checks that: the sweeps are separate
files, the runs are already on disk, and a drifting field would silently turn
the reference arm into a different experiment plotted as if it were the same.
"""

from __future__ import annotations

import yaml

from bdhx.config import (
    Config,
    apply_dotted_overrides,
    config_hash,
    hashable_yaml,
    load_raw,
)

A1 = "configs/stage_a/a1_first_experiment.yaml"
A4 = "configs/stage_a/a4_convergence.yaml"

# Stamped per sweep by tools/generate_sweep.py, hashed, and irrelevant to what
# the model trains on.
IDENTITY_FIELDS = ("name", "stage", "tags")


def _resolved(sweep_path: str, kind: str, task: str = "propagate") -> Config:
    with open(sweep_path) as fh:
        sweep = yaml.safe_load(fh)
    raw = apply_dotted_overrides(load_raw(sweep["base"]), dict(sweep["overrides"]))
    raw = apply_dotted_overrides(raw, {"model.recurrence.kind": kind, "task.name": task})
    raw["experiment"] = {"name": sweep["sweep"]["name"], "stage": sweep["sweep"]["stage"]}
    return Config.model_validate(raw)


def test_a4_plain_arm_matches_a1_except_for_sweep_identity():
    """A4's reference arm must be A1's run in every field that trains the model."""
    a1 = _resolved(A1, "plain")
    a4 = _resolved(A4, "plain")

    d1 = a1.model_dump(mode="json")
    d4 = a4.model_dump(mode="json")
    for field in IDENTITY_FIELDS:
        d1["experiment"].pop(field, None)
        d4["experiment"].pop(field, None)
    # Observation-only and already excluded from the config hash; A4 sets it so
    # the onset table has an R=32 curve, which A1 could not produce.
    d1["evaluation"].pop("intermediate_reasoning_steps", None)
    d4["evaluation"].pop("intermediate_reasoning_steps", None)

    assert d1 == d4, (
        "a4_convergence.yaml no longer describes the same training run as "
        "a1_first_experiment.yaml, so A1's bdh_cq/propagate seeds are not its "
        "plain arm; either restore the field or add `plain` back to A4's grid"
    )


def test_a4_and_a1_are_matched_on_content_not_on_hash():
    """The hashes differ, and the config comment must not claim otherwise."""
    assert config_hash(_resolved(A1, "plain")) != config_hash(_resolved(A4, "plain"))
    difference = set(hashable_yaml(_resolved(A1, "plain")).splitlines()) ^ set(
        hashable_yaml(_resolved(A4, "plain")).splitlines()
    )
    assert all(
        any(token in line for token in ("name:", "tags:", "- a1", "- a4", "[]"))
        for line in difference
    ), f"a1/a4 differ outside sweep identity: {sorted(difference)}"


def test_a4_measures_the_onset_at_the_depth_it_reports():
    """The primary metric is read at R=32; the onset table needs R=32 mid-training."""
    a4 = _resolved(A4, "attn_residual")
    assert a4.evaluation.intermediate_reasoning_steps is not None
    assert 32 in a4.evaluation.intermediate_reasoning_steps
    assert set(a4.evaluation.intermediate_reasoning_steps) <= set(a4.evaluation.reasoning_steps)
    # 16 keeps the curve comparable to A1's plain arm, which only has 1/4/16.
    assert 16 in a4.evaluation.intermediate_reasoning_steps


def test_a4_still_runs_the_full_step_budget():
    """The 8000-step pilot sat entirely before the onset window (12500-20000)."""
    assert _resolved(A4, "attn_residual").training.steps == 40000
