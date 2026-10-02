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
A5 = "configs/stage_a/a5_r_train_extension.yaml"
A2 = "configs/stage_a/a2_curriculum.yaml"

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


def test_a5_matches_a1_except_for_the_one_manipulation():
    """A5 reuses A1's arm too, so only `reasoning.train_steps` may differ.

    A5's whole claim is that extending the trained R set is the *only* change
    between it and A1. Any second difference would make a moved accuracy peak
    unattributable, which is the one thing the sweep exists to measure.
    """
    a1 = _resolved(A1, "plain")
    a5 = _resolved(A5, "plain")

    d1 = a1.model_dump(mode="json")
    d5 = a5.model_dump(mode="json")
    for field in IDENTITY_FIELDS:
        d1["experiment"].pop(field, None)
        d5["experiment"].pop(field, None)
    d1["evaluation"].pop("intermediate_reasoning_steps", None)
    d5["evaluation"].pop("intermediate_reasoning_steps", None)

    assert d1["reasoning"]["train_steps"] == [1, 2, 4]
    assert d5["reasoning"]["train_steps"] == [1, 2, 4, 8]
    d1.pop("reasoning")
    d5.pop("reasoning")

    assert d1 == d5, (
        "a5_r_train_extension.yaml differs from a1_first_experiment.yaml in "
        "something other than reasoning.train_steps, so a moved peak could not "
        "be attributed to the trained R set"
    )


def test_a5_actually_extends_r_train_max_which_is_what_a2_does_not():
    """The point of A5 over A2: A2's two arms both cap R_train_max at 4.

    A2's curriculum schedule is [1,2,4] over train_steps [1,2,4], so its
    `r_max` is 4 in both arms and it cannot move a peak past a value it never
    trains. RESULTS.md described A2 as the probe of whether the readout is the
    binding constraint; this test is what stops that description coming back.
    """
    from bdhx.training.curriculum import RTrainSampler

    a5 = _resolved(A5, "plain")
    assert RTrainSampler(a5.reasoning, total_steps=40000, seed=1).r_max == 8

    with open(A2) as fh:
        a2 = yaml.safe_load(fh)
    raw = apply_dotted_overrides(load_raw(a2["base"]), dict(a2["overrides"]))
    raw["experiment"] = {"name": a2["sweep"]["name"], "stage": a2["sweep"]["stage"]}
    for sampling in a2["grid"]["reasoning.train_step_sampling"]:
        cfg = Config.model_validate(
            apply_dotted_overrides(
                dict(raw), {"reasoning.train_step_sampling": sampling, "task.name": "propagate"}
            )
        )
        r_max = RTrainSampler(cfg.reasoning, total_steps=40000, seed=1).r_max
        assert r_max == 4, (
            f"a2_curriculum.yaml's '{sampling}' arm now trains up to R={r_max}; "
            "if A2 has been changed to extend R_train_max, it overlaps A5 and "
            "one of the two should go"
        )


def test_a5_can_observe_its_own_peak_before_the_final_checkpoint():
    """R=8 is the new R_train_max and the value the sweep is read at."""
    a5 = _resolved(A5, "plain")
    assert a5.evaluation.intermediate_reasoning_steps is not None
    assert 8 in a5.evaluation.intermediate_reasoning_steps
    assert set(a5.evaluation.intermediate_reasoning_steps) <= set(a5.evaluation.reasoning_steps)
    # The extrapolation range the result is read over has to be evaluated.
    assert {8, 16, 32} <= set(a5.evaluation.reasoning_steps)
