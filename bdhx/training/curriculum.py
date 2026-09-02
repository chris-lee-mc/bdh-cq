"""R_train scheduling: sampling, curriculum, delayed start (FRAMEWORK_SPEC section 7)."""

from __future__ import annotations

from bdhx.seeding import task_rng

SAMPLING_KINDS = ("uniform", "curriculum", "fixed")


def progress(step: int, total_steps: int) -> float:
    """Fraction of training done at the start of `step` (0-based), clamped to [0, 1]."""
    if total_steps <= 0:
        return 0.0
    return min(max(step / total_steps, 0.0), 1.0)


def curriculum_value(curriculum, step: int, total_steps: int) -> int:
    """`schedule[i]` for the last `switch_at[i]` that has been reached."""
    if curriculum is None:
        raise ValueError("train_step_sampling 'curriculum' needs reasoning.curriculum")
    schedule, switch_at = list(curriculum.schedule), list(curriculum.switch_at)
    if len(schedule) != len(switch_at):
        raise ValueError("curriculum.schedule and curriculum.switch_at must have equal length")
    if not schedule:
        raise ValueError("curriculum.schedule is empty")
    p = progress(step, total_steps)
    value = schedule[0]
    for r, s in zip(schedule, switch_at):
        if p + 1e-12 >= s:
            value = r
    return int(value)


def delayed_start_active(reasoning_cfg, step: int, total_steps: int) -> bool:
    """H5: recurrence is forced to a single step before `delayed_start_fraction`."""
    return progress(step, total_steps) < float(reasoning_cfg.delayed_start_fraction)


class RTrainSampler:
    """Per-step R_train: a pure function of (seed, step) so resume is bit-exact."""

    def __init__(self, reasoning_cfg, total_steps: int, seed: int = 0):
        kind = reasoning_cfg.train_step_sampling
        if kind not in SAMPLING_KINDS:
            raise ValueError(f"unknown train_step_sampling '{kind}'; expected {SAMPLING_KINDS}")
        if not reasoning_cfg.train_steps:
            raise ValueError("reasoning.train_steps is empty")
        if kind == "curriculum" and reasoning_cfg.curriculum is None:
            raise ValueError("train_step_sampling 'curriculum' needs reasoning.curriculum")
        self.cfg = reasoning_cfg
        self.kind = kind
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.train_steps = [int(r) for r in reasoning_cfg.train_steps]

    @property
    def r_max(self) -> int:
        values = list(self.train_steps)
        if self.cfg.curriculum is not None:
            values += [int(r) for r in self.cfg.curriculum.schedule]
        return max(values)

    def sample(self, step: int) -> int:
        if delayed_start_active(self.cfg, step, self.total_steps):
            return 1
        if self.kind == "fixed":
            return self.train_steps[0]
        if self.kind == "curriculum":
            return curriculum_value(self.cfg.curriculum, step, self.total_steps)
        rng = task_rng(self.seed, "r_train", int(step))
        return int(self.train_steps[int(rng.integers(len(self.train_steps)))])
