"""Recurrent-stability instrumentation (FRAMEWORK_SPEC sections 5 and 6).

Per reasoning step the evaluator records `state_norm`, `update_norm`,
`cos_consecutive` (all produced by the model itself through
`solve(collect_diagnostics=True)`), the fraction of active neurons and a
percentile summary of the per-neuron activation magnitude (recorded here from
the activation modules of the block that the step applies), NaN counts, and at
`reasoning_steps` in {8, 32} a power-iteration estimate of the dominant
eigenvalue magnitude of the step Jacobian.

Deviations, documented:

- The eigenvalue estimate uses finite differences of the step map around the
  state the run actually reached (`(F(h + eps v) - F(h)) / eps`), five
  iterations, on a random probe direction. It needs no double backward and
  therefore works for every block, including the community BDH block.
- `active_neuron_frac` / `activation_percentiles` are read from the ReLU
  neurons of the BDH family, and from the block output for models whose blocks
  have no separate activation module (Transformer, gated DeltaNet).
- Models with no recurrence runner (fixed-depth baselines) have no per-step
  series of their own beyond the last block, so their series length is the
  number of block applications (`model.depth`), not `reasoning_steps`.
- A recurrent model whose step applies `model.depth` layers fires the
  activation tap `depth` times per step; the last layer of each step is kept so
  that every series has one entry per reasoning step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from bdhx.models.recurrence import RecurrenceRunner

PERCENTILES = (5.0, 25.0, 50.0, 75.0, 95.0)
JACOBIAN_DEPTHS = (8, 32)
JACOBIAN_EPISODES = 32
POWER_ITERATIONS = 5
POWER_EPS = 1e-3
ACTIVE_EPS = 1e-6
SERIES_KEYS = (
    "state_norm",
    "update_norm",
    "cos_consecutive",
    "active_neuron_frac",
    "activation_percentiles",
    "jacobian_eigenvalue_estimate",
)


# -- primitive stats -------------------------------------------------------
def activation_stats(x: Tensor) -> tuple[float, list[float], int]:
    """(active fraction, percentiles of the per-neuron magnitude, NaN/Inf count)."""
    with torch.no_grad():
        f = x.detach().float()
        bad = int((~torch.isfinite(f)).sum())
        f = torch.nan_to_num(f)
        active = float((f.abs() > ACTIVE_EPS).float().mean())
        per_neuron = f.abs().reshape(-1, f.shape[-1]).mean(dim=0)
        qs = torch.tensor([p / 100.0 for p in PERCENTILES], dtype=per_neuron.dtype)
        pcts = [float(v) for v in torch.quantile(per_neuron, qs)]
    return active, pcts, bad


def state_pair_stats(prev: Tensor, cur: Tensor) -> tuple[float, float, float]:
    """(state_norm, update_norm, cos_consecutive) for one step, as `RecurrenceRunner`."""
    with torch.no_grad():
        a, b = prev.detach().float().flatten(1), cur.detach().float().flatten(1)
        norm = float(b.norm(dim=-1).mean())
        if a.shape != b.shape:
            return norm, 0.0, 1.0
        update = float((b - a).norm(dim=-1).mean())
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8).mean())
    return norm, update, cos


# -- taps ------------------------------------------------------------------
def activation_modules(model: nn.Module) -> list[nn.Module]:
    """Modules whose output measures neuron activation, one call per block application."""
    blocks: list[nn.Module] = []
    if hasattr(model, "block_modules"):
        blocks = list(model.block_modules)
    elif isinstance(getattr(model, "blocks", None), nn.ModuleList):
        blocks = list(model.blocks)
    elif isinstance(getattr(model, "block", None), nn.Module):
        blocks = [model.block]
    out: list[nn.Module] = []
    for blk in blocks:
        act = getattr(blk, "qk_activation", None)  # community BDH sparse neurons
        out.append(act if isinstance(act, nn.Module) else blk)
    return out


class ActivationTap:
    """Records activation stats for each call of `modules` while enabled."""

    def __init__(self, modules: Iterable[nn.Module], keep_states: bool = False):
        self.modules = list(modules)
        self.enabled = True
        self.keep_states = keep_states
        self.records: list[tuple[float, list[float], int]] = []
        self.states: list[Tensor] = []
        self._handles: list[Any] = []

    def _hook(self, _mod, _inp, out):
        if not self.enabled:
            return
        x = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(x):
            return
        self.records.append(activation_stats(x))
        if self.keep_states:
            self.states.append(x.detach())

    def __enter__(self):
        self._handles = [m.register_forward_hook(self._hook) for m in self.modules]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# -- jacobian --------------------------------------------------------------
def power_iteration_eigenvalue(
    step_fn: Callable[[Tensor], Tensor],
    h: Tensor,
    n_iter: int = POWER_ITERATIONS,
    eps: float = POWER_EPS,
) -> float:
    """Dominant eigenvalue magnitude of the step Jacobian at `h` (finite differences)."""
    with torch.no_grad():
        base = step_fn(h).detach().float()
        v = torch.randn_like(h.float())
        v = v / (v.norm() + 1e-12)
        sigma = 0.0
        for _ in range(max(int(n_iter), 1)):
            probe = (h.float() + eps * v).to(h.dtype)
            jv = (step_fn(probe).detach().float() - base) / eps
            norm = float(jv.norm())
            if not math.isfinite(norm) or norm <= 0.0:
                return norm if math.isfinite(norm) else float("nan")
            sigma = norm
            v = jv / norm
    return sigma


def _snapshot(model: nn.Module) -> dict[str, Any]:
    """Latent-loop scratch that a probe call would clobber (bdh_cq)."""
    snap: dict[str, Any] = {}
    if hasattr(model, "_mem"):
        snap["_mem"] = model._mem
    outs = getattr(model, "_all_block_outputs", None)
    if outs is not None:
        snap["_all_block_outputs"] = list(outs)
    return snap


def _restore(model: nn.Module, snap: dict[str, Any]) -> None:
    for key, value in snap.items():
        setattr(model, key, value)


# -- collection ------------------------------------------------------------
def collect(
    model: nn.Module,
    solve_fn: Callable[[], Any],
    reasoning_steps: int,
    jacobian: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Runs `solve_fn()` (which must pass `collect_diagnostics=True`) with taps on.

    Returns `(solve_output, series)` where `series` has the keys of
    `results.schema.Diagnostics`.
    """
    runner = getattr(model, "recurrence", None)
    runner = runner if isinstance(runner, RecurrenceRunner) else None
    tap = ActivationTap(activation_modules(model), keep_states=runner is None)
    jac: list[float] = []
    original = runner.block if runner is not None else None

    if runner is not None:
        tap.enabled = False
        steps = max(int(reasoning_steps), 0)
        seen = [0]

        def wrapped(h, s=None, _orig=original):
            seen[0] += 1
            tap.enabled = True
            try:
                out = _orig(h, s)
            finally:
                tap.enabled = False
            if jacobian and not jac and seen[0] >= steps:
                snap = _snapshot(model)
                try:
                    jac.append(power_iteration_eigenvalue(lambda x: _orig(x, s), h))
                finally:
                    _restore(model, snap)
            return out

        runner._block = (wrapped,)

    try:
        with torch.no_grad(), tap:
            out = solve_fn()
    finally:
        if runner is not None:
            runner._block = (original,)

    diag = dict(getattr(out, "diagnostics", None) or {})
    records = tap.records
    if runner is None:
        records = records[: max(int(getattr(out, "block_applications", 0)), 1)]
    length = len(diag.get("state_norm", []) or [])
    if runner is not None and length and len(records) > length and not len(records) % length:
        # `model.depth` layers per reasoning step fire the activation tap once
        # per layer; keep the last layer of each step so the series stay aligned
        # with the one state the runner records per step.
        per_step = len(records) // length
        records = records[per_step - 1 :: per_step]
    if length != len(records):
        # fixed-depth baselines only record the last block; rebuild from the tap.
        length = len(records)
        norms, updates, coss = [], [], []
        for i, state in enumerate(tap.states[:length]):
            prev = tap.states[i - 1] if i else state
            n, u, c = state_pair_stats(prev, state)
            norms.append(n)
            updates.append(u)
            coss.append(c)
        diag = {"state_norm": norms, "update_norm": updates, "cos_consecutive": coss}
    nan_count = int(sum(int(n) for n in diag.get("nan_count", []) or []))
    nan_count += sum(r[2] for r in records)
    series = {
        "state_norm": [float(v) for v in diag.get("state_norm", [])][:length],
        "update_norm": [float(v) for v in diag.get("update_norm", [])][:length],
        "cos_consecutive": [float(v) for v in diag.get("cos_consecutive", [])][:length],
        "active_neuron_frac": [r[0] for r in records[:length]],
        "activation_percentiles": [r[1] for r in records[:length]],
        "jacobian_eigenvalue_estimate": jac,
        "nan_count": nan_count,
    }
    tap.states.clear()
    return out, series


def merge(series_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Elementwise mean over episodes/batches, truncated to the shortest series."""
    series_list = [s for s in series_list if s]
    if not series_list:
        return {k: [] for k in SERIES_KEYS} | {"nan_count": 0}
    out: dict[str, Any] = {"nan_count": int(sum(s.get("nan_count", 0) for s in series_list))}
    for key in SERIES_KEYS:
        cols = [s.get(key) or [] for s in series_list]
        cols = [c for c in cols if c]
        if not cols:
            out[key] = []
            continue
        n = min(len(c) for c in cols)
        if key == "activation_percentiles":
            out[key] = [
                [sum(c[i][j] for c in cols) / len(cols) for j in range(len(PERCENTILES))]
                for i in range(n)
            ]
        else:
            out[key] = [sum(float(c[i]) for c in cols) / len(cols) for i in range(n)]
    return out


class GradNormTracker:
    """Running summary of the per-step gradient norms logged by `trainer.py`."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.max = 0.0
        self.last = 0.0
        self.nonfinite = 0

    def update(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            self.nonfinite += 1
            return value
        self.count += 1
        self.total += value
        self.max = max(self.max, value)
        self.last = value
        return value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "grad_norm_mean": self.mean,
            "grad_norm_max": self.max,
            "grad_norm_last": self.last,
            "grad_norm_nonfinite": float(self.nonfinite),
        }
