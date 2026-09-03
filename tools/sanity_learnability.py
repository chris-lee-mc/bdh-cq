"""Pre-sweep learnability gate: does the pipeline learn anything at all?

Usage:
    python tools/sanity_learnability.py [--quick] [--out results/sanity] [--json PATH]

Runs the two checks of the Gate A diagnosis (EXPERIMENT_PLAN section 10) as
short CPU jobs on the `binding` task, where the answer is a single token that
is present verbatim in the context:

1. `transformer`, depth 2 distinct layers, R fixed at 1.
2. `looped_transformer`, depth 2 layers inside the shared block, R_train
   {1, 2}, evaluated at R_test 1 and 2.

Each check passes when exact match on the `interp` n_bindings=1 cell reaches
`--threshold` (default 0.9) at every evaluated R, and when the final training
loss has left the ln(vocab_size) plateau that `results.aggregate.at_chance`
flags. Both used to fail at exactly chance: the tied unembedding was
initialized at N(0, 1) so the init loss was tens of nats, and `model.depth`
was ignored by the looped model, leaving one layer per reasoning step.

n_bindings >= 2 is deliberately *not* a gate. Predicting the answer from the
[ANSWER] marker makes binding a three-hop induction problem (copy the key
forward, copy the query forward, then match), and no model in this framework
solves it at these budgets; see `docs/RESULTS.md` section A0. The gate is
"the pipeline learns", not "the models are good".

Exit code 0 when every check passes, 1 otherwise. Run it before every sweep
(`docs/HANDOFF_TASKS.md` task 23b).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bdhx.models
import bdhx.tasks  # noqa: F401  (registers the tasks)
from bdhx.config import config_hash, load_config
from bdhx.results.aggregate import at_chance, chance_loss
from bdhx.seeding import seed_everything
from bdhx.training.evaluate import run_evaluation
from bdhx.training.trainer import (
    Trainer,
    apply_compute_settings,
    build_model,
    build_task,
    max_target_length,
)

BASE_CONFIG = "configs/base/tiny_smoke.yaml"
THRESHOLD = 0.9
# --quick shrinks params and steps by about 4x, so it cannot reach the full
# threshold; 0.5 is still three orders of magnitude above the 1/vocab floor.
QUICK_THRESHOLD = 0.5
GATE_DIFFICULTY = {"n_bindings": 1}

COMMON: dict[str, Any] = {
    "task.name": "binding",
    "task.train_difficulties": [{"n_bindings": 1}, {"n_bindings": 2}],
    "task.eval_difficulties": {
        "interp": [{"n_bindings": 1}, {"n_bindings": 2}],
        "mild": [{"n_bindings": 4}],
        "strong": [{"n_bindings": 8}],
    },
    "model.params_target": 1_500_000,
    "model.depth": 2,
    "training.batch_size": 32,
    "training.lr": 1.0e-3,
    "training.warmup_steps": 100,
    "training.steps": 3000,
    "training.eval_every_steps": 0,
    "training.checkpoint_every_steps": 0,
    "task.n_eval_episodes": 100,
    "evaluation.diagnostics": False,
    "compute.device": "cpu",
    "compute.deterministic": False,
    "compute.max_wall_clock_minutes": 20,
}

CHECKS: list[dict[str, Any]] = [
    {
        "name": "transformer_depth2",
        "overrides": {
            "model.name": "transformer",
            "reasoning.train_steps": [1],
            "evaluation.reasoning_steps": [1],
        },
    },
    {
        "name": "looped_transformer_depth2",
        "overrides": {
            "model.name": "looped_transformer",
            "reasoning.train_steps": [1, 2],
            "evaluation.reasoning_steps": [1, 2],
        },
    },
]

QUICK: dict[str, Any] = {
    "model.params_target": 400_000,
    "training.steps": 400,
    "task.n_eval_episodes": 32,
}


def run_check(check: dict[str, Any], out_root: Path, quick: bool, threshold: float) -> dict:
    overrides = dict(COMMON)
    overrides.update(check["overrides"])
    if quick:
        overrides.update(QUICK)
    cfg = load_config(BASE_CONFIG, overrides)
    run_dir = out_root / check["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    device = apply_compute_settings(cfg)
    seed_everything(cfg.training.seed)
    task = build_task(cfg)
    model = build_model(cfg, task, max_target_length(cfg, task))
    trainer = Trainer(cfg, model, task, run_dir, device=device, config_hash=config_hash(cfg))
    started = time.perf_counter()
    state = trainer.train()
    rows = run_evaluation(
        model, task, cfg, state.step, splits=("interp",), device=device, collect_diagnostics=False
    )

    gate_rows = [
        r for r in rows if r.split == "interp" and dict(r.difficulty) == dict(GATE_DIFFICULTY)
    ]
    accuracy = {int(r.reasoning_steps): float(r.exact_match) for r in gate_rows}
    other = {
        f"n_bindings={r.difficulty.get('n_bindings')} R={r.reasoning_steps}": round(
            float(r.exact_match), 3
        )
        for r in rows
        if dict(r.difficulty) != dict(GATE_DIFFICULTY)
    }
    final_loss = _final_loss(run_dir)
    flat = at_chance(final_loss, cfg.task.name)
    passed = bool(accuracy) and not flat and all(v >= threshold for v in accuracy.values())
    return {
        "name": check["name"],
        "model": cfg.model.name,
        "depth": cfg.model.depth,
        "params": model.param_report().trainable,
        "steps": state.step,
        "status": state.status,
        "final_train_loss": final_loss,
        "chance_loss": chance_loss(cfg.task.name),
        "at_chance": flat,
        "exact_match_n_bindings_1": accuracy,
        "other_cells": other,
        "wall_clock_s": round(time.perf_counter() - started, 1),
        "passed": passed,
        "run_dir": str(run_dir),
    }


def _final_loss(run_dir: Path) -> float | None:
    import csv
    import math

    path = run_dir / "train_log.csv"
    if not path.exists():
        return None
    last = None
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                value = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                last = value
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/sanity_learnability")
    parser.add_argument("--quick", action="store_true", help="shrink params and steps for CI")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"exact-match gate (default {THRESHOLD}, {QUICK_THRESHOLD} with --quick)",
    )
    parser.add_argument("--only", default=None, help="run one check by name")
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    threshold = args.threshold
    if threshold is None:
        threshold = QUICK_THRESHOLD if args.quick else THRESHOLD
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    checks = [c for c in CHECKS if args.only in (None, c["name"])]
    if not checks:
        raise SystemExit(f"--only {args.only!r} matches no check")

    reports = []
    for check in checks:
        print(f"[sanity] running {check['name']} ...", flush=True)
        report = run_check(check, out_root, args.quick, threshold)
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)

    print(f"\n[sanity] summary (exact match on binding interp n_bindings=1, gate {threshold})")
    print(f"{'check':30s} {'params':>9s} {'steps':>6s} {'loss':>7s}  accuracy  result")
    for r in reports:
        acc = " ".join(f"R{k}={v:.2f}" for k, v in sorted(r["exact_match_n_bindings_1"].items()))
        loss = "n/a" if r["final_train_loss"] is None else f"{r['final_train_loss']:.3f}"
        verdict = "PASS" if r["passed"] else ("AT_CHANCE" if r["at_chance"] else "FAIL")
        print(f"{r['name']:30s} {r['params']:>9,} {r['steps']:>6d} {loss:>7s}  {acc}  {verdict}")

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
    failed = [r["name"] for r in reports if not r["passed"]]
    if failed:
        print(f"\n[sanity] FAILED: {', '.join(failed)} -- do not launch a sweep", flush=True)
        return 1
    print("\n[sanity] all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
