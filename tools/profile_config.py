"""Cost guard: profile a config before launching it (FRAMEWORK_SPEC section 11).

Usage:
    python tools/profile_config.py --config configs/base/default.yaml \
        [--steps 50] [--eval-episodes 32] [--overrides model.name=transformer] [--json]

Runs `--steps` training steps plus one evaluation pass at the largest
`evaluation.reasoning_steps` and prints seconds/step, evaluation seconds, peak
memory and the projected minutes for the full config. The projection scales the
measured evaluation pass by the configured episode count, splits and depth list,
using the most expensive depth for every depth, so it is an upper bound.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bdhx.models
import bdhx.tasks  # noqa: F401  (registers the tasks)
from bdhx.config import load_config
from bdhx.seeding import seed_everything
from bdhx.training.evaluate import (
    EVAL_SPLITS,
    reduced_reasoning_steps,
    run_evaluation,
)
from bdhx.training.trainer import (
    Trainer,
    apply_compute_settings,
    build_model,
    build_task,
)
from tools.run_experiment import parse_overrides

WARMUP_STEPS = 2


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def profile(cfg, steps: int = 50, eval_episodes: int = 32) -> dict:
    device = apply_compute_settings(cfg)
    seed_everything(cfg.training.seed)
    task = build_task(cfg)
    model = build_model(cfg, task)
    report = model.param_report()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(cfg, model, task, Path(tmp), device=device)
        timed = 0.0
        for step in range(steps + WARMUP_STEPS):
            t0 = time.perf_counter()
            batch = trainer.sample_batch(step)
            trainer.train_step(batch, trainer.sampler.sample(step))
            if step >= WARMUP_STEPS:
                timed += time.perf_counter() - t0
    seconds_per_step = timed / max(steps, 1)

    r_max = max(cfg.evaluation.reasoning_steps)
    t0 = time.perf_counter()
    rows = run_evaluation(
        model,
        task,
        cfg,
        0,
        reasoning_steps=[r_max],
        splits=("interp",),
        n_episodes=eval_episodes,
        device=device,
    )
    eval_seconds = time.perf_counter() - t0

    n_eval = int(cfg.task.n_eval_episodes)
    scale = (n_eval / max(eval_episodes, 1)) * len(EVAL_SPLITS)
    full_eval_s = eval_seconds * scale * len(cfg.evaluation.reasoning_steps)
    reduced_eval_s = eval_seconds * scale * len(reduced_reasoning_steps(cfg))
    every = int(cfg.training.eval_every_steps or 0)
    n_intermediate = (int(cfg.training.steps) // every) if every else 0
    total_s = (
        seconds_per_step * int(cfg.training.steps) + full_eval_s + n_intermediate * reduced_eval_s
    )
    return {
        "config_model": cfg.model.name,
        "config_task": cfg.task.name,
        "device": str(device),
        "params_trainable": report.trainable,
        "solved_width": getattr(model, "solved_width", None),
        "profiled_steps": steps,
        "seconds_per_step": seconds_per_step,
        "eval_seconds": eval_seconds,
        "eval_episodes_profiled": eval_episodes,
        "eval_reasoning_steps": r_max,
        "eval_rows": len(rows),
        "peak_memory_bytes": peak_memory_bytes(device),
        "projected_train_minutes": seconds_per_step * int(cfg.training.steps) / 60.0,
        "projected_eval_minutes": (full_eval_s + n_intermediate * reduced_eval_s) / 60.0,
        "projected_minutes": total_s / 60.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--overrides", nargs="*", default=None, metavar="KEY=VALUE")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.overrides))
    out = profile(cfg, steps=args.steps, eval_episodes=args.eval_episodes)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model            {out['config_model']} ({out['params_trainable']} params)")
        print(f"task             {out['config_task']} on {out['device']}")
        print(f"seconds/step     {out['seconds_per_step']:.4f}")
        print(
            f"eval seconds     {out['eval_seconds']:.3f} "
            f"({out['eval_episodes_profiled']} episodes at R={out['eval_reasoning_steps']})"
        )
        print(f"peak memory      {out['peak_memory_bytes'] / 1e6:.1f} MB")
        print(f"projected train  {out['projected_train_minutes']:.1f} min")
        print(f"projected eval   {out['projected_eval_minutes']:.1f} min")
        print(f"projected total  {out['projected_minutes']:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
