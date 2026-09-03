"""Expand a sweep YAML into resolved job configs + manifest.csv (FRAMEWORK_SPEC
sections 2, 8, 11; configs/README.md).

Usage:
    python tools/generate_sweep.py configs/stage_a/a1_first_experiment.yaml
    python tools/generate_sweep.py configs/stage_c/c1_recurrence_engineering.yaml \\
        --matched-controls
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from bdhx.config import PROJECT_ROOT, Config, apply_dotted_overrides, config_hash, load_raw
from bdhx.models.bdh import bdh_param_count
from bdhx.models.param_budget import solve_width
from bdhx.tasks.vocab import VOCAB_SIZE

# Recurrence r_max is the models' own default (bdh_cq.py / looped_transformer.py
# / unified_block.py all default r_max=8); the sweep generator has no resolved
# model instance to ask, so it uses that constant rather than deriving one per
# job from reasoning.train_steps.
DEFAULT_R_MAX = 8

# FRAMEWORK_SPEC section 3.1 "extra params" column, closed form in terms of
# r_max, width and adapter rank k. `combo` is the sum of the three additive
# variants it composes (step_gate + step_emb + adapter); `plain`/`residual`
# add nothing and are omitted.
PARAM_ADDING_KINDS = ("step_gate", "init_skip", "step_emb", "adapter", "attn_residual", "combo")


def _extra_params(kind: str, width: int, r_max: int, adapter_rank: int) -> int:
    if kind in ("step_gate", "init_skip"):
        return r_max
    if kind == "step_emb":
        return r_max * width
    if kind == "adapter":
        return r_max * 2 * width * max(adapter_rank, 0)
    if kind == "attn_residual":
        # recurrence.AttnResidual: learned query (width) + RMSNorm weight
        # (width) + a scalar distance bias.
        return 2 * width + 1
    if kind == "combo":
        return (
            _extra_params("step_gate", width, r_max, adapter_rank)
            + _extra_params("step_emb", width, r_max, adapter_rank)
            + _extra_params("adapter", width, r_max, adapter_rank)
        )
    return 0


def matched_control_params_target(
    base_params_target: int, kind: str, depth: int, adapter_rank: int, r_max: int = DEFAULT_R_MAX
) -> int | None:
    """`plain`'s params_target so it gains ~the params `kind` adds (section 3.1).

    Baseline width is solved analytically against `bdh_param_count` (fast,
    no nn.Module allocation) as a model-agnostic stand-in for whichever
    architecture the sweep is testing -- the exact same closed-form width
    solve `param_budget.solve_width` performs, just against a cheap proxy
    rather than constructing the real (possibly community) model.
    """
    if kind not in PARAM_ADDING_KINDS or base_params_target is None:
        return None
    width, realized = solve_width(
        lambda w: bdh_param_count(w, VOCAB_SIZE, depth=depth), base_params_target
    )
    extra = _extra_params(kind, width, r_max, adapter_rank)
    if extra <= 0:
        return None
    return realized + extra


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_sweep(path: str | Path) -> dict[str, Any]:
    with open(_resolve(path)) as f:
        data = yaml.safe_load(f) or {}
    if "base" not in data:
        raise ValueError(f"sweep file {path} is missing required key 'base'")
    return data


def _grid_combinations(grid: dict[str, list]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


class Job:
    """One resolved config: `overrides` documents how it was built."""

    def __init__(
        self,
        cfg: Config,
        overrides: dict[str, Any],
        seed: int,
        control_of: str | None = None,
        est_gpu_minutes: float | None = None,
    ):
        self.cfg = cfg
        self.overrides = overrides
        self.seed = seed
        self.control_of = control_of
        self.hash = config_hash(cfg)
        self.est_gpu_minutes = est_gpu_minutes

    @property
    def model(self) -> str:
        return self.cfg.model.name

    @property
    def task(self) -> str:
        return self.cfg.task.name


def expand_sweep(
    sweep_path: str | Path,
    *,
    dev: bool = False,
    matched_controls: bool = False,
    estimates: dict[str, float] | None = None,
) -> tuple[list[Job], dict[str, Any]]:
    """Expand `grid x seeds` (+ optional matched controls) into `Job`s."""
    sweep = load_sweep(sweep_path)
    meta = sweep.get("sweep", {})
    base_path = sweep["base"]
    grid = sweep.get("grid", {})
    seeds = sweep.get("seeds", [])
    overrides = sweep.get("overrides", {})
    sweep_dev = bool(sweep.get("dev", False)) or dev

    if len(seeds) < 3 and not sweep_dev:
        raise ValueError(
            f"sweep '{meta.get('name', sweep_path)}' has {len(seeds)} seed(s) (< 3); "
            "pass --dev (or set `dev: true` in the sweep file) to allow this"
        )

    base_raw = load_raw(base_path)
    exp_overrides = {
        "experiment.name": meta.get("name", "sweep"),
        "experiment.stage": meta.get("stage", "dev"),
        "experiment.tags": [t for t in (meta.get("name"), "dev" if sweep_dev else None) if t],
    }

    combos = _grid_combinations(grid)
    estimates = estimates or {}
    jobs: list[Job] = []
    for combo in combos:
        for seed in seeds:
            dotted = {**exp_overrides, **overrides, **combo, "training.seed": seed}
            raw = apply_dotted_overrides(base_raw, dotted)
            cfg = Config.model_validate(raw)
            job = Job(cfg, dotted, seed)
            job.est_gpu_minutes = estimates.get(job.hash)
            jobs.append(job)

    if matched_controls and "model.recurrence.kind" in grid:
        seen_kinds: set[str] = set()
        for job in list(jobs):
            kind = job.cfg.model.recurrence.kind
            if kind in seen_kinds or kind not in PARAM_ADDING_KINDS:
                continue
            base_target = job.cfg.model.params_target
            new_target = matched_control_params_target(
                base_target,
                kind,
                depth=job.cfg.model.depth,
                adapter_rank=job.cfg.model.recurrence.adapter_rank,
            )
            seen_kinds.add(kind)
            if new_target is None:
                continue
            control_dotted = {
                **job.overrides,
                "model.recurrence.kind": "plain",
                "model.params_target": new_target,
                "experiment.name": f"{job.cfg.experiment.name}_control_{kind}",
                "experiment.tags": [*job.cfg.experiment.tags, "matched_control"],
            }
            raw = apply_dotted_overrides(base_raw, control_dotted)
            control_cfg = Config.model_validate(raw)
            control_job = Job(control_cfg, control_dotted, job.seed, control_of=kind)
            control_job.est_gpu_minutes = estimates.get(control_job.hash)
            jobs.append(control_job)

    return jobs, meta


def write_jobs(jobs: list[Job], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(jobs):
        path = out_dir / f"exp_{i:03d}.yaml"
        path.write_text(yaml.safe_dump(job.cfg.model_dump(mode="json"), sort_keys=True))
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["exp", "config_hash", "seed", "model", "task", "control_of", "est_gpu_minutes"]
        )
        for i, job in enumerate(jobs):
            writer.writerow(
                [
                    f"exp_{i:03d}",
                    job.hash,
                    job.seed,
                    job.model,
                    job.task,
                    job.control_of or "",
                    "" if job.est_gpu_minutes is None else job.est_gpu_minutes,
                ]
            )
    return manifest_path


def total_estimated_gpu_hours(jobs: list[Job]) -> float:
    return sum(j.est_gpu_minutes for j in jobs if j.est_gpu_minutes) / 60.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", help="path to a sweep YAML under configs/")
    parser.add_argument("--dev", action="store_true", help="exempt from the 3-seed minimum")
    parser.add_argument("--matched-controls", action="store_true")
    parser.add_argument("--estimates", default=None, help="json {config_hash: minutes}")
    parser.add_argument("--max-gpu-hours", type=float, default=20.0)
    parser.add_argument("--allow-large-sweep", action="store_true")
    parser.add_argument("--out", default=None, help="override generated/<sweep>/ output dir")
    args = parser.parse_args(argv)

    estimates = None
    if args.estimates:
        estimates = json.loads(Path(args.estimates).read_text())

    jobs, meta = expand_sweep(
        args.sweep, dev=args.dev, matched_controls=args.matched_controls, estimates=estimates
    )

    total_hours = total_estimated_gpu_hours(jobs)
    if total_hours > args.max_gpu_hours and not args.allow_large_sweep:
        parser.error(
            f"sweep estimated at {total_hours:.1f} GPU-hours exceeds "
            f"--max-gpu-hours={args.max_gpu_hours}; pass --allow-large-sweep to proceed"
        )

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "generated" / meta.get("name", "sweep")
    manifest_path = write_jobs(jobs, out_dir)
    print(f"wrote {len(jobs)} configs to {out_dir} ({manifest_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
