"""Re-evaluate a finished run's checkpoint at reasoning-step counts it never reported.

A run only stores the R values its config listed in `evaluation.reasoning_steps`,
so a question like "does R=6 work?" cannot be answered from `results/` even
though the trained weights are sitting right there. This re-runs the same
evaluation the trainer runs, from `metadata.json`'s resolved config, at whatever
R values are asked for -- no GPU time and no retraining.

It exists because A5 trained on {1,2,4,8} and tested only powers of two, which
cannot distinguish "learned the trained SET" (R=6 fails, sitting between two
trained values) from "learned the trained INTERVAL" (R=6 works). See
RESULTS.md section A5.

Always re-evaluate at least one R the run already reported: the printed
`stored` column is the reproduction check, and a mismatch there means the
numbers in the new column are not trustworthy either.

Usage:
    python tools/reeval_checkpoint.py results/75f014a16a02_s1 \\
        --reasoning-steps 3 4 5 6 7 8 --split interp --out reports/a5b.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdhx.config import Config
from bdhx.training.evaluate import run_evaluation
from bdhx.training.trainer import build_model, build_task


def load_run(run_dir: Path):
    """Returns (cfg, task, model, step) with the final checkpoint's weights loaded."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    cfg = Config.model_validate(meta["config"])
    task = build_task(cfg)
    model = build_model(cfg, task)

    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    path = ckpts[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)

    # Same guard the trainer applies on --resume: a checkpoint from a different
    # experiment loaded into this config would silently produce numbers that
    # look real. metadata.json and the checkpoint must agree.
    stored = payload.get("config_hash")
    if stored != meta["config_hash"]:
        raise ValueError(
            f"{path}: checkpoint config hash {stored} != metadata {meta['config_hash']}"
        )
    model.load_state_dict(payload["model"])
    model.eval()
    return cfg, task, model, int(payload["step"])


def stored_rows(run_dir: Path, step: int, split: str) -> dict[tuple[str, int], float]:
    """The exact_match this run already reported, keyed by (difficulty, R)."""
    d = json.loads((run_dir / "results.json").read_text())
    out: dict[tuple[str, int], float] = {}
    for r in d["evaluations"]:
        if r["step"] != step or r["split"] != split:
            continue
        # Last row wins: a resumed run appends duplicates rather than replacing.
        out[(json.dumps(r["difficulty"], sort_keys=True), r["reasoning_steps"])] = r["exact_match"]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--reasoning-steps", type=int, nargs="+", required=True)
    p.add_argument("--split", default="interp")
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--out", default=None, help="write a CSV here")
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="collect the per-iteration diagnostics too; off by default because "
        "they dominate CPU runtime and answer a different question",
    )
    args = p.parse_args(argv)

    rows_out = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        cfg, task, model, step = load_run(run_dir)
        known = stored_rows(run_dir, step, args.split)
        rows = run_evaluation(
            model,
            task,
            cfg,
            step,
            reasoning_steps=args.reasoning_steps,
            splits=(args.split,),
            n_episodes=args.n_episodes,
            collect_diagnostics=args.diagnostics,
            device=torch.device("cpu"),
        )
        print(f"=== {run_dir.name}  step={step}  split={args.split}")
        print(f"{'difficulty':<46} {'R':>3} {'exact_match':>12} {'stored':>9} {'delta':>8}")
        for r in rows:
            key = (json.dumps(r.difficulty, sort_keys=True), r.reasoning_steps)
            prev = known.get(key)
            delta = "" if prev is None else f"{r.exact_match - prev:+.4f}"
            print(
                f"{json.dumps(r.difficulty, sort_keys=True):<46} {r.reasoning_steps:>3} "
                f"{r.exact_match:>12.4f} {'' if prev is None else f'{prev:.4f}':>9} {delta:>8}"
            )
            rows_out.append(
                {
                    "run_id": run_dir.name,
                    "step": step,
                    "split": args.split,
                    "difficulty": json.dumps(r.difficulty, sort_keys=True),
                    "reasoning_steps": r.reasoning_steps,
                    "exact_match": r.exact_match,
                    "token_acc": r.token_acc,
                    "n_episodes": r.n_episodes,
                    "stored_exact_match": "" if prev is None else prev,
                }
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
