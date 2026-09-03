"""Run one experiment config end to end (FRAMEWORK_SPEC sections 4-7).

Usage:
    python tools/run_experiment.py --config configs/base/tiny_smoke.yaml \
        --out results/ [--run-id my_run] [--resume] \
        [--overrides model.name=transformer training.steps=10] [--sync-bucket s3://...]

Writes `<out>/<run_id>/` with metadata.json, results.json, train_log.csv,
log.txt and checkpoints/. `run_id` defaults to `<config_hash>_s<seed>`.
Cached evaluation shards are read from `$BDHX_DATA_ROOT` (default `data/`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bdhx.models
import bdhx.tasks  # noqa: F401  (registers the tasks)
from bdhx.config import config_hash, load_config
from bdhx.metadata import collect_metadata, update_metadata
from bdhx.models.param_budget import HARD_TOL
from bdhx.results.schema import ResultsWriter
from bdhx.seeding import seed_everything
from bdhx.training.evaluate import reduced_reasoning_steps, run_evaluation
from bdhx.training.trainer import (
    Trainer,
    apply_compute_settings,
    build_model,
    build_task,
    max_target_length,
)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not key=value")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        out[key.strip()] = value
    return out


class Logger:
    """Tees to `<run_dir>/log.txt` and stdout."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "log.txt"
        self._fh = self.path.open("a")

    def __call__(self, message: str) -> None:
        line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def check_param_budget(cfg, report, log) -> None:
    target = cfg.model.params_target
    if not target:
        return
    off = abs(report.trainable - target) / target
    log(f"params trainable={report.trainable} target={target} off_by={off:.3%}")
    if off > HARD_TOL:
        raise SystemExit(
            f"realized parameters {report.trainable} are {off:.2%} from params_target "
            f"{target} (hard tolerance {HARD_TOL:.0%})"
        )


def build_checkpoint_sync(bucket: str | None, run_id: str, config_hash: str, log):
    """A `CheckpointSync` when a bucket is configured, else None.

    S3 is optional throughout (RUNPOD.md sections 3 and 4): with no bucket the
    run writes only to local disk and is collected over scp by
    `tools/runpod_launch.py collect`.
    """
    from bdhx.s3sync import CheckpointSync, S3Settings, make_client

    settings = S3Settings.from_env(bucket)
    if settings is None:
        return None
    log(f"checkpoint sync enabled: s3://{settings.bucket}/runs/{run_id}")
    return CheckpointSync(make_client(settings), settings.bucket, run_id, config_hash, log=log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="results")
    parser.add_argument("--overrides", nargs="*", default=None, metavar="KEY=VALUE")
    parser.add_argument(
        "--sync-bucket",
        default=None,
        help="S3-compatible bucket for checkpoint/result sync; defaults to $S3_BUCKET",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.overrides))
    chash = config_hash(cfg)
    run_id = args.run_id or f"{chash}_s{cfg.training.seed}"
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir)
    log(f"run_id={run_id} config={args.config} config_hash={chash}")

    device = apply_compute_settings(cfg)
    seed_everything(cfg.training.seed)
    task = build_task(cfg)
    target_length = max_target_length(cfg, task)
    model = build_model(cfg, task, target_length)
    report = model.param_report()
    check_param_budget(cfg, report, log)
    log(
        f"model={cfg.model.name} task={cfg.task.name} device={device} target_length={target_length}"
    )

    collect_metadata(cfg, run_dir)
    update_metadata(
        run_dir,
        param_total=report.total,
        param_trainable=report.trainable,
        param_breakdown=report.breakdown,
        serialized_bytes=report.serialized_bytes,
        solved_width=getattr(model, "solved_width", None),
        log_path=str(log.path),
    )

    gate = getattr(getattr(model, "recurrence", None), "gate_extrapolation", "hold_last")
    writer = ResultsWriter(
        run_dir,
        run_id=run_id,
        config_hash=chash,
        seed=cfg.training.seed,
        model=cfg.model.name,
        task=cfg.task.name,
        params=report.trainable,
        solved_width=getattr(model, "solved_width", None),
        gate_extrapolation=gate,
        status="running",
    )
    if args.resume and (run_dir / "results.json").exists():
        previous = ResultsWriter.load(run_dir)
        if previous.config_hash == chash:  # keep the pre-interruption learning curve
            writer.results.evaluations.extend(previous.evaluations)
            log(f"resume: kept {len(previous.evaluations)} evaluation rows")
    writer.flush()

    eval_seconds = 0.0

    def intermediate_eval(step: int) -> None:
        nonlocal eval_seconds
        t0 = time.perf_counter()
        log(f"intermediate eval at step {step}")
        run_evaluation(
            model,
            task,
            cfg,
            step,
            reasoning_steps=reduced_reasoning_steps(cfg),
            writer=writer,
            device=device,
        )
        eval_seconds += time.perf_counter() - t0

    sync = build_checkpoint_sync(args.sync_bucket, run_id, chash, log)
    trainer = Trainer(
        cfg,
        model,
        task,
        run_dir,
        device=device,
        eval_fn=intermediate_eval,
        log=log,
        resume=args.resume,
        config_hash=chash,
        on_checkpoint=(lambda path, step, _reason: sync.upload_checkpoint(path, step))
        if sync
        else None,
    )
    state = trainer.train()
    log(f"training finished status={state.status} steps={state.step}")

    t0 = time.perf_counter()
    run_evaluation(model, task, cfg, state.step, writer=writer, device=device)
    eval_seconds += time.perf_counter() - t0
    writer.results.status = state.status
    writer.flush()

    update_metadata(
        run_dir,
        end_time=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        wall_clock_train_s=state.wall_clock_train_s,
        wall_clock_eval_s=eval_seconds,
        steps_completed=state.step,
        examples_seen=state.examples_seen,
        tokens_seen=state.tokens_seen,
        train_flops_estimate=state.train_flops_estimate,
        peak_vram_bytes=state.peak_vram_bytes,
        nan_events=state.nan_events,
        checkpoint_path=state.checkpoint_path,
        resumed_from=state.resumed_from,
        preemptions=state.preemptions,
    )
    if sync:
        try:
            sync.upload_results(run_dir)
        except Exception as e:  # noqa: BLE001 - the run itself already succeeded
            log(f"final result upload failed: {e}")
    log(f"done: {run_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
