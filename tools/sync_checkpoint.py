"""Upload a run's checkpoint and results to the S3-compatible store.

Implements the upload half of the RUNPOD.md section 3 protocol. The trigger
policy it serves ("every `checkpoint_every_steps` AND at least every 5 minutes
of wall clock, whichever comes first") already lives in
`bdhx/training/trainer.py`; this tool is what runs on each of those triggers,
either through the trainer's `on_checkpoint` hook (driven by
`tools/run_experiment.py --sync-bucket`) or as the one-shot `--final` call at
the end of `docker/entrypoint.sh`.

Usage:
    python tools/sync_checkpoint.py --run-id <id> [--bucket <name>] \
        [--run-dir <dir>] [--step N | --final]

With no bucket configured (no --bucket and no S3_BUCKET) this is a deliberate
no-op that exits 0: most runs today have no bucket and are collected from the
pod over scp by `tools/runpod_launch.py collect` instead.

Exit codes: 0 sync done or nothing to do, 1 upload failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdhx.s3sync import CheckpointSync, S3Settings, make_client


def read_config_hash(run_dir: Path) -> str | None:
    """The run's config hash from whichever artefact already records it."""
    for name in ("results.json", "metadata.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text()).get("config_hash")
        except (OSError, ValueError):
            continue
        if value:
            return str(value)
    return None


def build_sync(bucket: str | None, run_id: str, run_dir: Path, config_hash: str | None, log):
    """A `CheckpointSync`, or None when no bucket is configured."""
    settings = S3Settings.from_env(bucket)
    if settings is None:
        return None
    chash = config_hash or read_config_hash(run_dir)
    if not chash:
        raise SystemExit(
            f"no config hash for run {run_id!r}: {run_dir}/results.json and metadata.json are "
            "missing or have no config_hash, and --config-hash was not given. Refusing to "
            "publish a latest.json that cannot be checked on resume."
        )
    return CheckpointSync(make_client(settings), settings.bucket, run_id, chash, log=log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket", default=None, help="overrides $S3_BUCKET")
    parser.add_argument("--run-dir", default=None, help="default: <out>/<run_id>, see --out")
    parser.add_argument("--out", default="results", help="run dirs live under here")
    parser.add_argument("--step", type=int, default=None, help="upload this step's checkpoint")
    parser.add_argument(
        "--final", action="store_true", help="also upload results.json, train_log.csv, metadata"
    )
    parser.add_argument("--config-hash", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out) / args.run_id

    def log(msg: str) -> None:
        print(msg, flush=True)

    sync = build_sync(args.bucket, args.run_id, run_dir, args.config_hash, log)
    if sync is None:
        log("no S3_BUCKET configured; nothing to sync (results stay on local disk)")
        return 0
    if not run_dir.exists():
        log(f"run dir {run_dir} does not exist; nothing to sync")
        return 0

    try:
        if args.step is not None:
            path = run_dir / "checkpoints" / f"step_{args.step:08d}.pt"
            if not path.exists():
                raise SystemExit(f"no checkpoint at {path}")
            sync.upload_checkpoint(path, args.step)
        else:
            sync.upload_latest_local_checkpoint(run_dir)
        if args.final:
            sync.upload_results(run_dir)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - a failed upload must not look like success
        print(f"error: sync failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
