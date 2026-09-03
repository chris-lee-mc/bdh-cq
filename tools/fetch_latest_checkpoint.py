"""Fetch a run's newest checkpoint from S3 so training can resume.

Implements the download half of the RUNPOD.md section 3 protocol: on start,
read `runs/<run_id>/latest.json`; if it exists and its config hash matches the
config this job is about to train, download the checkpoint it names into
`<dest>/checkpoints/` and write the local pointer that
`Trainer(resume=True)` reads. A hash mismatch aborts loudly -- the one thing
this must never do is silently train a different config under an old run id.

Usage:
    python tools/fetch_latest_checkpoint.py --run-id <id> --dest <dir> \
        [--bucket <name>] [--config <path> | --config-hash <hash>]

Exit codes:
    0  resumed, or nothing to resume from (no bucket, or a cold run id)
    1  transport or bucket error; the caller must NOT treat this as a cold
       start, because a checkpoint may exist and restarting from step 0 would
       overwrite it
    3  config hash mismatch (fatal, never retried)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdhx.s3sync import ConfigHashMismatch, S3Settings, fetch_latest_checkpoint, make_client

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HASH_MISMATCH = 3


def expected_hash(config_path: str | None, explicit: str | None) -> str | None:
    """The config hash to demand, from --config-hash or by hashing --config."""
    if explicit:
        return explicit
    if not config_path:
        return None
    from bdhx.config import config_hash, load_config  # local import: keeps --help cheap

    return config_hash(load_config(config_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dest", required=True, help="the run directory to resume into")
    parser.add_argument("--bucket", default=None, help="overrides $S3_BUCKET")
    parser.add_argument("--config", default=None, help="config to hash and check against")
    parser.add_argument("--config-hash", default=None, help="overrides --config")
    args = parser.parse_args(argv)

    settings = S3Settings.from_env(args.bucket)
    if settings is None:
        print("no S3_BUCKET configured; starting without a remote checkpoint", flush=True)
        return EXIT_OK

    want = expected_hash(args.config, args.config_hash)
    if not want:
        print(
            "warning: neither --config nor --config-hash given; resuming without verifying "
            "that the stored run used this config",
            file=sys.stderr,
        )

    try:
        latest = fetch_latest_checkpoint(
            make_client(settings), settings.bucket, args.run_id, args.dest, want
        )
    except ConfigHashMismatch as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return EXIT_HASH_MISMATCH
    except Exception as e:  # noqa: BLE001 - never let this look like a cold start
        print(f"error: could not read s3://{settings.bucket}: {e}", file=sys.stderr)
        return EXIT_ERROR

    if latest is None:
        print(f"no checkpoint in s3://{settings.bucket} for run {args.run_id}; cold start")
        return EXIT_OK
    print(f"fetched {latest['key']} (step {latest['step']}) into {args.dest}/checkpoints")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
