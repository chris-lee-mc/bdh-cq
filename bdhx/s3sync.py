"""S3-compatible checkpoint and result sync (RUNPOD.md section 3).

The authoritative off-pod store for a run is an S3-compatible bucket. This
module owns the key layout and the commit ordering; `tools/sync_checkpoint.py`
and `tools/fetch_latest_checkpoint.py` are thin CLIs over it, and
`tools/run_experiment.py` drives `CheckpointSync` from the trainer's
checkpoint hook.

Key layout for one run:

    runs/<run_id>/ckpt_<step:08d>.pt   the checkpoint payloads
    runs/<run_id>/latest.json          the pointer: key, step, config hash
    runs/<run_id>/results/<name>       results.json, train_log.csv, ...

Steps are zero padded to 8 digits so that a plain lexicographic listing is
also numeric order; nothing in this module has to sort parsed integers to
find the newest object.

Two deliberate deviations from a naive reading of RUNPOD.md section 3:

- Section 3 says `latest.json` is updated "with a write-then-rename". Object
  stores have no rename, and none of R2, B2 or the RunPod volume API expose
  one that is atomic. What that requirement actually buys is that a reader
  never sees a pointer to a checkpoint that is not fully stored, and a single
  S3 PutObject is already atomic at the object level. So the ordering is the
  guarantee here: the checkpoint object is uploaded and confirmed FIRST, and
  only then is `latest.json` overwritten. A crash between the two leaves an
  orphan checkpoint object (harmless, pruned later), never a dangling
  pointer. Where a genuine rename does exist -- the local filesystem, in
  `fetch_latest_checkpoint` -- write-then-`os.replace` is used literally.
- Section 3 says to delete checkpoints older than the last two. That prune
  runs against both the local run directory (the trainer already does it)
  and the bucket (`CheckpointSync.prune_remote`).

Credentials and endpoint come from the environment only, never from configs:

    S3_BUCKET           bucket name (RunPod: the network volume id)
    S3_ENDPOINT_URL     https://<account>.r2.cloudflarestorage.com,
                        https://s3.<region>.backblazeb2.com, or
                        https://s3api-<DATACENTER>.runpod.io/
    S3_REGION           region ("auto" for R2, the datacenter id for RunPod);
                        falls back to AWS_DEFAULT_REGION / AWS_REGION
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   read by botocore itself

`S3_ENDPOINT_URL` unset means real AWS S3.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_PREFIX = "ckpt_"
CHECKPOINT_SUFFIX = ".pt"
LATEST_NAME = "latest.json"
KEEP_REMOTE_CHECKPOINTS = 2

# Files uploaded by `--final` and after each evaluation (RUNPOD.md section 3).
RESULT_FILES = ("results.json", "train_log.csv", "metadata.json", "log.txt")

_CKPT_RE = re.compile(rf"^{CHECKPOINT_PREFIX}(\d+){re.escape(CHECKPOINT_SUFFIX)}$")


class ConfigHashMismatch(RuntimeError):
    """The stored run was trained under a different config than the one asked for.

    Fatal by design (RUNPOD.md section 3): resuming here would silently
    continue a different experiment under an existing run id.
    """


# -- key layout ------------------------------------------------------------


def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}"


def checkpoint_key(run_id: str, step: int) -> str:
    return f"{run_prefix(run_id)}/{CHECKPOINT_PREFIX}{int(step):08d}{CHECKPOINT_SUFFIX}"


def latest_key(run_id: str) -> str:
    return f"{run_prefix(run_id)}/{LATEST_NAME}"


def result_key(run_id: str, name: str) -> str:
    return f"{run_prefix(run_id)}/results/{name}"


def step_of_key(key: str) -> int | None:
    """Step encoded in a checkpoint key, or None if `key` is not one."""
    match = _CKPT_RE.match(key.rsplit("/", 1)[-1])
    return int(match.group(1)) if match else None


# -- settings and client ---------------------------------------------------


@dataclass(frozen=True)
class S3Settings:
    """Everything needed to build a client, read from the environment."""

    bucket: str
    endpoint_url: str | None = None
    region: str | None = None

    @classmethod
    def from_env(cls, bucket: str | None = None, env: dict[str, str] | None = None):
        """Settings, or None when no bucket is configured anywhere.

        Returning None (rather than raising) is what lets every caller treat
        S3 as optional: with no bucket the pod keeps results on local disk and
        `runpod_launch.py collect` pulls them over scp instead.
        """
        env = os.environ if env is None else env
        name = (bucket or env.get("S3_BUCKET") or "").strip()
        if not name:
            return None
        region = env.get("S3_REGION") or env.get("AWS_DEFAULT_REGION") or env.get("AWS_REGION")
        return cls(
            bucket=name.removeprefix("s3://").strip("/"),
            endpoint_url=(env.get("S3_ENDPOINT_URL") or "").strip() or None,
            region=(region or "").strip() or None,
        )


def make_client(settings: S3Settings, boto3_module: Any = None):
    """A boto3 S3 client configured for third-party S3-compatible endpoints.

    Two settings are not optional for non-AWS stores:

    - `request_checksum_calculation` / `response_checksum_validation` are
      forced to "when_required". Since botocore 1.36 the default is to send an
      `x-amz-checksum-crc32` header on every PutObject; Cloudflare R2 and
      Backblaze B2 both reject it ("Unsupported header" / "not implemented"),
      so the default would make every upload fail.
    - path-style addressing, because a virtual-host bucket name is not
      routable on the R2 / B2 / RunPod endpoint hostnames.
    """
    if boto3_module is None:
        try:
            import boto3 as boto3_module  # local import: optional dependency
        except ImportError as e:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "boto3 is required for S3 sync; install the extra: pip install -e '.[s3]'"
            ) from e
    from botocore.config import Config  # local import: ships with boto3

    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"max_attempts": 5, "mode": "standard"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3_module.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        region_name=settings.region,
        config=config,
    )


# -- sync ------------------------------------------------------------------


class CheckpointSync:
    """Uploads for one run. `client` is any object with the S3 client API."""

    def __init__(
        self,
        client,
        bucket: str,
        run_id: str,
        config_hash: str,
        *,
        keep: int = KEEP_REMOTE_CHECKPOINTS,
        log=None,
    ):
        self.client = client
        self.bucket = bucket
        self.run_id = run_id
        self.config_hash = config_hash
        self.keep = int(keep)
        self.log = log or (lambda msg: None)

    # -- reads -------------------------------------------------------------
    def read_latest(self) -> dict | None:
        """The `latest.json` pointer for this run, or None if there is none."""
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=latest_key(self.run_id))["Body"]
        except Exception as e:  # any not-found means "cold start"
            if not _is_missing(e):
                raise
            return None
        return json.loads(body.read().decode("utf-8"))

    def list_checkpoint_keys(self) -> list[str]:
        """Checkpoint keys for this run, oldest first (keys sort numerically)."""
        keys: list[str] = []
        token: str | None = None
        prefix = f"{run_prefix(self.run_id)}/{CHECKPOINT_PREFIX}"
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            keys.extend(
                obj["Key"]
                for obj in page.get("Contents", [])
                if step_of_key(obj["Key"]) is not None
            )
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated") or not token:
                break
        return sorted(keys)

    # -- writes ------------------------------------------------------------
    def upload_checkpoint(self, local_path: str | Path, step: int) -> str:
        """Upload one checkpoint and then publish it as `latest.json`.

        Order matters and is the whole point: the payload is stored before the
        pointer that names it, so a reader never resolves `latest.json` to a
        key that does not exist yet.
        """
        local_path = Path(local_path)
        key = checkpoint_key(self.run_id, step)
        with open(local_path, "rb") as fh:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=fh)
        self.client.put_object(
            Bucket=self.bucket,
            Key=latest_key(self.run_id),
            Body=json.dumps(
                {"key": key, "step": int(step), "config_hash": self.config_hash},
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            ContentType="application/json",
        )
        self.log(f"s3: uploaded {key} and published {latest_key(self.run_id)}")
        self.prune_remote()
        return key

    def prune_remote(self) -> list[str]:
        """Delete all but the newest `keep` checkpoint objects for this run."""
        keys = self.list_checkpoint_keys()
        stale = keys[: -self.keep] if self.keep > 0 else keys
        for key in stale:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        if stale:
            self.log(f"s3: pruned {len(stale)} old checkpoint objects")
        return stale

    def upload_results(self, run_dir: str | Path, names=RESULT_FILES) -> list[str]:
        """Upload whichever of the result files currently exist."""
        run_dir = Path(run_dir)
        uploaded = []
        for name in names:
            path = run_dir / name
            if not path.exists():
                continue
            key = result_key(self.run_id, name)
            with open(path, "rb") as fh:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=fh)
            uploaded.append(key)
        if uploaded:
            self.log(f"s3: uploaded {len(uploaded)} result files")
        return uploaded

    def upload_latest_local_checkpoint(self, run_dir: str | Path) -> str | None:
        """Upload the newest `checkpoints/step_*.pt` under `run_dir`, if any.

        This is the `--final` path: the trainer has already written its last
        checkpoint locally, and this pushes that exact file off the pod.
        """
        files = sorted(Path(run_dir).glob("checkpoints/step_*.pt"))
        if not files:
            self.log("s3: no local checkpoint to upload")
            return None
        newest = files[-1]
        step = int(newest.stem.split("_")[-1])
        return self.upload_checkpoint(newest, step)


# -- fetch -----------------------------------------------------------------


def fetch_latest_checkpoint(
    client,
    bucket: str,
    run_id: str,
    dest: str | Path,
    expected_config_hash: str | None = None,
) -> dict | None:
    """Download the run's newest checkpoint into `dest` so `--resume` finds it.

    Returns the `latest.json` payload, or None when the run has no checkpoint
    in the bucket yet (a cold start, which is not an error). Raises
    `ConfigHashMismatch` when the stored run used a different config than
    `expected_config_hash`: per RUNPOD.md section 3 that must abort loudly
    rather than silently train a different config under an old run id.

    The local pointer is written with write-then-rename, so a process killed
    mid-write leaves the previous pointer intact rather than a truncated one.
    """
    sync = CheckpointSync(client, bucket, run_id, expected_config_hash or "")
    latest = sync.read_latest()
    if latest is None:
        return None
    stored_hash = latest.get("config_hash")
    if expected_config_hash and stored_hash != expected_config_hash:
        raise ConfigHashMismatch(
            f"run_id {run_id!r} in s3://{bucket} was trained with config hash "
            f"{stored_hash!r}, but this job's config hashes to {expected_config_hash!r}; "
            "refusing to resume a different experiment under an existing run id"
        )
    key = latest["key"]
    step = int(latest["step"])
    ckpt_dir = Path(dest) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / f"step_{step:08d}.pt"
    tmp = target.with_suffix(".pt.part")
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    with open(tmp, "wb") as fh:
        fh.write(body.read())
    os.replace(tmp, target)

    pointer = ckpt_dir / LATEST_NAME
    pointer_tmp = pointer.with_suffix(".json.part")
    pointer_tmp.write_text(
        json.dumps({"path": str(target), "step": step, "reason": "fetched"}, indent=2)
    )
    os.replace(pointer_tmp, pointer)
    return latest


def _is_missing(exc: Exception) -> bool:
    """True for the several ways an S3 client reports "no such key"."""
    name = type(exc).__name__
    if name in ("NoSuchKey", "404", "ClientError"):
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        return str(code) in ("NoSuchKey", "NoSuchBucket", "404", "NotFound") or name == "NoSuchKey"
    return name in ("NoSuchKey", "KeyError", "FileNotFoundError")
