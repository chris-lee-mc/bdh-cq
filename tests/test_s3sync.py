"""S3 checkpoint sync against a fake S3 client (HANDOFF_TASKS.md task 21).

No test here needs credentials or a network: `FakeS3` implements the handful
of operations `bdhx/s3sync.py` uses, over an in-memory dict, and records the
call order so the RUNPOD.md section 3 commit ordering can be asserted.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdhx.s3sync import (
    CheckpointSync,
    ConfigHashMismatch,
    S3Settings,
    checkpoint_key,
    fetch_latest_checkpoint,
    latest_key,
    make_client,
    step_of_key,
)
from tools import fetch_latest_checkpoint as fetch_cli
from tools import sync_checkpoint as sync_cli

BUCKET = "test-bucket"
RUN_ID = "abc123_s1"
CHASH = "hash-of-this-config"


class NoSuchKey(Exception):
    """Stands in for botocore's generated NoSuchKey exception."""


class FakeS3:
    """Minimal fake of the S3 client API that bdhx/s3sync.py uses."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []

    def put_object(self, Bucket, Key, Body, **kwargs):  # boto3 argument spelling
        assert Bucket == BUCKET
        data = Body.read() if hasattr(Body, "read") else Body
        self.objects[Key] = data
        self.calls.append(("put", Key))
        return {}

    def get_object(self, Bucket, Key):  # boto3 argument spelling
        assert Bucket == BUCKET
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, Bucket, Key):  # boto3 argument spelling
        self.objects.pop(Key, None)
        self.calls.append(("delete", Key))
        return {}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


@pytest.fixture
def client():
    return FakeS3()


@pytest.fixture
def sync(client):
    return CheckpointSync(client, BUCKET, RUN_ID, CHASH)


def write_ckpt(run_dir: Path, step: int, payload: bytes = b"weights") -> Path:
    d = run_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step_{step:08d}.pt"
    path.write_bytes(payload)
    return path


# -- key layout -------------------------------------------------------------


def test_checkpoint_keys_are_zero_padded_so_lexicographic_order_is_numeric():
    keys = [checkpoint_key(RUN_ID, s) for s in (9, 10, 100, 2000)]
    assert keys == sorted(keys)
    assert checkpoint_key(RUN_ID, 40) == f"runs/{RUN_ID}/ckpt_00000040.pt"
    assert latest_key(RUN_ID) == f"runs/{RUN_ID}/latest.json"


def test_step_of_key_ignores_non_checkpoint_objects():
    assert step_of_key(checkpoint_key(RUN_ID, 7)) == 7
    assert step_of_key(latest_key(RUN_ID)) is None
    assert step_of_key(f"runs/{RUN_ID}/results/results.json") is None


# -- settings ---------------------------------------------------------------


def test_settings_are_none_without_a_bucket_so_s3_stays_optional():
    assert S3Settings.from_env(None, env={}) is None
    assert S3Settings.from_env(None, env={"S3_BUCKET": "  "}) is None


def test_settings_read_endpoint_and_region_from_env():
    s = S3Settings.from_env(
        None,
        env={
            "S3_BUCKET": "s3://my-bucket/",
            "S3_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
            "S3_REGION": "auto",
        },
    )
    assert s.bucket == "my-bucket"
    assert s.endpoint_url == "https://acct.r2.cloudflarestorage.com"
    assert s.region == "auto"


def test_settings_fall_back_to_standard_aws_region_vars():
    s = S3Settings.from_env(None, env={"S3_BUCKET": "b", "AWS_DEFAULT_REGION": "us-east-1"})
    assert s.region == "us-east-1"
    s = S3Settings.from_env(None, env={"S3_BUCKET": "b", "AWS_REGION": "eu-west-1"})
    assert s.region == "eu-west-1"


def test_explicit_bucket_argument_overrides_the_environment():
    s = S3Settings.from_env("cli-bucket", env={"S3_BUCKET": "env-bucket"})
    assert s.bucket == "cli-bucket"


def test_client_disables_default_checksums_that_r2_and_b2_reject():
    """Regression guard: botocore >= 1.36 sends x-amz-checksum-crc32 by default,
    which Cloudflare R2 and Backblaze B2 both reject, breaking every upload."""
    captured = {}

    class FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            captured.update(kwargs, service=service)
            return object()

    make_client(S3Settings(bucket="b", endpoint_url="https://x", region="auto"), FakeBoto3)
    cfg = captured["config"]
    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "https://x"
    assert captured["region_name"] == "auto"
    assert cfg.request_checksum_calculation == "when_required"
    assert cfg.response_checksum_validation == "when_required"
    assert cfg.s3["addressing_style"] == "path"


# -- upload ordering --------------------------------------------------------


def test_checkpoint_is_stored_before_the_pointer_that_names_it(tmp_path, sync, client):
    """latest.json must never resolve to an object that is not there yet."""
    path = write_ckpt(tmp_path, 40)
    sync.upload_checkpoint(path, 40)
    puts = [key for kind, key in client.calls if kind == "put"]
    assert puts.index(checkpoint_key(RUN_ID, 40)) < puts.index(latest_key(RUN_ID))


def test_latest_json_records_key_step_and_config_hash(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path, 40), 40)
    latest = json.loads(client.objects[latest_key(RUN_ID)])
    assert latest == {"key": checkpoint_key(RUN_ID, 40), "step": 40, "config_hash": CHASH}


def test_uploaded_checkpoint_bytes_match_the_local_file(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path, 5, b"exact-bytes"), 5)
    assert client.objects[checkpoint_key(RUN_ID, 5)] == b"exact-bytes"


def test_read_latest_is_none_for_a_run_with_no_checkpoint(sync):
    assert sync.read_latest() is None


# -- pruning ----------------------------------------------------------------


def test_prune_keeps_only_the_newest_two_remote_checkpoints(tmp_path, sync, client):
    for step in (10, 20, 30, 40):
        sync.upload_checkpoint(write_ckpt(tmp_path, step), step)
    remaining = sorted(k for k in client.objects if step_of_key(k) is not None)
    assert remaining == [checkpoint_key(RUN_ID, 30), checkpoint_key(RUN_ID, 40)]


def test_prune_never_deletes_the_pointer_or_results(tmp_path, sync, client):
    for step in (10, 20, 30):
        sync.upload_checkpoint(write_ckpt(tmp_path, step), step)
    (tmp_path / "results.json").write_text("{}")
    sync.upload_results(tmp_path)
    sync.prune_remote()
    assert latest_key(RUN_ID) in client.objects
    assert f"runs/{RUN_ID}/results/results.json" in client.objects


def test_upload_latest_local_checkpoint_picks_the_newest_step(tmp_path, sync, client):
    write_ckpt(tmp_path, 10)
    write_ckpt(tmp_path, 200)
    assert sync.upload_latest_local_checkpoint(tmp_path) == checkpoint_key(RUN_ID, 200)


def test_upload_latest_local_checkpoint_is_a_noop_without_one(tmp_path, sync):
    assert sync.upload_latest_local_checkpoint(tmp_path) is None


def test_upload_results_skips_files_that_do_not_exist(tmp_path, sync, client):
    (tmp_path / "results.json").write_text('{"ok": true}')
    uploaded = sync.upload_results(tmp_path)
    assert uploaded == [f"runs/{RUN_ID}/results/results.json"]
    assert client.objects[uploaded[0]] == b'{"ok": true}'


# -- fetch / resume ---------------------------------------------------------


def test_fetch_returns_none_for_a_cold_run_id(tmp_path, client):
    assert fetch_latest_checkpoint(client, BUCKET, RUN_ID, tmp_path, CHASH) is None


def test_fetch_writes_the_checkpoint_and_a_pointer_the_trainer_can_read(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40, b"payload"), 40)

    dest = tmp_path / "dest"
    latest = fetch_latest_checkpoint(client, BUCKET, RUN_ID, dest, CHASH)

    assert latest["step"] == 40
    ckpt = dest / "checkpoints" / "step_00000040.pt"
    assert ckpt.read_bytes() == b"payload"
    pointer = json.loads((dest / "checkpoints" / "latest.json").read_text())
    assert pointer["path"] == str(ckpt) and pointer["step"] == 40
    # The trainer resolves resume through exactly this pointer file.
    assert Path(pointer["path"]).exists()


def test_fetch_leaves_no_partial_files_behind(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40), 40)
    dest = tmp_path / "dest"
    fetch_latest_checkpoint(client, BUCKET, RUN_ID, dest, CHASH)
    assert list((dest / "checkpoints").glob("*.part")) == []


def test_fetch_aborts_loudly_on_a_config_hash_mismatch(tmp_path, sync, client):
    """The one thing this must never do is resume a different experiment."""
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40), 40)
    with pytest.raises(ConfigHashMismatch, match="refusing to resume"):
        fetch_latest_checkpoint(client, BUCKET, RUN_ID, tmp_path / "dest", "a-different-hash")


def test_fetch_downloads_nothing_when_the_hash_mismatches(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40), 40)
    dest = tmp_path / "dest"
    with pytest.raises(ConfigHashMismatch):
        fetch_latest_checkpoint(client, BUCKET, RUN_ID, dest, "a-different-hash")
    assert not (dest / "checkpoints" / "step_00000040.pt").exists()


def test_fetch_without_an_expected_hash_still_resumes(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40), 40)
    latest = fetch_latest_checkpoint(client, BUCKET, RUN_ID, tmp_path / "dest", None)
    assert latest["step"] == 40


def test_a_second_fetch_after_more_training_gets_the_newer_step(tmp_path, sync, client):
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40), 40)
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 80), 80)
    latest = fetch_latest_checkpoint(client, BUCKET, RUN_ID, tmp_path / "dest", CHASH)
    assert latest["step"] == 80


# -- CLI behaviour ----------------------------------------------------------


def test_sync_cli_is_a_noop_without_a_bucket(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    code = sync_cli.main(["--run-id", RUN_ID, "--run-dir", str(tmp_path)])
    assert code == 0
    assert "nothing to sync" in capsys.readouterr().out


def test_fetch_cli_is_a_noop_without_a_bucket(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    code = fetch_cli.main(["--run-id", RUN_ID, "--dest", str(tmp_path)])
    assert code == 0
    assert "without a remote checkpoint" in capsys.readouterr().out


def test_fetch_cli_exits_3_on_a_hash_mismatch(tmp_path, monkeypatch, sync, client, capsys):
    sync.upload_checkpoint(write_ckpt(tmp_path / "src", 40), 40)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr(fetch_cli, "make_client", lambda settings: client)
    code = fetch_cli.main(
        ["--run-id", RUN_ID, "--dest", str(tmp_path), "--config-hash", "other-hash"]
    )
    assert code == fetch_cli.EXIT_HASH_MISMATCH
    assert "FATAL" in capsys.readouterr().err


def test_fetch_cli_exits_1_on_a_transport_error_not_0(tmp_path, monkeypatch):
    """A dead bucket must not look like a cold start, or the run restarts at 0."""

    class Broken:
        def get_object(self, **kwargs):
            raise TimeoutError("endpoint unreachable")

    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr(fetch_cli, "make_client", lambda settings: Broken())
    code = fetch_cli.main(["--run-id", RUN_ID, "--dest", str(tmp_path), "--config-hash", CHASH])
    assert code == fetch_cli.EXIT_ERROR


def test_sync_cli_uploads_the_newest_checkpoint_and_results(tmp_path, monkeypatch, client):
    run_dir = tmp_path / RUN_ID
    write_ckpt(run_dir, 40)
    (run_dir / "results.json").write_text(json.dumps({"config_hash": CHASH}))
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr(sync_cli, "make_client", lambda settings: client)

    code = sync_cli.main(["--run-id", RUN_ID, "--run-dir", str(run_dir), "--final"])

    assert code == 0
    assert checkpoint_key(RUN_ID, 40) in client.objects
    assert f"runs/{RUN_ID}/results/results.json" in client.objects
    assert json.loads(client.objects[latest_key(RUN_ID)])["config_hash"] == CHASH


def test_sync_cli_refuses_to_publish_without_a_config_hash(tmp_path, monkeypatch, client):
    """A latest.json with no hash could not be checked on resume."""
    run_dir = tmp_path / RUN_ID
    write_ckpt(run_dir, 40)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr(sync_cli, "make_client", lambda settings: client)
    with pytest.raises(SystemExit, match="config hash"):
        sync_cli.main(["--run-id", RUN_ID, "--run-dir", str(run_dir)])


def test_sync_cli_reports_failure_instead_of_exiting_0(tmp_path, monkeypatch):
    class Broken:
        def put_object(self, **kwargs):
            raise TimeoutError("endpoint unreachable")

    run_dir = tmp_path / RUN_ID
    write_ckpt(run_dir, 40)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr(sync_cli, "make_client", lambda settings: Broken())
    code = sync_cli.main(["--run-id", RUN_ID, "--run-dir", str(run_dir), "--config-hash", CHASH])
    assert code == 1
