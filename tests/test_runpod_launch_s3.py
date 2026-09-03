"""The launcher's opt-in S3 path (HANDOFF_TASKS.md task 21).

`tools/runpod_launch.py` works today without any bucket, and that has to keep
being true: these tests pin the default startup command byte-for-byte and only
then check what `--s3-bucket` adds. The existing suite in
`tests/test_runpod_launch.py` covers the default path and is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.runpod_launch import PodRecord, build_docker_args, s3_env

GIT_REF = "abcdef1"


def record() -> PodRecord:
    return PodRecord(
        sweep="toy_sweep",
        exp="exp_000",
        run_id="h0_s1",
        pod_id=None,
        gpu_type="NVIDIA RTX A5000",
        cloud_type="COMMUNITY",
        max_seconds=600,
        config_path="bdh-cq-experiments/generated/toy_sweep/exp_000.yaml",
        name="bdhx-toy_sweep-exp_000",
    )


# -- the default path must not move ----------------------------------------


def test_default_startup_command_is_unchanged_by_the_s3_option():
    """No bucket means the command is exactly what it was before task 21."""
    assert build_docker_args(record(), GIT_REF) == build_docker_args(
        record(), GIT_REF, s3_bucket=None
    )


def test_default_startup_command_mentions_no_s3_anything():
    cmd = build_docker_args(record(), GIT_REF)
    assert "fetch_latest_checkpoint" not in cmd
    assert "--sync-bucket" not in cmd
    assert "S3" not in cmd


def test_default_startup_command_still_has_its_essential_parts():
    cmd = build_docker_args(record(), GIT_REF)
    assert "git clone" in cmd and GIT_REF in cmd
    assert 'pip install -q -e ".[gpu]"' in cmd
    assert "bdh-cq @ git+https://github.com/lucidrains/bdh-cq@c246f890" in cmd
    assert "timeout --signal=TERM 600s" in cmd
    assert "--run-id h0_s1" in cmd and "--resume" in cmd
    assert "EXIT_CODE" in cmd


def test_no_bucket_means_no_s3_env_is_handed_to_the_pod():
    assert s3_env(None) == {}
    assert s3_env("") == {}


# -- the opt-in path -------------------------------------------------------


def test_bucket_adds_a_fetch_before_training_and_a_sync_during_it():
    cmd = build_docker_args(record(), GIT_REF, s3_bucket="my-bucket")
    assert "tools/fetch_latest_checkpoint.py --bucket my-bucket" in cmd
    assert "--sync-bucket my-bucket" in cmd


def test_fetch_runs_before_the_trainer_and_gates_it():
    """A failed fetch must stop the job, not silently restart it from step 0."""
    cmd = build_docker_args(record(), GIT_REF, s3_bucket="my-bucket")
    assert cmd.index("fetch_latest_checkpoint.py") < cmd.index("run_experiment.py")
    fetch_to_train = cmd[cmd.index("fetch_latest_checkpoint.py") : cmd.index("timeout --signal")]
    assert fetch_to_train.rstrip().endswith("&&")


def test_fetch_checks_the_config_so_a_reused_run_id_cannot_resume_silently():
    cmd = build_docker_args(record(), GIT_REF, s3_bucket="my-bucket")
    assert "--config bdh-cq-experiments/generated/toy_sweep/exp_000.yaml" in cmd
    assert "--run-id h0_s1 --dest /workspace/runs/h0_s1" in cmd


def test_credentials_are_forwarded_from_the_launcher_environment():
    env = {
        "S3_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
        "S3_REGION": "auto",
        "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "UNRELATED": "ignored",
    }
    got = s3_env("my-bucket", env)
    assert got["S3_BUCKET"] == "my-bucket"
    assert got["S3_ENDPOINT_URL"] == env["S3_ENDPOINT_URL"]
    assert got["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
    assert "UNRELATED" not in got


def test_absent_credentials_are_simply_not_forwarded():
    assert s3_env("my-bucket", {}) == {"S3_BUCKET": "my-bucket"}


def test_no_secret_is_ever_baked_into_the_startup_command():
    """Credentials travel through create_pod(env=...), never the command line,
    which is visible in the RunPod console and in pod metadata."""
    cmd = build_docker_args(record(), GIT_REF, s3_bucket="my-bucket")
    for name in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "RUNPOD_API_KEY"):
        assert name not in cmd
