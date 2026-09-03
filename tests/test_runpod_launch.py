"""tools/runpod_launch.py against a fake RunPod SDK (HANDOFF_TASKS.md task 22)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.runpod_launch import (
    DEFAULT_RATES_FILE,
    ManifestJob,
    build_docker_args,
    collect,
    estimate,
    launch,
    read_manifest,
    reap,
    relaunch,
    status,
    watchdog,
)


def write_manifest(dir_: Path, rows: list[dict]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "config_hash", "seed", "model", "task", "control_of", "est_gpu_minutes"])
        for r in rows:
            w.writerow(
                [
                    r["exp"],
                    r["config_hash"],
                    r["seed"],
                    r["model"],
                    r["task"],
                    r.get("control_of", ""),
                    "" if r.get("est_gpu_minutes") is None else r["est_gpu_minutes"],
                ]
            )
        (dir_ / f"{rows[0]['exp']}.yaml").write_text("model: {}\n")
    return path


def three_job_manifest(tmp_path, minutes=50.0) -> Path:
    d = tmp_path / "generated" / "toy_sweep"
    rows = [
        {
            "exp": f"exp_{i:03d}",
            "config_hash": f"h{i}",
            "seed": i,
            "model": "bdh",
            "task": "compose",
            "est_gpu_minutes": minutes,
        }
        for i in range(3)
    ]
    write_manifest(d, rows)
    for r in rows:
        (d / f"{r['exp']}.yaml").write_text("model: {}\n")
    return d


class FakeRunpod:
    """Minimal fake of the parts of the runpod SDK the launcher touches."""

    def __init__(self):
        self.pods: dict[str, dict] = {}
        self._next_id = 0
        self.terminated: list[str] = []

    def create_pod(self, **kwargs):
        self._next_id += 1
        pid = f"pod-{self._next_id}"
        self.pods[pid] = {
            "id": pid,
            "name": kwargs["name"],
            "desiredStatus": "RUNNING",
            "uptimeSeconds": 0,
            "costPerHr": 0.16,
            "runtime": {
                "ports": [
                    {
                        "privatePort": 22,
                        "publicPort": 10000 + self._next_id,
                        "ip": "1.2.3.4",
                        "isIpPublic": True,
                    }
                ]
            },
        }
        return {"id": pid}

    def get_pod(self, pod_id):
        if pod_id not in self.pods:
            raise ValueError("not found")
        return self.pods[pod_id]

    def get_pods(self):
        return list(self.pods.values())

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        self.pods.pop(pod_id, None)


# -- estimate ---------------------------------------------------------------


def test_estimate_computes_gpu_hours_and_usd(tmp_path):
    d = three_job_manifest(tmp_path, minutes=60.0)  # 3 jobs * 1 hour = 3 GPU-hours
    est = estimate(d, gpu_type="NVIDIA RTX A5000", cloud_type="COMMUNITY")
    assert est["n_jobs"] == 3
    assert est["gpu_hours"] == pytest.approx(3.0)
    assert est["rate_per_hr"] == pytest.approx(0.16)
    assert est["usd"] == pytest.approx(3.0 * 0.16)
    assert est["usd_with_contingency"] == pytest.approx(3.0 * 0.16 * 1.3)


def test_estimate_refuses_missing_estimates(tmp_path):
    d = tmp_path / "generated" / "toy_sweep"
    rows = [
        {
            "exp": "exp_000",
            "config_hash": "h0",
            "seed": 1,
            "model": "bdh",
            "task": "compose",
            "est_gpu_minutes": None,
        }
    ]
    write_manifest(d, rows)
    with pytest.raises(ValueError, match="est_gpu_minutes"):
        estimate(d)


def test_estimate_unknown_gpu_type_raises(tmp_path):
    d = three_job_manifest(tmp_path)
    with pytest.raises(KeyError):
        estimate(d, gpu_type="NVIDIA Potato", cloud_type="COMMUNITY")


def test_rates_file_has_verified_a5000_and_4090():
    rates = yaml.safe_load(DEFAULT_RATES_FILE.read_text())
    assert rates["rates"]["NVIDIA RTX A5000"]["COMMUNITY"] == 0.16
    assert rates["rates"]["NVIDIA GeForce RTX 4090"]["COMMUNITY"] == 0.34


# -- launch / cost gate -------------------------------------------------


def test_launch_refuses_above_thresholds_without_allow_large_sweep(tmp_path):
    d = three_job_manifest(tmp_path, minutes=1000.0)  # 3 jobs * ~16.7h = way over 20h
    with pytest.raises(ValueError, match="allow-large-sweep"):
        launch(d, git_ref="deadbeef", client=FakeRunpod(), state_path=tmp_path / "state.jsonl")


def test_launch_creates_pods_and_appends_state_before_returning(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(d, git_ref="deadbeef", max_concurrent=2, client=client, state_path=state_path)
    assert len(created) == 3
    launched = [r for r in created if r.pod_id]
    queued = [r for r in created if not r.pod_id]
    assert len(launched) == 2  # respects max_concurrent
    assert len(queued) == 1
    assert len(client.pods) == 2
    # state file has one line per job, written before launch() returned
    lines = state_path.read_text().strip().splitlines()
    assert len(lines) == 3


def test_launch_dry_run_creates_no_pods(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    launch(d, git_ref="deadbeef", client=client, state_path=tmp_path / "state.jsonl", dry_run=True)
    assert len(client.pods) == 0


def test_docker_args_never_contains_api_key():
    from tools.runpod_launch import PodRecord

    rec = PodRecord(
        sweep="s",
        exp="exp_000",
        run_id="h0_s1",
        pod_id=None,
        gpu_type="g",
        cloud_type="COMMUNITY",
        max_seconds=100,
        config_path="cfg.yaml",
    )
    args = build_docker_args(rec, "deadbeef")
    assert "RUNPOD_API_KEY" not in args
    assert "h0_s1" in args and "cfg.yaml" in args


# -- status / relaunch / watchdog ----------------------------------------


def test_status_classifies_running_queued_missing(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(d, git_ref="x", max_concurrent=2, client=client, state_path=state_path)
    # kill one launched pod out from under the launcher (simulates preemption)
    dead_pod_id = created[0].pod_id
    del client.pods[dead_pod_id]

    rows = status(state_path, client=client)
    states = {r["run_id"]: r["state"] for r in rows}
    assert states[created[0].run_id] == "MISSING"
    assert states[created[1].run_id] == "RUNNING"
    assert states[created[2].run_id] == "QUEUED"


def test_relaunch_fills_freed_slots(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    launch(d, git_ref="x", max_concurrent=1, client=client, state_path=state_path)
    r = relaunch(d, git_ref="x", max_concurrent=2, state_path=state_path, client=client)
    assert len(r) == 1
    rows = status(state_path, client=client)
    running = [row for row in rows if row["state"] == "RUNNING"]
    assert len(running) == 2


def test_relaunch_resumes_same_run_id_after_missing_pod(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(d, git_ref="x", max_concurrent=1, client=client, state_path=state_path)
    del client.pods[created[0].pod_id]  # preempted
    relaunch(d, git_ref="x", max_concurrent=1, state_path=state_path, client=client)
    rows = status(state_path, client=client)
    row = next(r for r in rows if r["run_id"] == created[0].run_id)
    assert row["state"] == "RUNNING"
    # relaunch used the SAME run_id, so --resume in build_docker_args resumes it
    assert row["run_id"] == created[0].run_id


def test_watchdog_terminates_pods_over_grace_period(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(
        d,
        git_ref="x",
        max_concurrent=1,
        max_wall_clock_minutes=10,
        client=client,
        state_path=state_path,
    )
    pod = client.pods[created[0].pod_id]
    pod["uptimeSeconds"] = 10 * 60 * 1.6  # over 1.5x the 10-minute limit
    terminated = watchdog(state_path, client=client)
    assert terminated == [created[0].pod_id]
    assert created[0].pod_id not in client.pods


def test_watchdog_leaves_pods_within_grace_period(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(
        d,
        git_ref="x",
        max_concurrent=1,
        max_wall_clock_minutes=10,
        client=client,
        state_path=state_path,
    )
    pod = client.pods[created[0].pod_id]
    pod["uptimeSeconds"] = 10 * 60 * 1.2  # under 1.5x
    terminated = watchdog(state_path, client=client)
    assert terminated == []


# -- reap -----------------------------------------------------------------


def test_reap_terminates_only_matching_prefix(tmp_path):
    client = FakeRunpod()
    client.create_pod(name="bdhx-a1-exp_000")
    client.create_pod(name="bdhx-a1-exp_001")
    client.create_pod(name="unrelated-pod")
    result = reap("bdhx-a1", client=client)
    assert result["matched"] == 2
    assert len(result["terminated"]) == 2
    assert "unrelated-pod" not in [
        p for p in client.pods.values() if p.get("name") == "unrelated-pod"
    ]
    remaining_names = {p["name"] for p in client.pods.values()}
    assert remaining_names == {"unrelated-pod"}


def test_reap_reports_failures_without_crashing(tmp_path):
    client = FakeRunpod()
    client.create_pod(name="bdhx-a1-exp_000")

    def boom(pod_id):
        raise RuntimeError("api down")

    client.terminate_pod = boom
    result = reap("bdhx-a1", client=client)
    assert result["matched"] == 1
    assert result["terminated"] == []
    assert len(result["failed"]) == 1


def test_reap_empty_when_nothing_matches():
    client = FakeRunpod()
    client.create_pod(name="other-thing")
    result = reap("bdhx-a1", client=client)
    assert result == {"matched": 0, "terminated": [], "failed": []}


# -- collect --------------------------------------------------------------


def test_collect_skips_pods_with_no_reachable_ssh_port(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(d, git_ref="x", max_concurrent=1, client=client, state_path=state_path)
    client.pods[created[0].pod_id]["runtime"]["ports"] = []  # no ssh port exposed yet
    result = collect(d, tmp_path / "out", state_path=state_path, client=client)
    assert result["pulled"] == []
    assert len(result["skipped"]) == 1
    assert "ssh" in result["skipped"][0]["reason"]


def test_collect_skips_missing_pods_without_crashing(tmp_path):
    d = three_job_manifest(tmp_path)
    client = FakeRunpod()
    state_path = tmp_path / "state.jsonl"
    created = launch(d, git_ref="x", max_concurrent=1, client=client, state_path=state_path)
    del client.pods[created[0].pod_id]
    result = collect(d, tmp_path / "out", state_path=state_path, client=client)
    assert result["pulled"] == []
    assert result["skipped"][0]["run_id"] == created[0].run_id


# -- manifest reader ------------------------------------------------------


def test_read_manifest_run_id_matches_config_hash_and_seed(tmp_path):
    d = three_job_manifest(tmp_path)
    jobs = read_manifest(d)
    assert jobs[0].run_id == "h0_s0"
    assert all(isinstance(j, ManifestJob) for j in jobs)
