"""RunPod launcher (RUNPOD.md sections 5-6; HANDOFF_TASKS.md task 22).

Every function that talks to RunPod takes a `client` argument (default: the
real `runpod` module) so tests can pass a fake with the same attribute names
(`create_pod`, `get_pod`, `get_pods`, `terminate_pod`, `api_key`).

Default (no S3): jobs write to /workspace/runs/<run_id> on the pod's own disk
and `collect` pulls it over SSH/scp before the pod is terminated. A job that
is preempted before `collect` runs loses its progress beyond the last local
checkpoint on that pod's disk -- there is no off-pod copy until collect runs.
Say this plainly in any run report. This is still the default because no S3
bucket is configured for this project.

Opt-in (`--s3-bucket`): the RUNPOD.md section 3 protocol is used instead, via
`tools/fetch_latest_checkpoint.py` before training and
`run_experiment.py --sync-bucket` during it, so every checkpoint lands off the
pod as it is written and a preemption costs only the work since the last
upload. Credentials are forwarded from the launcher's own environment (see
`s3_env`), never stored in a config or the state file. Untested against a real
bucket: no S3 credentials exist for this project yet.

Usage:
    python tools/runpod_launch.py estimate generated/a1_first_experiment
    python tools/runpod_launch.py launch generated/a1_first_experiment \\
        --max-concurrent 6 --git-ref <sha>
    python tools/runpod_launch.py status
    python tools/runpod_launch.py relaunch generated/a1_first_experiment
    python tools/runpod_launch.py watchdog
    python tools/runpod_launch.py collect generated/a1_first_experiment --out results/
    python tools/runpod_launch.py reap --prefix bdhx-a1_first_experiment
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdhx.config import PROJECT_ROOT

DEFAULT_STATE_FILE = PROJECT_ROOT / "runpod_state.jsonl"
DEFAULT_RATES_FILE = PROJECT_ROOT / "configs" / "runpod_rates.yaml"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu"
DEFAULT_GPU_TYPE = "NVIDIA RTX A5000"
DEFAULT_CLOUD_TYPE = "COMMUNITY"
REPO_URL = "https://github.com/chris-lee-mc/prism-llama-cpu-kernels"
REPO_SUBDIR = "bdh-cq-experiments"

RUNNING_STATUSES = {"RUNNING"}
TERMINAL_STATUSES = {"EXITED", "TERMINATED"}


# -- manifest / cost -----------------------------------------------------


@dataclass
class ManifestJob:
    exp: str
    config_hash: str
    seed: int
    model: str
    task: str
    control_of: str
    est_gpu_minutes: float | None

    @property
    def run_id(self) -> str:
        return f"{self.config_hash}_s{self.seed}"


def read_manifest(generated_dir: Path) -> list[ManifestJob]:
    path = Path(generated_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.csv in {generated_dir}; run generate_sweep.py first")
    jobs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            minutes = float(row["est_gpu_minutes"]) if row["est_gpu_minutes"] else None
            jobs.append(
                ManifestJob(
                    exp=row["exp"],
                    config_hash=row["config_hash"],
                    seed=int(row["seed"]),
                    model=row["model"],
                    task=row["task"],
                    control_of=row["control_of"],
                    est_gpu_minutes=minutes,
                )
            )
    return jobs


def load_rates(rates_path: Path = DEFAULT_RATES_FILE) -> dict:
    return yaml.safe_load(Path(rates_path).read_text())


def gpu_rate(rates: dict, gpu_type: str, cloud_type: str) -> float:
    try:
        return float(rates["rates"][gpu_type][cloud_type])
    except KeyError as e:
        raise KeyError(
            f"no rate for gpu_type={gpu_type!r} cloud_type={cloud_type!r} in "
            f"{DEFAULT_RATES_FILE}; add it before estimating"
        ) from e


def estimate(
    generated_dir: Path,
    gpu_type: str = DEFAULT_GPU_TYPE,
    cloud_type: str = DEFAULT_CLOUD_TYPE,
    rates_path: Path = DEFAULT_RATES_FILE,
) -> dict:
    jobs = read_manifest(generated_dir)
    missing = [j.exp for j in jobs if j.est_gpu_minutes is None]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(jobs)} jobs have no est_gpu_minutes in the manifest "
            f"(missing: {missing[:5]}{'...' if len(missing) > 5 else ''}); "
            "re-run generate_sweep.py with --estimates after profiling"
        )
    rates = load_rates(rates_path)
    rate = gpu_rate(rates, gpu_type, cloud_type)
    contingency = float(rates.get("contingency_multiplier", 1.3))
    total_minutes = sum(j.est_gpu_minutes for j in jobs)
    gpu_hours = total_minutes / 60.0
    usd = gpu_hours * rate
    usd_with_contingency = usd * contingency
    return {
        "generated_dir": str(generated_dir),
        "n_jobs": len(jobs),
        "gpu_type": gpu_type,
        "cloud_type": cloud_type,
        "rate_per_hr": rate,
        "gpu_hours": gpu_hours,
        "contingency_multiplier": contingency,
        "usd": usd,
        "usd_with_contingency": usd_with_contingency,
    }


def print_estimate(est: dict) -> None:
    print(f"jobs                  {est['n_jobs']}")
    print(f"gpu type / cloud      {est['gpu_type']} / {est['cloud_type']}")
    print(f"rate                  ${est['rate_per_hr']:.2f}/hr")
    print(f"GPU-hours             {est['gpu_hours']:.2f}")
    print(f"USD (no contingency)  ${est['usd']:.2f}")
    print(f"USD ({est['contingency_multiplier']}x contingency)  ${est['usd_with_contingency']:.2f}")


# -- state file (append-only) ---------------------------------------------


@dataclass
class PodRecord:
    sweep: str
    exp: str
    run_id: str
    pod_id: str | None  # None means QUEUED, not yet created
    gpu_type: str
    cloud_type: str
    max_seconds: int
    config_path: str
    created_at: float = field(default_factory=time.time)
    name: str = ""


def append_state(record: PodRecord, state_path: Path = DEFAULT_STATE_FILE) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def read_state(state_path: Path = DEFAULT_STATE_FILE) -> list[dict]:
    if not Path(state_path).exists():
        return []
    rows = []
    with open(state_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_state_by_run_id(state_path: Path = DEFAULT_STATE_FILE) -> dict[str, dict]:
    """Last record per run_id (a run_id may appear more than once across relaunches)."""
    out: dict[str, dict] = {}
    for row in read_state(state_path):
        out[row["run_id"]] = row
    return out


# -- launch -----------------------------------------------------------------


def build_docker_args(cfg: PodRecord, git_ref: str, *, s3_bucket: str | None = None) -> str:
    """Startup command for the stock runpod/pytorch image (no Dockerfile needed).

    Deliberately does not embed RUNPOD_API_KEY or self-terminate: the account's
    master key is not distributed to every pod. Termination is the launcher's
    job (collect --terminate-on-collect, or watchdog/reap).

    `s3_bucket` is opt-in and off by default. Without it (the only mode in use
    today, since no bucket is configured) the command is byte-for-byte what it
    has always been: train to local disk, collect over scp. With it, the job
    additionally follows the RUNPOD.md section 3 protocol -- fetch the run's
    latest checkpoint before training, and sync every checkpoint off the pod
    during it -- so a preemption costs at most the work since the last upload
    instead of everything since the last `collect`. The credentials that go
    with the bucket are passed through `create_pod(env=...)`, never baked in
    here; see `s3_env`.
    """
    out_dir = f"/workspace/runs/{cfg.run_id}"
    fetch = ""
    sync = ""
    if s3_bucket:
        fetch = (
            f"python tools/fetch_latest_checkpoint.py --bucket {s3_bucket} "
            f"--run-id {cfg.run_id} --dest {out_dir} --config {cfg.config_path} && "
        )
        sync = f"--sync-bucket {s3_bucket} "
    return (
        "bash -lc '"
        "set -uo pipefail; "
        f"mkdir -p {out_dir}; "
        "cd /workspace; "
        f"if [ ! -d repo ]; then git clone --depth 200 {REPO_URL} repo; fi; "
        f"cd repo && git fetch origin {git_ref} && git checkout {git_ref} && "
        f"cd {REPO_SUBDIR} && "
        'pip install -q -e ".[gpu]" && '
        'pip install -q "bdh-cq @ git+https://github.com/lucidrains/bdh-cq@c246f890"; '
        f"{fetch}"
        f"timeout --signal=TERM {cfg.max_seconds}s python tools/run_experiment.py "
        f"--config {cfg.config_path} --run-id {cfg.run_id} --resume "
        f"{sync}"
        f"--out {out_dir} > {out_dir}/job.log 2>&1; "
        f"echo $? > {out_dir}/EXIT_CODE"
        "'"
    )


# S3 settings the pod needs to reach the bucket. Read from the launcher's own
# environment and forwarded through create_pod(env=...); nothing is ever read
# from a config file or written to the state file.
S3_ENV_VARS = (
    "S3_ENDPOINT_URL",
    "S3_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
)


def s3_env(bucket: str | None, environ=None) -> dict[str, str]:
    """The S3 env vars to hand a pod, or {} when no bucket is configured."""
    if not bucket:
        return {}
    environ = os.environ if environ is None else environ
    env = {"S3_BUCKET": bucket}
    for name in S3_ENV_VARS:
        value = environ.get(name)
        if value:
            env[name] = value
    return env


def launch(
    generated_dir: Path,
    git_ref: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    cloud_type: str = DEFAULT_CLOUD_TYPE,
    max_concurrent: int = 6,
    max_gpu_hours: float = 20.0,
    max_jobs: int = 12,
    allow_large_sweep: bool = False,
    max_wall_clock_minutes: int = 180,
    image: str = DEFAULT_IMAGE,
    state_path: Path = DEFAULT_STATE_FILE,
    rates_path: Path = DEFAULT_RATES_FILE,
    client=None,
    dry_run: bool = False,
    s3_bucket: str | None = None,
) -> list[PodRecord]:
    if client is None:
        import runpod as client

    generated_dir = Path(generated_dir)
    sweep = generated_dir.name
    jobs = read_manifest(generated_dir)
    est = estimate(generated_dir, gpu_type, cloud_type, rates_path)
    if (len(jobs) > max_jobs or est["gpu_hours"] > max_gpu_hours) and not allow_large_sweep:
        raise ValueError(
            f"sweep has {len(jobs)} jobs / {est['gpu_hours']:.1f} GPU-hours "
            f"(limits: --max-jobs={max_jobs}, --max-gpu-hours={max_gpu_hours}); "
            "pass --allow-large-sweep to proceed"
        )
    existing = latest_state_by_run_id(state_path)
    already_running = sum(
        1 for r in existing.values() if r.get("sweep") == sweep and r.get("pod_id")
    )
    slots = max(0, max_concurrent - already_running)
    max_seconds = max_wall_clock_minutes * 60

    created: list[PodRecord] = []
    for job in jobs:
        if job.run_id in existing and existing[job.run_id].get("pod_id"):
            continue  # already launched (or queued) this run_id; use relaunch for retries
        name = f"bdhx-{sweep}-{job.exp}"
        rel = os.path.relpath(generated_dir, PROJECT_ROOT)
        config_path = f"{REPO_SUBDIR}/{rel}/{job.exp}.yaml"
        rec = PodRecord(
            sweep=sweep,
            exp=job.exp,
            run_id=job.run_id,
            pod_id=None,
            gpu_type=gpu_type,
            cloud_type=cloud_type,
            max_seconds=max_seconds,
            config_path=config_path,
            name=name,
        )
        if len(created) >= slots or dry_run:
            append_state(rec, state_path)  # QUEUED: no pod_id yet
            created.append(rec)
            continue
        docker_args = build_docker_args(rec, git_ref, s3_bucket=s3_bucket)
        pod = client.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type,
            cloud_type=cloud_type,
            gpu_count=1,
            container_disk_in_gb=20,
            docker_args=docker_args,
            ports="22/tcp",
            env={
                "RUN_ID": rec.run_id,
                "MAX_SECONDS": str(max_seconds),
                **s3_env(s3_bucket),
            },
        )
        rec.pod_id = pod["id"] if isinstance(pod, dict) else pod
        append_state(rec, state_path)  # cost-safety rule: append BEFORE moving on
        created.append(rec)
    return created


# -- status / relaunch / watchdog -------------------------------------------


def classify(pod: dict | None) -> str:
    if pod is None:
        return "MISSING"
    status = (pod.get("desiredStatus") or "").upper()
    if status in RUNNING_STATUSES:
        return "RUNNING"
    if status in TERMINAL_STATUSES:
        return "EXITED"
    return status or "UNKNOWN"


def status(state_path: Path = DEFAULT_STATE_FILE, client=None) -> list[dict]:
    if client is None:
        import runpod as client

    rows = []
    for rec in latest_state_by_run_id(state_path).values():
        if not rec.get("pod_id"):
            rows.append({**rec, "state": "QUEUED"})
            continue
        try:
            pod = client.get_pod(rec["pod_id"])
        except Exception:  # noqa: BLE001 - a not-found pod means MISSING, not a crash
            pod = None
        rows.append({**rec, "state": classify(pod), "pod": pod})
    return rows


def print_status(rows: list[dict]) -> None:
    for r in rows:
        cost = ""
        pod = r.get("pod")
        if pod and pod.get("costPerHr") is not None and pod.get("uptimeSeconds") is not None:
            cost = f"  ${float(pod['costPerHr']) * float(pod['uptimeSeconds']) / 3600:.3f} so far"
        print(f"{r['run_id']:40s} {r['state']:10s} pod={r.get('pod_id') or '-'}{cost}")


def relaunch(
    generated_dir: Path,
    git_ref: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    cloud_type: str = DEFAULT_CLOUD_TYPE,
    max_concurrent: int = 6,
    max_wall_clock_minutes: int = 180,
    image: str = DEFAULT_IMAGE,
    state_path: Path = DEFAULT_STATE_FILE,
    client=None,
    s3_bucket: str | None = None,
) -> list[PodRecord]:
    if client is None:
        import runpod as client

    rows = status(state_path, client=client)
    running = sum(1 for r in rows if r["state"] == "RUNNING")
    slots = max(0, max_concurrent - running)
    to_relaunch = [r for r in rows if r["state"] in ("QUEUED", "MISSING") and r["run_id"]]
    relaunched: list[PodRecord] = []
    max_seconds = max_wall_clock_minutes * 60
    for r in to_relaunch[:slots]:
        rec = PodRecord(
            sweep=r["sweep"],
            exp=r["exp"],
            run_id=r["run_id"],
            pod_id=None,
            gpu_type=r.get("gpu_type", gpu_type),
            cloud_type=r.get("cloud_type", cloud_type),
            max_seconds=r.get("max_seconds", max_seconds),
            config_path=r["config_path"],
            name=r.get("name", f"bdhx-{r['sweep']}-{r['exp']}"),
        )
        docker_args = build_docker_args(rec, git_ref, s3_bucket=s3_bucket)
        pod = client.create_pod(
            name=rec.name,
            image_name=image,
            gpu_type_id=rec.gpu_type,
            cloud_type=rec.cloud_type,
            gpu_count=1,
            container_disk_in_gb=20,
            docker_args=docker_args,
            ports="22/tcp",
            env={
                "RUN_ID": rec.run_id,
                "MAX_SECONDS": str(rec.max_seconds),
                **s3_env(s3_bucket),
            },
        )
        rec.pod_id = pod["id"] if isinstance(pod, dict) else pod
        append_state(rec, state_path)
        relaunched.append(rec)
    return relaunched


def watchdog(state_path: Path = DEFAULT_STATE_FILE, client=None, grace: float = 1.5) -> list[str]:
    if client is None:
        import runpod as client

    terminated = []
    for r in status(state_path, client=client):
        pod = r.get("pod")
        if not pod or r["state"] != "RUNNING":
            continue
        uptime = pod.get("uptimeSeconds")
        limit = r.get("max_seconds")
        if uptime is not None and limit and uptime > grace * limit:
            client.terminate_pod(r["pod_id"])
            terminated.append(r["pod_id"])
    return terminated


# -- reap ---------------------------------------------------------------


def reap(prefix: str, client=None) -> dict:
    if client is None:
        import runpod as client

    pods = client.get_pods()
    matched = [p for p in pods if str(p.get("name", "")).startswith(prefix)]
    terminated, failed = [], []
    for p in matched:
        try:
            client.terminate_pod(p["id"])
            terminated.append(p["id"])
        except Exception as e:  # noqa: BLE001 - report, don't crash the safety net
            failed.append((p["id"], str(e)))
    return {"matched": len(matched), "terminated": terminated, "failed": failed}


# -- collect --------------------------------------------------------------


def pod_ssh_target(pod: dict) -> tuple[str, int] | None:
    for port in (pod.get("runtime") or {}).get("ports", []) or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], int(port["publicPort"])
    return None


def collect(
    generated_dir: Path,
    out_dir: Path,
    state_path: Path = DEFAULT_STATE_FILE,
    client=None,
    terminate_on_collect: bool = False,
    ssh_user: str = "root",
) -> dict:
    if client is None:
        import runpod as client

    sweep = Path(generated_dir).name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pulled, skipped = [], []
    for run_id, rec in latest_state_by_run_id(state_path).items():
        if rec.get("sweep") != sweep or not rec.get("pod_id"):
            continue
        try:
            pod = client.get_pod(rec["pod_id"])
        except Exception:  # noqa: BLE001 - a not-found pod is just uncollectable
            pod = None
        target = pod_ssh_target(pod) if pod else None
        if target is None:
            skipped.append({"run_id": run_id, "reason": "no reachable ssh port (pod gone?)"})
            continue
        ip, port = target
        dest = out_dir / run_id
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=15",
            "-P",
            str(port),
            "-r",
            f"{ssh_user}@{ip}:/workspace/runs/{run_id}/*",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            skipped.append({"run_id": run_id, "reason": result.stderr.strip()[-300:]})
            continue
        pulled.append(run_id)
        if terminate_on_collect:
            try:
                client.terminate_pod(rec["pod_id"])
            except Exception:  # noqa: BLE001, S110 - reap is the final safety net
                pass
    return {"pulled": pulled, "skipped": skipped}


# -- CLI ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_est = sub.add_parser("estimate")
    p_est.add_argument("generated_dir")
    p_est.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    p_est.add_argument("--cloud-type", default=DEFAULT_CLOUD_TYPE)

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("generated_dir")
    p_launch.add_argument("--git-ref", required=True, help="commit sha or branch to check out")
    p_launch.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    p_launch.add_argument("--cloud-type", default=DEFAULT_CLOUD_TYPE)
    p_launch.add_argument("--max-concurrent", type=int, default=6)
    p_launch.add_argument("--max-gpu-hours", type=float, default=20.0)
    p_launch.add_argument("--max-jobs", type=int, default=12)
    p_launch.add_argument("--allow-large-sweep", action="store_true")
    p_launch.add_argument("--max-wall-clock-minutes", type=int, default=180)
    p_launch.add_argument("--image", default=DEFAULT_IMAGE)
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.add_argument(
        "--s3-bucket",
        default=None,
        help="opt in to off-pod checkpoint sync (RUNPOD.md section 3); "
        "default is local disk plus `collect` over scp",
    )

    sub.add_parser("status")

    p_relaunch = sub.add_parser("relaunch")
    p_relaunch.add_argument("generated_dir")
    p_relaunch.add_argument("--git-ref", required=True)
    p_relaunch.add_argument("--max-concurrent", type=int, default=6)
    p_relaunch.add_argument("--s3-bucket", default=None)

    sub.add_parser("watchdog")

    p_reap = sub.add_parser("reap")
    p_reap.add_argument("--prefix", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("generated_dir")
    p_collect.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    p_collect.add_argument("--terminate-on-collect", action="store_true")

    args = parser.parse_args(argv)

    try:
        return _dispatch(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "estimate":
        est = estimate(args.generated_dir, args.gpu_type, args.cloud_type)
        print_estimate(est)
    elif args.cmd == "launch":
        created = launch(
            args.generated_dir,
            args.git_ref,
            gpu_type=args.gpu_type,
            cloud_type=args.cloud_type,
            max_concurrent=args.max_concurrent,
            max_gpu_hours=args.max_gpu_hours,
            max_jobs=args.max_jobs,
            allow_large_sweep=args.allow_large_sweep,
            max_wall_clock_minutes=args.max_wall_clock_minutes,
            image=args.image,
            dry_run=args.dry_run,
            s3_bucket=args.s3_bucket,
        )
        launched = sum(1 for r in created if r.pod_id)
        print(f"{launched} pods created, {len(created) - launched} queued")
    elif args.cmd == "status":
        print_status(status())
    elif args.cmd == "relaunch":
        r = relaunch(
            args.generated_dir,
            args.git_ref,
            max_concurrent=args.max_concurrent,
            s3_bucket=args.s3_bucket,
        )
        print(f"relaunched {len(r)} jobs")
    elif args.cmd == "watchdog":
        t = watchdog()
        print(f"terminated {len(t)} pods over 1.5x their wall-clock limit")
    elif args.cmd == "reap":
        r = reap(args.prefix)
        print(f"matched {r['matched']} pods, terminated {len(r['terminated'])}")
        if r["failed"]:
            print(f"FAILED to terminate: {r['failed']}")
            return 1
    elif args.cmd == "collect":
        r = collect(args.generated_dir, args.out, terminate_on_collect=args.terminate_on_collect)
        print(f"pulled {len(r['pulled'])} run dirs, skipped {len(r['skipped'])}")
        for s in r["skipped"]:
            print(f"  SKIPPED {s['run_id']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
