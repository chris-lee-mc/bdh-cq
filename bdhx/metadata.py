"""Run metadata capture (FRAMEWORK_SPEC section 4)."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import torch

from bdhx.config import Config, config_hash

METADATA_FIELDS = (
    "git_commit",
    "git_dirty",
    "config_hash",
    "config",
    "seed",
    "task_seed",
    "hostname",
    "gpu_name",
    "gpu_count",
    "cuda_version",
    "cudnn_version",
    "driver_version",
    "torch_version",
    "python_version",
    "platform",
    "pip_freeze",
    "param_total",
    "param_trainable",
    "param_breakdown",
    "serialized_bytes",
    "solved_width",
    "start_time",
    "end_time",
    "wall_clock_train_s",
    "wall_clock_eval_s",
    "steps_completed",
    "examples_seen",
    "tokens_seen",
    "train_flops_estimate",
    "peak_vram_bytes",
    "nan_events",
    "checkpoint_path",
    "log_path",
    "resumed_from",
    "preemptions",
)


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _write_pip_freeze(run_dir: Path) -> str:
    path = run_dir / "pip_freeze.txt"
    lines = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]
    )
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def collect_metadata(cfg: Config, run_dir: str | Path) -> dict[str, Any]:
    """Write run_dir/metadata.json with every field of spec section 4."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    status = _git(["status", "--porcelain"])
    meta: dict[str, Any] = dict.fromkeys(METADATA_FIELDS)
    meta.update(
        {
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": None if status is None else bool(status),
            "config_hash": config_hash(cfg),
            "config": cfg.model_dump(mode="json"),
            "seed": cfg.training.seed,
            "task_seed": cfg.task.seed,
            "hostname": socket.gethostname(),
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_count": torch.cuda.device_count() if cuda else 0,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda else None,
            "driver_version": _nvidia_driver() if cuda else None,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pip_freeze": _write_pip_freeze(run_dir),
            "start_time": time.time(),
            "nan_events": 0,
            "preemptions": 0,
        }
    )
    _dump(run_dir, meta)
    return meta


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout else None


def _dump(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))


def load_metadata(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "metadata.json").read_text())


def update_metadata(run_dir: str | Path, **fields: Any) -> dict[str, Any]:
    """Merge `fields` into an existing metadata.json and rewrite it."""
    run_dir = Path(run_dir)
    meta = load_metadata(run_dir)
    unknown = set(fields) - set(METADATA_FIELDS)
    if unknown:
        raise KeyError(f"unknown metadata fields: {sorted(unknown)}")
    meta.update(fields)
    _dump(run_dir, meta)
    return meta
