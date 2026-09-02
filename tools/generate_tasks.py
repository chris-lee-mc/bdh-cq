"""Generate cached task shards (TASK_SUITE_SPEC section 4).

Usage:
    python tools/generate_tasks.py --task compose --seed 123 \
        --n_train 200000 --n_eval 2000 --out data/compose_s123/

Writes one npz shard per split (train/interp/mild/strong) plus a
manifest.json (task name, seed, knobs, split counts, git commit,
generator version).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

import bdhx.tasks  # noqa: F401  (registers task modules)
from bdhx.config import PROJECT_ROOT
from bdhx.registry import get_task
from bdhx.seeding import episode_id, task_rng
from bdhx.tasks.base import Episode, EpisodicTask

GENERATOR_VERSION = "0.1.0"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def generate_split(
    task_obj: EpisodicTask, task_seed: int, split: str, difficulties: list[dict], n: int
) -> list[Episode]:
    episodes = []
    for i in range(n):
        difficulty = difficulties[i % len(difficulties)]
        rng = task_rng(task_seed, split, i)
        eid = episode_id(task_seed, split, i)
        ep = task_obj.sample(rng, difficulty)
        episodes.append(
            Episode(
                demonstrations=ep.demonstrations,
                query=ep.query,
                target=ep.target,
                difficulty=ep.difficulty,
                split=split,
                episode_id=eid,
                extras=ep.extras,
            )
        )
    return episodes


_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def write_shard(task_obj: EpisodicTask, episodes: list[Episode], path: Path) -> None:
    """Write an npz shard with a fixed zip timestamp so identical content produces
    byte-identical files across runs (np.savez embeds the write time otherwise)."""
    arrays = {
        "serialized": np.array(
            [task_obj.serialize(ep).numpy().astype(np.int64) for ep in episodes], dtype=object
        ),
        "splits": np.array([ep.split for ep in episodes]),
        "difficulties": np.array([json.dumps(ep.difficulty, sort_keys=True) for ep in episodes]),
        "episode_ids": np.array([ep.episode_id for ep in episodes], dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arrays[key], allow_pickle=True)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_DATE_TIME)
            zf.writestr(info, buf.getvalue())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    task_cls = get_task(args.task)
    task_obj = task_cls()

    train_diff = task_obj.train_difficulties()
    eval_diff = task_obj.eval_difficulties()

    out_dir = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    split_counts = {}
    episodes_by_split = {}
    episodes_by_split["train"] = generate_split(
        task_obj, args.seed, "train", train_diff, args.n_train
    )
    split_counts["train"] = args.n_train
    for split in ("interp", "mild", "strong"):
        episodes_by_split[split] = generate_split(
            task_obj, args.seed, split, eval_diff[split], args.n_eval
        )
        split_counts[split] = args.n_eval

    for split, episodes in episodes_by_split.items():
        write_shard(task_obj, episodes, out_dir / f"{split}.npz")

    manifest = {
        "task": task_obj.name,
        "seed": args.seed,
        "knobs": {"train_difficulties": train_diff, "eval_difficulties": eval_diff},
        "split_counts": split_counts,
        "generator_version": GENERATOR_VERSION,
        "git_commit": _git_commit(),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
