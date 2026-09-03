"""Evaluation protocol (FRAMEWORK_SPEC section 6).

Evaluation always runs on the fixed cached episodes of `tasks.cache`: one row
per (step, reasoning_steps, split, difficulty), split labels mandatory.

Deviations, documented:

- Predictions come from the batched `forward_episode(batch, R)` path rather
  than from `solve` per episode; `forward_episode` starts from a fresh context
  built out of the batch (no cross-episode leakage) and is identical to
  `reset_context; ingest_context; solve` for single-token targets. Multi-token
  targets are scored teacher-forced. `solve(collect_diagnostics=True)` is still
  run, episode by episode, on the diagnostics subset.
- Missing shards are generated in memory with the same deterministic
  `(task_seed, split, index)` recipe `tools/generate_tasks.py` uses, so a run
  never silently evaluates on different episodes than a cached shard would give.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from bdhx.results.schema import AdaptationCost, Diagnostics, EvaluationRow
from bdhx.seeding import episode_id as make_episode_id
from bdhx.seeding import task_rng
from bdhx.tasks.base import Episode, EpisodicTask, pad_and_batch
from bdhx.tasks.cache import load_eval_episodes
from bdhx.training import diagnostics as diag_mod

EVAL_SPLITS = ("interp", "mild", "strong")
INTERMEDIATE_REASONING_STEPS = (1, 4, 16)
DIAG_EPISODES = 8


def data_root() -> Path | None:
    """`BDHX_DATA_ROOT` overrides the default shard root (tests, RunPod volumes)."""
    root = os.environ.get("BDHX_DATA_ROOT")
    return Path(root) if root else None


def generate_episodes(
    task_obj: EpisodicTask, task_seed: int, split: str, difficulties: list[dict], n: int
) -> list[Episode]:
    """In-memory mirror of `tools/generate_tasks.generate_split` (same rng recipe)."""
    episodes = []
    for i in range(n):
        difficulty = difficulties[i % len(difficulties)]
        ep = task_obj.sample(task_rng(task_seed, split, i), difficulty)
        episodes.append(
            Episode(
                demonstrations=ep.demonstrations,
                query=ep.query,
                target=ep.target,
                difficulty=ep.difficulty,
                split=split,
                episode_id=make_episode_id(task_seed, split, i),
                extras=ep.extras,
            )
        )
    return episodes


def load_episodes(
    task_obj: EpisodicTask,
    task_seed: int,
    split: str,
    n: int,
    root: Path | None = None,
    difficulties: list[dict] | None = None,
) -> list[Episode]:
    try:
        return load_eval_episodes(task_obj, task_seed, split, n, root=root or data_root())
    except (FileNotFoundError, ValueError):
        diffs = difficulties or task_obj.eval_difficulties()[split]
        return generate_episodes(task_obj, task_seed, split, diffs, n)


def group_by_difficulty(episodes: list[Episode]) -> list[tuple[dict, list[Episode]]]:
    groups: dict[str, tuple[dict, list[Episode]]] = {}
    for ep in episodes:
        key = repr(sorted(ep.difficulty.items()))
        groups.setdefault(key, (ep.difficulty, []))[1].append(ep)
    return list(groups.values())


def reduced_reasoning_steps(cfg) -> list[int]:
    """The intermediate-checkpoint depth list, [1, 4, 16] restricted to the config."""
    configured = list(cfg.evaluation.reasoning_steps)
    reduced = [r for r in INTERMEDIATE_REASONING_STEPS if r in configured]
    return reduced or configured[:1]


def _int_difficulty(difficulty: dict) -> dict[str, int]:
    return {k: int(v) for k, v in difficulty.items() if isinstance(v, (int, bool))}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def predictions_for(model, batch, reasoning_steps: int) -> torch.Tensor:
    logits = model.forward_episode(batch, reasoning_steps)
    return logits.argmax(dim=-1)


def score_episodes(task, predictions: torch.Tensor, episodes: list[Episode]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for i, ep in enumerate(episodes):
        for key, value in task.score(predictions[i].detach().cpu(), ep).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {k: v / max(len(episodes), 1) for k, v in totals.items()}


def episode_diagnostics(
    model,
    episodes: list[Episode],
    reasoning_steps: int,
    jacobian: bool,
    n_episodes: int = DIAG_EPISODES,
) -> tuple[dict[str, Any], float]:
    """Per-step series averaged over a bounded subset, plus mean context latency (ms)."""
    n = min(len(episodes), max(int(n_episodes), 1))
    series, ingest_ms = [], 0.0
    for ep in episodes[:n]:
        model.reset_context()
        t0 = time.perf_counter()
        model.ingest_context(ep.demonstrations)
        ingest_ms += (time.perf_counter() - t0) * 1000.0
        query = ep.query.unsqueeze(0)
        target_length = int(ep.target.numel())
        _, s = diag_mod.collect(
            model,
            lambda q=query, tl=target_length: model.solve(
                q, reasoning_steps, collect_diagnostics=True, target_length=tl
            ),
            reasoning_steps,
            jacobian=jacobian,
        )
        series.append(s)
        model.reset_context()
    return diag_mod.merge(series), ingest_ms / max(n, 1)


def evaluate_group(
    model,
    task,
    episodes: list[Episode],
    reasoning_steps: int,
    step: int,
    split: str,
    difficulty: dict,
    *,
    batch_size: int = 32,
    collect_diagnostics: bool = True,
    record_adaptation_cost: bool = False,
    device: torch.device | None = None,
) -> EvaluationRow:
    device = device or torch.device("cpu")
    totals: dict[str, float] = {}
    flops_per_episode = 0.0
    elapsed = 0.0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            batch = pad_and_batch(chunk, task).to(device)
            model.reset_context()
            _sync(device)
            t0 = time.perf_counter()
            preds = predictions_for(model, batch, reasoning_steps)
            _sync(device)
            elapsed += time.perf_counter() - t0
            flops_per_episode = model.flops_estimate(batch, reasoning_steps).per_episode
            scores = score_episodes(task, preds, chunk)
            for key, value in scores.items():
                totals[key] = totals.get(key, 0.0) + value * len(chunk)
    n = len(episodes)
    means = {k: v / max(n, 1) for k, v in totals.items()}
    task_metrics = {k: v for k, v in means.items() if k not in ("exact_match", "token_acc")}

    diagnostics = None
    adaptation = None
    if collect_diagnostics:
        jacobian = reasoning_steps in diag_mod.JACOBIAN_DEPTHS
        n_diag = diag_mod.JACOBIAN_EPISODES if jacobian else DIAG_EPISODES
        series, ingest_ms = episode_diagnostics(
            model, episodes, reasoning_steps, jacobian, n_episodes=n_diag
        )
        diagnostics = Diagnostics(**series)
        if record_adaptation_cost:
            adaptation = AdaptationCost(adaptation_latency_ms=ingest_ms)
    elif record_adaptation_cost:
        adaptation = AdaptationCost()

    return EvaluationRow(
        step=step,
        reasoning_steps=reasoning_steps,
        split=split,
        difficulty=_int_difficulty(difficulty),
        n_episodes=n,
        exact_match=means.get("exact_match", 0.0),
        token_acc=means.get("token_acc", 0.0),
        inference_flops_per_episode=flops_per_episode,
        latency_ms_per_episode=elapsed * 1000.0 / max(n, 1),
        task_metrics=task_metrics,
        adaptation=adaptation,
        diagnostics=diagnostics,
    )


def run_evaluation(
    model,
    task,
    cfg,
    step: int,
    *,
    reasoning_steps: list[int] | None = None,
    splits: tuple[str, ...] = EVAL_SPLITS,
    n_episodes: int | None = None,
    batch_size: int | None = None,
    writer=None,
    root: Path | None = None,
    device: torch.device | None = None,
    collect_diagnostics: bool | None = None,
) -> list[EvaluationRow]:
    """Runs the full (reasoning_steps x split x difficulty) grid and returns the rows."""
    steps = list(reasoning_steps or cfg.evaluation.reasoning_steps)
    n = int(n_episodes or cfg.task.n_eval_episodes)
    bs = int(batch_size or cfg.training.batch_size)
    diag = cfg.evaluation.diagnostics if collect_diagnostics is None else collect_diagnostics
    eval_difficulties = cfg.task.eval_difficulties or task.eval_difficulties()
    was_training = model.training
    model.eval()
    rows: list[EvaluationRow] = []
    try:
        for split in splits:
            episodes = load_episodes(
                task,
                cfg.task.seed,
                split,
                n,
                root=root,
                difficulties=eval_difficulties.get(split),
            )
            for difficulty, group in group_by_difficulty(episodes):
                for r in steps:
                    row = evaluate_group(
                        model,
                        task,
                        group,
                        int(r),
                        step,
                        split,
                        difficulty,
                        batch_size=bs,
                        collect_diagnostics=bool(diag),
                        record_adaptation_cost=bool(cfg.evaluation.record_adaptation_cost),
                        device=device,
                    )
                    rows.append(row)
                    if writer is not None:
                        writer.add_evaluation(row)
    finally:
        model.train(was_training)
        model.reset_context()
    if writer is not None:
        writer.flush()
    return rows
