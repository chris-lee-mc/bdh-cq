"""Aggregation and plots over results/ (FRAMEWORK_SPEC sections 9, 10).

`aggregate(results_root, out_dir)` walks `results/<run>/{results.json,
metadata.json}`, writes `all_runs.csv`, `summary.csv`, `flags.csv`, and the
six plots of section 10 (PNG + backing CSV) to `out_dir`.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bdhx.results.schema import ResultsFile, ResultsWriter

NOT_MATCHED_PARAMS_TOL = 0.05
NOT_MATCHED_FLOPS_TOL = 0.15
HIGH_VAR_CV = 0.3
# A run whose final train loss is still within this fraction of ln(vocab_size)
# never left the uniform-prediction plateau: it learned nothing (AT_CHANCE).
AT_CHANCE_TOL = 0.03
README_GRADE_SEEDS = 5
BOOTSTRAP_RESAMPLES = 1000


def _difficulty_shortener(keys):
    """Render difficulty dicts by the fields that actually differ between them.

    A propagate legend entry is `{"dim": 1, "distance": 4, "length": 12,
    "n_sources": 1}` where only `distance` varies; printing all four makes a
    legend wider than the figure and hides the one field the reader needs.
    """
    parsed = [json.loads(k) for k in dict.fromkeys(keys)]
    if not parsed:
        return lambda k: k
    varying = sorted({f for f in parsed[0] if len({str(p.get(f)) for p in parsed}) > 1})

    def render(key: str) -> str:
        d = json.loads(key)
        fields = varying or sorted(d)
        return " ".join(f"{f}={d[f]}" for f in fields if f in d)

    return render


def _difficulty_key(difficulty: dict[str, int]) -> str:
    return json.dumps(dict(sorted((difficulty or {}).items())))


@dataclass
class RunRecord:
    run_dir: Path
    results: ResultsFile
    metadata: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_dev(self) -> bool:
        tags = ((self.metadata or {}).get("config") or {}).get("experiment", {}).get("tags", [])
        return "dev" in tags

    @property
    def max_r_train(self) -> int | None:
        cfg = (self.metadata or {}).get("config")
        if not cfg:
            return None
        steps = cfg.get("reasoning", {}).get("train_steps")
        return max(steps) if steps else None

    @property
    def train_flops_estimate(self) -> float | None:
        return (self.metadata or {}).get("train_flops_estimate")

    @property
    def final_train_loss(self) -> float | None:
        """Last finite `loss` in `train_log.csv`, or None when there is no curve."""
        path = self.run_dir / "train_log.csv"
        if not path.exists():
            return None
        last = None
        try:
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        value = float(row.get("loss", ""))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        last = value
        except OSError:
            return None
        return last


def chance_loss(task_name: str) -> float:
    """Cross-entropy of the uniform distribution over the task vocabulary."""
    from bdhx.registry import get_task
    from bdhx.tasks.vocab import VOCAB_SIZE

    try:
        vocab = int(get_task(task_name)().vocab_size)
    except Exception:  # noqa: BLE001 - unknown task name: fall back to the shared vocab
        vocab = VOCAB_SIZE
    return math.log(max(vocab, 2))


def at_chance(final_loss: float | None, task_name: str, tol: float = AT_CHANCE_TOL) -> bool:
    """True when `final_loss` is still within `tol` of ln(vocab_size)."""
    if final_loss is None or not math.isfinite(final_loss):
        return False
    reference = chance_loss(task_name)
    return abs(final_loss - reference) <= tol * reference


def _dedupe_evaluations(evaluations: list) -> list:
    """Keep only the last evaluation row per (step, reasoning_steps, split,
    difficulty) within a single run.

    Found live on the real A1 sweep: a job whose pod got relaunched *after*
    it had already reached its final step (checkpoint intact, nothing left
    to train) still re-runs run_experiment.py's unconditional final
    `run_evaluation()` call once resumed. That call appends a fresh row for
    the same (step, reasoning_steps, split, difficulty) on top of the
    evaluations already carried over from the previous attempt's
    results.json, so a single run_id can end up with the same cell recorded
    2-3x over. build_summary() groups by that key across runs without
    deduping first, so every duplicate silently counted as an independent
    seed -- e.g. a 3-seed sweep cell read back with `n_seeds` in the 30s.
    """
    seen: dict[tuple, int] = {}
    deduped: list = []
    for row in evaluations:
        key = (row.step, row.reasoning_steps, row.split, _difficulty_key(row.difficulty))
        if key in seen:
            deduped[seen[key]] = row
        else:
            seen[key] = len(deduped)
            deduped.append(row)
    return deduped


def walk_runs(results_root: str | Path) -> list[RunRecord]:
    """Load every `results/<run>/results.json` under `results_root`."""
    root = Path(results_root)
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for results_path in sorted(root.glob("*/results.json")):
        run_dir = results_path.parent
        try:
            results = ResultsWriter.load(run_dir)
            results.evaluations = _dedupe_evaluations(results.evaluations)
        except Exception as exc:  # noqa: BLE001 - keep aggregating other runs
            records.append(
                RunRecord(run_dir, ResultsFile.model_construct(), None, [f"unreadable: {exc}"])
            )
            continue
        metadata = None
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                metadata = None
        record = RunRecord(run_dir, results, metadata)
        if metadata is None:
            record.warnings.append("missing metadata.json")
        records.append(record)
    return records


# -- all_runs.csv -----------------------------------------------------------

ALL_RUNS_COLUMNS = (
    "run_id",
    "run_dir",
    "config_hash",
    "seed",
    "model",
    "task",
    "params",
    "status",
    "dev",
    "step",
    "reasoning_steps",
    "split",
    "difficulty",
    "n_episodes",
    "exact_match",
    "token_acc",
    "inference_flops_per_episode",
    "latency_ms_per_episode",
    "task_metrics",
)


def build_all_runs_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        r = rec.results
        for ev in r.evaluations:
            rows.append(
                {
                    "run_id": r.run_id,
                    "run_dir": str(rec.run_dir),
                    "config_hash": r.config_hash,
                    "seed": r.seed,
                    "model": r.model,
                    "task": r.task,
                    "params": r.params,
                    "status": r.status,
                    "dev": rec.is_dev,
                    "step": ev.step,
                    "reasoning_steps": ev.reasoning_steps,
                    "split": ev.split,
                    "difficulty": _difficulty_key(ev.difficulty),
                    "n_episodes": ev.n_episodes,
                    "exact_match": ev.exact_match,
                    "token_acc": ev.token_acc,
                    "inference_flops_per_episode": ev.inference_flops_per_episode,
                    "latency_ms_per_episode": ev.latency_ms_per_episode,
                    "task_metrics": json.dumps(ev.task_metrics.model_dump(mode="json")),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path, columns: tuple[str, ...] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or (tuple(rows[0].keys()) if rows else ())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cols))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in cols})
    return path


# -- summary.csv --------------------------------------------------------------

SUMMARY_COLUMNS = (
    "config_hash",
    "model",
    "task",
    "step",
    "reasoning_steps",
    "split",
    "difficulty",
    "n_seeds",
    "provisional",
    "dev",
    "diverged_k",
    "exact_match_mean",
    "exact_match_std",
    "exact_match_min",
    "exact_match_max",
    "exact_match_values",
    "exact_match_ci95_lo",
    "exact_match_ci95_hi",
    "exact_match_mean_incl_failed",
    "token_acc_mean",
    "params",
)


def _bootstrap_ci95(
    values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else float("nan")
        return v, v
    rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def build_summary(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    for rec in records:
        r = rec.results
        for ev in r.evaluations:
            key = (
                r.config_hash,
                ev.step,
                ev.reasoning_steps,
                ev.split,
                _difficulty_key(ev.difficulty),
            )
            g = groups.setdefault(
                key,
                {
                    "config_hash": r.config_hash,
                    "model": r.model,
                    "task": r.task,
                    "step": ev.step,
                    "reasoning_steps": ev.reasoning_steps,
                    "split": ev.split,
                    "difficulty": _difficulty_key(ev.difficulty),
                    "params": r.params,
                    "exact_match": [],
                    "token_acc": [],
                    "dev": False,
                    "n_total": 0,
                    "diverged_k": 0,
                },
            )
            g["n_total"] += 1
            g["dev"] = g["dev"] or rec.is_dev
            if r.status == "diverged":
                g["diverged_k"] += 1
                g["exact_match"].append(0.0)
            else:
                g["exact_match"].append(ev.exact_match)
            g["token_acc"].append(ev.token_acc)

    summary = []
    for g in groups.values():
        values = g["exact_match"]
        n_seeds = g["n_total"]
        ok_values = [v for v in values]  # includes 0.0 for diverged, per spec section 8
        mean = float(np.mean(ok_values)) if ok_values else float("nan")
        std = float(np.std(ok_values)) if ok_values else float("nan")
        lo, hi = _bootstrap_ci95(ok_values)
        summary.append(
            {
                "config_hash": g["config_hash"],
                "model": g["model"],
                "task": g["task"],
                "step": g["step"],
                "reasoning_steps": g["reasoning_steps"],
                "split": g["split"],
                "difficulty": g["difficulty"],
                "n_seeds": n_seeds,
                "provisional": n_seeds < README_GRADE_SEEDS,
                "dev": g["dev"],
                "diverged_k": g["diverged_k"],
                "exact_match_mean": mean,
                "exact_match_std": std,
                "exact_match_min": float(np.min(ok_values)) if ok_values else float("nan"),
                "exact_match_max": float(np.max(ok_values)) if ok_values else float("nan"),
                "exact_match_values": json.dumps(values),
                "exact_match_ci95_lo": lo,
                "exact_match_ci95_hi": hi,
                "exact_match_mean_incl_failed": mean,
                "token_acc_mean": float(np.mean(g["token_acc"]))
                if g["token_acc"]
                else float("nan"),
                "params": g["params"],
            }
        )
    summary.sort(key=lambda r: (r["task"], r["model"], r["split"], r["reasoning_steps"]))
    return summary


# -- flags.csv ------------------------------------------------------------


def build_flags(records: list[RunRecord], summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    for row in summary:
        if row["provisional"]:
            flags.append(
                {
                    "flag": "PROVISIONAL",
                    "config_hash": row["config_hash"],
                    "model": row["model"],
                    "task": row["task"],
                    "detail": f"n_seeds={row['n_seeds']} < {README_GRADE_SEEDS}",
                }
            )
        if row["dev"]:
            flags.append(
                {
                    "flag": "DEV",
                    "config_hash": row["config_hash"],
                    "model": row["model"],
                    "task": row["task"],
                    "detail": "excluded from README-grade tables",
                }
            )
        if row["diverged_k"] > 0:
            flags.append(
                {
                    "flag": "DIVERGED",
                    "config_hash": row["config_hash"],
                    "model": row["model"],
                    "task": row["task"],
                    "detail": f"{row['diverged_k']}/{row['n_seeds']}",
                }
            )
        mean, std = row["exact_match_mean"], row["exact_match_std"]
        if row["n_seeds"] >= 2 and mean not in (0, None) and not np.isnan(mean) and mean > 0:
            cv = std / mean
            if cv > HIGH_VAR_CV:
                flags.append(
                    {
                        "flag": "HIGH VAR",
                        "config_hash": row["config_hash"],
                        "model": row["model"],
                        "task": row["task"],
                        "detail": f"cv={cv:.2f} (std={std:.3f}, mean={mean:.3f})",
                    }
                )

    for rec in records:
        loss = rec.final_train_loss
        if at_chance(loss, rec.results.task):
            flags.append(
                {
                    "flag": "AT_CHANCE",
                    "config_hash": rec.results.config_hash,
                    "model": rec.results.model,
                    "task": rec.results.task,
                    "detail": (
                        f"final train loss {loss:.3f} is within "
                        f"{AT_CHANCE_TOL:.0%} of ln(vocab)="
                        f"{chance_loss(rec.results.task):.3f}: the run learned nothing"
                    ),
                }
            )
        if rec.results.status == "incomplete":
            flags.append(
                {
                    "flag": "UNCONVERGED",
                    "config_hash": rec.results.config_hash,
                    "model": rec.results.model,
                    "task": rec.results.task,
                    "detail": f"run {rec.results.run_id} did not complete",
                }
            )

    # NOT MATCHED: compare params across config_hashes sharing an eval key.
    by_eval_key: dict[tuple, dict[str, Any]] = {}
    for row in summary:
        ek = (row["task"], row["step"], row["reasoning_steps"], row["split"], row["difficulty"])
        by_eval_key.setdefault(ek, {})[row["config_hash"]] = row

    # Worst mismatch per (task, comparison group), not per eval key.
    worst: dict[tuple[str, str], tuple[float, str, list[int]]] = {}
    for ek, by_hash in by_eval_key.items():
        if len(by_hash) < 2:
            continue
        params = {h: r["params"] for h, r in by_hash.items() if r["params"]}
        if len(params) < 2:
            continue
        lo, hi = min(params.values()), max(params.values())
        if lo <= 0:
            continue
        pct = (hi - lo) / lo
        if pct <= NOT_MATCHED_PARAMS_TOL:
            continue
        hashes = sorted(params)
        key = (ek[0], ",".join(hashes))
        models = ",".join(sorted({by_hash[h]["model"] for h in hashes}))
        prev = worst.get(key)
        if prev is None or pct > prev[0]:
            worst[key] = (pct, models, [lo, hi])

    for (task, hashes), (pct, models, (lo, hi)) in sorted(worst.items()):
        flags.append(
            {
                "flag": "NOT MATCHED",
                "config_hash": hashes,
                "model": models,
                "task": task,
                "detail": (
                    f"params differ by up to {pct * 100:.1f}% "
                    f"(> {NOT_MATCHED_PARAMS_TOL * 100:.0f}%): {lo} vs {hi}"
                ),
            }
        )

    # One flag per distinct (flag, config_hash, model, task, detail): summary rows
    # are per (R, split, difficulty), so the same condition fires many times.
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    for f in flags:
        key = tuple(f[c] for c in FLAGS_COLUMNS)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


FLAGS_COLUMNS = ("flag", "config_hash", "model", "task", "detail")


# -- recurrence convergence ---------------------------------------------------

CONVERGENCE_COLUMNS = (
    "model",
    "arm",
    "config_hash",
    "recurrence_kind",
    "task",
    "split",
    "difficulty",
    "reasoning_steps",
    "n_seeds",
    "cos_last",
    "cos_last_vs_seed",
    "cos_min_after_first",
    "token_acc",
    "exact_match",
)


def recurrence_kind(rec: RunRecord) -> str:
    cfg = (rec.metadata or {}).get("config") or {}
    return str(((cfg.get("model") or {}).get("recurrence") or {}).get("kind", ""))


def arm_label(rec: RunRecord) -> str:
    """What distinguishes one sweep arm from another in a plot legend.

    `results.model` is `cfg.model.name`, which is "bdh_cq" for EVERY arm of
    a4_convergence -- they differ only in `model.recurrence.kind`. Keying a
    table or a plot on the model name alone silently averages the update rules
    together and labels the result n_seeds=3, which is exactly the number the
    sweep is meant to compare.
    """
    kind = recurrence_kind(rec)
    return f"{rec.results.model}/{kind}" if kind else rec.results.model


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _convergence_measures(ev, cos: list[float]) -> dict[str, float]:
    """The convergence numbers read off one evaluation row.

    Shared by the final-step table and the onset table so the two cannot drift
    apart: a `cos_last` that means one thing in `recurrence_convergence_*.csv`
    and another in `convergence_onset_*.csv` would be worse than having only one
    of them.
    """
    return {
        "cos_last": cos[-1],
        # Iteration 0 compares against the ingested seed, not against
        # a previous latent, so it is excluded from the minimum.
        "cos_min_after_first": min(cos[1:]) if len(cos) > 1 else cos[-1],
        # ... which also means a single-iteration run's `cos_last` IS
        # that seed comparison, and is not the same quantity as the
        # R=32 rows it sits next to. Flagged rather than dropped, so
        # the row stays visible and uncomparable rather than absent.
        "cos_last_vs_seed": float(len(cos) == 1),
        "token_acc": ev.token_acc,
        "exact_match": ev.exact_match,
    }


def convergence_rows(records: list[RunRecord], task: str) -> list[dict[str, Any]]:
    """Per (model, split, difficulty, R_test): does the latent settle, and does it score?

    `cos_last` is `cos(H[R], H[R-1])` at the final iteration: 1.0 means the loop
    reached a fixed point, and a value that stops rising (or falls) as R grows
    means it never will. Reported next to accuracy at the same R so that the two
    can be read together, which is the comparison `EXPERIMENT_PLAN` section 6
    asks for and the pooled per-iteration plots cannot support.
    """
    cells: dict[tuple, list[dict[str, float]]] = {}
    for rec in records:
        if rec.results.task != task:
            continue
        # Per run, not global. A global max silently DROPS every run that
        # stopped short of it -- no row, no smaller n_seeds, no warning. That
        # is live risk, not theory: the pilots cap wall clock at 90 minutes
        # against an 80.1-minute estimate, and `Trainer` stops on that
        # deadline, so an overrunning job's final eval lands at e.g. 7400
        # steps instead of 8000 and the whole arm vanishes from this table.
        final_step = max((ev.step for ev in rec.results.evaluations), default=0)
        for ev in rec.results.evaluations:
            if ev.step != final_step or ev.diagnostics is None:
                continue
            cos = ev.diagnostics.cos_consecutive
            if not cos:
                continue
            key = (
                rec.results.model,
                arm_label(rec),
                rec.results.config_hash,
                recurrence_kind(rec),
                ev.split,
                _difficulty_key(ev.difficulty),
                ev.reasoning_steps,
            )
            cells.setdefault(key, []).append(_convergence_measures(ev, cos))
    rows = []
    for (model, arm, config_hash, kind, split, difficulty, r), seeds in sorted(cells.items()):
        rows.append(
            {
                "model": model,
                "arm": arm,
                "config_hash": config_hash,
                "recurrence_kind": kind,
                "task": task,
                "split": split,
                "difficulty": difficulty,
                "reasoning_steps": r,
                "n_seeds": len(seeds),
                **{
                    field_name: mean(s[field_name] for s in seeds)
                    for field_name in (
                        "cos_last",
                        "cos_last_vs_seed",
                        "cos_min_after_first",
                        "token_acc",
                        "exact_match",
                    )
                },
            }
        )
    return rows


ONSET_SPLIT = "interp"

ONSET_COLUMNS = (
    "model",
    "arm",
    "config_hash",
    "recurrence_kind",
    "task",
    "split",
    "difficulty",
    "reasoning_steps",
    "step",
    "n_seeds",
    "cos_last",
    "cos_last_vs_seed",
    "cos_min_after_first",
    "token_acc",
    "exact_match",
)


def convergence_onset_rows(records: list[RunRecord], task: str) -> list[dict[str, Any]]:
    """`convergence_rows`, but at every checkpoint instead of only the last one.

    The final-step table answers "does this arm's loop settle?". It cannot
    answer "when did it stop settling?", and those have different answers: on
    a1_first_experiment's bdh_cq/propagate arm `cos_last` at R=16 holds at
    0.985+ through step 12500, falls to 0.81 by 20000, and is then flat for the
    remaining 20000 steps -- the whole effect lives in a window the final row
    averages away. `intermediate_reasoning_steps` decides which R_test values
    have a curve here at all; the rest appear at one step and are dropped by
    the plot (see `make_plots`).
    """
    cells: dict[tuple, list[dict[str, float]]] = {}
    for rec in records:
        if rec.results.task != task:
            continue
        for ev in rec.results.evaluations:
            if ev.diagnostics is None:
                continue
            cos = ev.diagnostics.cos_consecutive
            if not cos:
                continue
            key = (
                rec.results.model,
                arm_label(rec),
                rec.results.config_hash,
                recurrence_kind(rec),
                ev.split,
                _difficulty_key(ev.difficulty),
                ev.reasoning_steps,
                ev.step,
            )
            cells.setdefault(key, []).append(_convergence_measures(ev, cos))
    rows = []
    for key, seeds in sorted(cells.items()):
        model, arm, config_hash, kind, split, difficulty, r, step = key
        rows.append(
            {
                "model": model,
                "arm": arm,
                "config_hash": config_hash,
                "recurrence_kind": kind,
                "task": task,
                "split": split,
                "difficulty": difficulty,
                "reasoning_steps": r,
                "step": step,
                "n_seeds": len(seeds),
                **{
                    field_name: mean(s[field_name] for s in seeds)
                    for field_name in (
                        "cos_last",
                        "cos_last_vs_seed",
                        "cos_min_after_first",
                        "token_acc",
                        "exact_match",
                    )
                },
            }
        )
    return rows


# -- plots ------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    """Write a figure with margins that fit its labels, then release it."""
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    _save(fig, path)


def _valid_rows(records: list[RunRecord]) -> list[RunRecord]:
    """Runs usable for plotting: metadata present and every row split-labelled."""
    out = []
    for rec in records:
        if rec.metadata is None:
            continue
        if any(not ev.split for ev in rec.results.evaluations):
            continue
        out.append(rec)
    return out


def make_plots(
    records: list[RunRecord], summary: list[dict[str, Any]], out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    usable = _valid_rows(records)
    tasks = sorted({r["task"] for r in summary}) or ["unknown"]

    r_train_by_hash = {rec.results.config_hash: rec.max_r_train for rec in usable}

    # 1. acc_vs_reasoning_steps_<task>.png
    for task in tasks:
        path = out_dir / f"acc_vs_reasoning_steps_{task}.png"
        rows = [r for r in summary if r["task"] == task]
        if not rows:
            _empty_plot(path, f"acc vs reasoning_steps ({task})")
        else:
            fig, ax = plt.subplots()
            lines: dict[tuple, list] = {}
            for r in rows:
                lines.setdefault((r["model"], r["split"]), []).append(r)
            for (model, split), pts in lines.items():
                pts.sort(key=lambda r: r["reasoning_steps"])
                xs = [p["reasoning_steps"] for p in pts]
                ys = [p["exact_match_mean"] for p in pts]
                es = [p["exact_match_std"] for p in pts]
                ax.errorbar(xs, ys, yerr=es, marker="o", label=f"{model}/{split}")
                for p in pts:
                    ax.annotate(f"n={p['n_seeds']}", (p["reasoning_steps"], p["exact_match_mean"]))
                r_train = r_train_by_hash.get(pts[0]["config_hash"])
                if r_train:
                    ax.axvline(r_train, linestyle="--", alpha=0.3)
            ax.set_xlabel("reasoning_steps")
            ax.set_ylabel("exact_match")
            ax.set_title(f"acc vs reasoning_steps ({task})")
            ax.legend(fontsize="small")
            _save(fig, path)
        write_csv(
            [r for r in rows] or [{}],
            out_dir / f"acc_vs_reasoning_steps_{task}.csv",
            SUMMARY_COLUMNS,
        )
        paths.append(path)

    # 2. acc_vs_difficulty_<task>.png
    for task in tasks:
        path = out_dir / f"acc_vs_difficulty_{task}.png"
        rows = [r for r in summary if r["task"] == task]
        if not rows:
            _empty_plot(path, f"acc vs difficulty ({task})")
        else:
            fig, ax = plt.subplots()
            lines: dict[tuple, list] = {}
            for r in rows:
                diff = json.loads(r["difficulty"])
                dscalar = max(diff.values()) if diff else 0
                lines.setdefault((r["model"], r["split"]), []).append((dscalar, r))
            colors = {"interp": "#88c", "mild": "#cc8", "strong": "#c88", "train": "#8c8"}
            for (model, split), pts in lines.items():
                pts.sort(key=lambda t: t[0])
                xs = [p[0] for p in pts]
                ys = [p[1]["exact_match_mean"] for p in pts]
                ax.plot(xs, ys, marker="o", label=f"{model}/{split}")
                for x in xs:
                    ax.axvspan(x - 0.4, x + 0.4, color=colors.get(split, "#ccc"), alpha=0.05)
            ax.set_xlabel("difficulty")
            ax.set_ylabel("exact_match")
            ax.set_title(f"acc vs difficulty ({task})")
            ax.legend(fontsize="small")
            _save(fig, path)
        write_csv(rows or [{}], out_dir / f"acc_vs_difficulty_{task}.csv", SUMMARY_COLUMNS)
        paths.append(path)

    # 3. acc_vs_n_bindings.png (Stage B)
    path = out_dir / "acc_vs_n_bindings.png"
    binding_rows = []
    for r in summary:
        diff = json.loads(r["difficulty"])
        if "n_bindings" in diff:
            binding_rows.append({**r, "n_bindings": diff["n_bindings"]})
    if not binding_rows:
        _empty_plot(path, "acc vs n_bindings")
    else:
        fig, ax = plt.subplots()
        lines: dict[tuple, list] = {}
        for r in binding_rows:
            lines.setdefault((r["model"], r["split"]), []).append(r)
        for (model, split), pts in lines.items():
            pts.sort(key=lambda r: r["n_bindings"])
            ax.errorbar(
                [p["n_bindings"] for p in pts],
                [p["exact_match_mean"] for p in pts],
                yerr=[p["exact_match_std"] for p in pts],
                marker="o",
                label=f"{model}/{split}",
            )
        ax.set_xlabel("n_bindings")
        ax.set_ylabel("exact_match")
        ax.set_title("acc vs n_bindings")
        ax.legend(fontsize="small")
        _save(fig, path)
    write_csv(binding_rows or [{}], out_dir / "acc_vs_n_bindings.csv")
    paths.append(path)

    # 4. acc_vs_inference_flops_<task>.png
    all_rows = build_all_runs_rows(usable)
    for task in tasks:
        path = out_dir / f"acc_vs_inference_flops_{task}.png"
        rows = [r for r in all_rows if r["task"] == task and r["inference_flops_per_episode"]]
        if not rows:
            _empty_plot(path, f"acc vs inference FLOPs ({task})")
        else:
            fig, ax = plt.subplots()
            by_model: dict[str, list] = {}
            for r in rows:
                by_model.setdefault(r["model"], []).append(r)
            for model, pts in by_model.items():
                pts.sort(key=lambda r: r["inference_flops_per_episode"])
                ax.plot(
                    [p["inference_flops_per_episode"] for p in pts],
                    [p["exact_match"] for p in pts],
                    marker="o",
                    linestyle="",
                    label=model,
                )
            ax.set_xscale("log")
            ax.set_xlabel("inference FLOPs / episode")
            ax.set_ylabel("exact_match")
            ax.set_title(f"acc vs inference FLOPs ({task})")
            ax.legend(fontsize="small")
            _save(fig, path)
        write_csv(rows or [{}], out_dir / f"acc_vs_inference_flops_{task}.csv", ALL_RUNS_COLUMNS)
        paths.append(path)

    # 5. acc_vs_params_<task>.png
    for task in tasks:
        path = out_dir / f"acc_vs_params_{task}.png"
        rows = [r for r in summary if r["task"] == task and r["params"]]
        if not rows:
            _empty_plot(path, f"acc vs params ({task})")
        else:
            fig, ax = plt.subplots()
            by_model: dict[str, list] = {}
            for r in rows:
                by_model.setdefault(r["model"], []).append(r)
            for model, pts in by_model.items():
                pts.sort(key=lambda r: r["params"])
                ax.plot(
                    [p["params"] for p in pts],
                    [p["exact_match_mean"] for p in pts],
                    marker="o",
                    label=model,
                )
            ax.set_xlabel("params")
            ax.set_ylabel("exact_match")
            ax.set_title(f"acc vs params ({task})")
            ax.legend(fontsize="small")
            _save(fig, path)
        write_csv(rows or [{}], out_dir / f"acc_vs_params_{task}.csv", SUMMARY_COLUMNS)
        paths.append(path)

    # 6. state_norm_vs_iteration_<model>_<task>.png (+ update_norm/cos companions)
    # Grouped by ARM, not by model name: every arm of a4_convergence reports
    # `model: bdh_cq` and differs only in `model.recurrence.kind`, so grouping
    # on the model name writes one file per task with all the update rules
    # pooled into it -- the same mistake, one level up, as the unlabelled
    # series below.
    arm_task_pairs = sorted({(arm_label(rec), rec.results.task) for rec in usable})
    diag_names = ("state_norm", "update_norm", "cos_consecutive")
    if not arm_task_pairs:
        path = out_dir / "state_norm_vs_iteration.png"
        _empty_plot(path, "state_norm vs iteration")
        paths.append(path)
    for arm, task in arm_task_pairs:
        slug = arm.replace("/", "_")
        # Each series is labelled with the cell it came from. Unlabelled series
        # pooled every checkpoint, split, difficulty and R_test into one picture,
        # which is how the A1 write-up first read a converging curve off cells
        # that do not converge (`RESULTS.md` section A1).
        series_by_diag: dict[str, list[tuple[dict, list[float]]]] = {n: [] for n in diag_names}
        for rec in usable:
            if arm_label(rec) != arm or rec.results.task != task:
                continue
            # Per run: a run that stopped short of the sweep's longest run must
            # still contribute its own final checkpoint (see convergence_rows).
            run_final_step = max((ev.step for ev in rec.results.evaluations), default=0)
            for ev in rec.results.evaluations:
                if ev.diagnostics is None:
                    continue
                label = {
                    "seed": rec.results.seed,
                    "config_hash": rec.results.config_hash,
                    "recurrence_kind": recurrence_kind(rec),
                    "step": ev.step,
                    "is_final_step": ev.step == run_final_step,
                    "reasoning_steps": ev.reasoning_steps,
                    "split": ev.split,
                    "difficulty": _difficulty_key(ev.difficulty),
                }
                for name in diag_names:
                    values = getattr(ev.diagnostics, name)
                    if values:
                        series_by_diag[name].append((label, values))
        for name in diag_names:
            fname = f"{name}_vs_iteration_{slug}_{task}.png"
            path = out_dir / fname
            series = series_by_diag[name]
            plotted = [(lab, v) for lab, v in series if lab["is_final_step"]]
            if not plotted:
                _empty_plot(path, f"{name} vs iteration ({arm}/{task})")
            else:
                fig, ax = plt.subplots()
                for i, (_, vals) in enumerate(plotted):
                    ax.plot(range(len(vals)), vals, alpha=0.6, label=None if i else name)
                ax.set_xlabel("iteration")
                ax.set_ylabel(name)
                ax.set_title(f"{name} vs iteration ({arm}/{task}, final checkpoint)")
                _save(fig, path)
            rows = [
                {**label, "iteration": i, name: v}
                for label, vals in series
                for i, v in enumerate(vals)
            ]
            write_csv(rows or [{}], out_dir / f"{name}_vs_iteration_{slug}_{task}.csv")
            paths.append(path)

    # 7. recurrence_convergence_<task>.png: does the latent reach a fixed point?
    for task in tasks:
        path = out_dir / f"recurrence_convergence_{task}.png"
        rows = convergence_rows(usable, task)
        if not rows:
            _empty_plot(path, f"recurrence convergence ({task})")
        else:
            fig, ax = plt.subplots()
            by_model: dict[str, list] = {}
            for r in rows:
                by_model.setdefault(r["model"], []).append(r)
            for name, pts in sorted(by_model.items()):
                merged: dict[int, list[float]] = {}
                for p in pts:
                    merged.setdefault(p["reasoning_steps"], []).append(p["cos_last"])
                xs = sorted(merged)
                ax.plot(xs, [mean(merged[x]) for x in xs], marker="o", label=name)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("R_test")
            ax.set_ylabel("cos(H[R], H[R-1]) at the last iteration")
            ax.set_title(f"recurrence convergence ({task}); 1.0 = fixed point")
            ax.legend(fontsize="small")
            _save(fig, path)
        write_csv(rows or [{}], out_dir / f"recurrence_convergence_{task}.csv", CONVERGENCE_COLUMNS)
        paths.append(path)

    # 8. convergence_onset_<task>.png: WHEN does the loop stop settling?
    #
    # Plotted on the interpolation split alone. `mild` and `strong` belong in
    # the CSV but not in this picture: a split the arm never learns holds
    # cos_last near 1.0 for the whole run for the trivial reason that an
    # unlearned map is a convergent one, and drawing that next to the split
    # that does learn invites exactly the confound this plot exists to expose.
    for task in tasks:
        path = out_dir / f"convergence_onset_{task}.png"
        onset = convergence_onset_rows(usable, task)
        rows = [r for r in onset if r["split"] == ONSET_SPLIT]
        # config_hash is in the key, not just `arm`: a 8000-step pilot and a
        # 40000-step sweep arm are both "bdh_cq/plain", and joining their
        # checkpoints into one line would draw a trajectory no run ever took.
        curves: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
        for r in rows:
            key = (r["arm"], r["config_hash"], r["reasoning_steps"], r["difficulty"])
            curves.setdefault(key, []).append(r)
        # Two points is a segment, not an onset. Only the R values in
        # `evaluation.intermediate_reasoning_steps` get a real trajectory.
        curves = {k: sorted(v, key=lambda r: r["step"]) for k, v in curves.items() if len(v) >= 3}
        # One R per arm -- the deepest one measured mid-training, where the
        # question lives. Every R stays in the CSV.
        deepest: dict[tuple[str, str], int] = {}
        for arm, chash, r, _ in curves:
            deepest[(arm, chash)] = max(deepest.get((arm, chash), 0), r)
        curves = {k: v for k, v in curves.items() if k[2] == deepest[(k[0], k[1])]}
        if not curves:
            _empty_plot(path, f"convergence onset ({task})")
        else:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.0, 8.5))
            shorten = _difficulty_shortener([k[3] for k in curves])
            for (arm, chash, r, difficulty), pts in sorted(curves.items()):
                xs = [p["step"] for p in pts]
                label = f"{arm} {chash[:6]} R={r} {shorten(difficulty)}"
                axes[0].plot(xs, [p["cos_last"] for p in pts], marker="o", ms=3, label=label)
                axes[1].plot(xs, [p["exact_match"] for p in pts], marker="o", ms=3, label=label)
            axes[0].set_ylabel("cos(H[R], H[R-1]) at last iteration")
            axes[0].set_title(f"convergence onset ({task}, {ONSET_SPLIT}); 1.0 = fixed point")
            axes[1].set_ylabel("exact_match")
            axes[1].set_xlabel("training step")
            axes[1].legend(
                fontsize="x-small",
                ncol=2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
            )
            _save(fig, path)
        write_csv(onset or [{}], out_dir / f"convergence_onset_{task}.csv", ONSET_COLUMNS)
        paths.append(path)

    return paths


# -- top level ----------------------------------------------------------------


def aggregate(results_root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    records = walk_runs(results_root)
    all_rows = build_all_runs_rows(records)
    summary = build_summary(records)
    flags = build_flags(records, summary)

    all_runs_path = write_csv(all_rows, out_dir / "all_runs.csv", ALL_RUNS_COLUMNS)
    summary_path = write_csv(summary, out_dir / "summary.csv", SUMMARY_COLUMNS)
    flags_path = write_csv(flags, out_dir / "flags.csv", FLAGS_COLUMNS)
    plot_paths = make_plots(records, summary, out_dir)

    return {
        "all_runs": all_runs_path,
        "summary": summary_path,
        "flags": flags_path,
        "plots": plot_paths,
    }
