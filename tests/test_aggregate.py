from __future__ import annotations

import csv
import json

import pytest

from bdhx.results.aggregate import aggregate, walk_runs
from bdhx.results.schema import ResultsWriter


def _write_run(
    results_root, run_id, *, config_hash, seed, model, task, params, exact_match, dev=False
):
    run_dir = results_root / run_id
    writer = ResultsWriter(
        run_dir,
        run_id=run_id,
        config_hash=config_hash,
        seed=seed,
        model=model,
        task=task,
        params=params,
        status="ok",
    )
    writer.add_evaluation(
        {
            "step": 1000,
            "reasoning_steps": 4,
            "split": "interp",
            "difficulty": {"depth": 2},
            "n_episodes": 32,
            "exact_match": exact_match,
            "token_acc": min(exact_match + 0.1, 1.0),
            "inference_flops_per_episode": 1.0e6,
            "latency_ms_per_episode": 2.0,
            "diagnostics": {
                "state_norm": [1.0, 1.1, 1.2, 1.3],
                "update_norm": [0.3, 0.2, 0.1, 0.05],
                "cos_consecutive": [0.9, 0.92, 0.95, 0.97],
                "nan_count": 0,
            },
        }
    )
    writer.flush()
    metadata = {
        "config": {
            "experiment": {"name": run_id, "stage": "dev", "tags": ["dev"] if dev else []},
            "reasoning": {"train_steps": [1, 2, 4]},
        },
        "train_flops_estimate": 1.0e9,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    return run_dir


@pytest.fixture
def two_group_results(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    # Group A: 3 seeds, params=1,000,000
    for seed, em in zip([1, 2, 3], [0.5, 0.6, 0.55]):
        _write_run(
            root,
            f"runA_s{seed}",
            config_hash="hashA",
            seed=seed,
            model="modelA",
            task="toytask",
            params=1_000_000,
            exact_match=em,
        )
    # Group B: 3 seeds, params=1,100,000 (10% higher -> NOT MATCHED)
    for seed, em in zip([1, 2, 3], [0.4, 0.45, 0.42]):
        _write_run(
            root,
            f"runB_s{seed}",
            config_hash="hashB",
            seed=seed,
            model="modelB",
            task="toytask",
            params=1_100_000,
            exact_match=em,
        )
    return root


def test_walk_runs_finds_all_six(two_group_results):
    records = walk_runs(two_group_results)
    assert len(records) == 6


def test_aggregate_writes_csvs_and_six_plots(two_group_results, tmp_path):
    out_dir = tmp_path / "report"
    result = aggregate(two_group_results, out_dir)

    assert result["all_runs"].exists()
    assert result["summary"].exists()
    assert result["flags"].exists()

    all_runs_rows = result["all_runs"].read_text().strip().splitlines()
    assert len(all_runs_rows) == 1 + 6  # header + one eval row per run

    summary_rows = result["summary"].read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 2  # header + one row per (config_hash, eval key)

    # the six plot categories of FRAMEWORK_SPEC section 10
    png_names = {p.name for p in out_dir.glob("*.png")}
    assert any(n.startswith("acc_vs_reasoning_steps_") for n in png_names)
    assert any(n.startswith("acc_vs_difficulty_") for n in png_names)
    assert "acc_vs_n_bindings.png" in png_names
    assert any(n.startswith("acc_vs_inference_flops_") for n in png_names)
    assert any(n.startswith("acc_vs_params_") for n in png_names)
    assert any(n.startswith("state_norm_vs_iteration_") for n in png_names)
    assert any(n.startswith("update_norm_vs_iteration_") for n in png_names)
    assert any(n.startswith("cos_consecutive_vs_iteration_") for n in png_names)


def test_not_matched_flag_fires_on_10_percent_params_gap(two_group_results, tmp_path):
    out_dir = tmp_path / "report"
    result = aggregate(two_group_results, out_dir)
    flags_text = result["flags"].read_text()
    assert "NOT MATCHED" in flags_text
    assert "hashA" in flags_text and "hashB" in flags_text


def test_provisional_flag_for_fewer_than_5_seeds(two_group_results, tmp_path):
    out_dir = tmp_path / "report"
    result = aggregate(two_group_results, out_dir)
    flags_text = result["flags"].read_text()
    assert "PROVISIONAL" in flags_text  # only 3 seeds per group


def test_summary_stats_and_bootstrap_ci(two_group_results, tmp_path):
    out_dir = tmp_path / "report"
    result = aggregate(two_group_results, out_dir)
    import csv

    with open(result["summary"]) as f:
        rows = list(csv.DictReader(f))
    row_a = next(r for r in rows if r["config_hash"] == "hashA")
    assert row_a["n_seeds"] == "3"
    mean = float(row_a["exact_match_mean"])
    assert abs(mean - (0.5 + 0.6 + 0.55) / 3) < 1e-9
    lo, hi = float(row_a["exact_match_ci95_lo"]), float(row_a["exact_match_ci95_hi"])
    assert lo <= mean <= hi


def test_diverged_run_flagged_and_counted_as_zero(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    for seed, em in zip([1, 2, 3], [0.5, 0.6, 0.55]):
        _write_run(
            root,
            f"run_s{seed}",
            config_hash="hashC",
            seed=seed,
            model="modelC",
            task="toytask2",
            params=500_000,
            exact_match=em,
        )
    # overwrite the third run's status to "diverged"
    diverged_dir = root / "run_s3"
    data = json.loads((diverged_dir / "results.json").read_text())
    data["status"] = "diverged"
    (diverged_dir / "results.json").write_text(json.dumps(data))

    out_dir = tmp_path / "report"
    result = aggregate(root, out_dir)
    assert "DIVERGED" in result["flags"].read_text()

    import csv

    with open(result["summary"]) as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["config_hash"] == "hashC")
    assert row["diverged_k"] == "1"


def test_missing_metadata_excluded_from_plot_data(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    run_dir = _write_run(
        root,
        "run_nometa",
        config_hash="hashD",
        seed=1,
        model="modelD",
        task="toytask3",
        params=200_000,
        exact_match=0.7,
    )
    (run_dir / "metadata.json").unlink()

    out_dir = tmp_path / "report"
    result = aggregate(root, out_dir)
    # still writes tables (all_runs.csv doesn't require metadata) and all plot files
    assert result["all_runs"].exists()
    png_names = {p.name for p in out_dir.glob("*.png")}
    assert any(n.startswith("acc_vs_reasoning_steps_") for n in png_names)


def _write_train_log(run_dir, losses):
    from bdhx.results.schema import TRAIN_LOG_COLUMNS

    lines = [",".join(TRAIN_LOG_COLUMNS)]
    for step, loss in enumerate(losses, start=1):
        row = {c: 0 for c in TRAIN_LOG_COLUMNS}
        row.update(step=step, loss=loss, lr=1e-3, grad_norm=1.0, r_train=1)
        lines.append(",".join(str(row[c]) for c in TRAIN_LOG_COLUMNS))
    (run_dir / "train_log.csv").write_text("\n".join(lines) + "\n")


def test_at_chance_helper_uses_ln_vocab():
    """`AT_CHANCE` fires within 3 percent of ln(vocab_size) and nowhere else."""
    import math

    import bdhx.tasks  # noqa: F401  (registers 'binding')
    from bdhx.results.aggregate import AT_CHANCE_TOL, at_chance, chance_loss
    from bdhx.tasks.vocab import VOCAB_SIZE

    reference = chance_loss("binding")
    assert reference == pytest.approx(math.log(VOCAB_SIZE))
    assert at_chance(reference, "binding")
    assert at_chance(reference * (1 - AT_CHANCE_TOL / 2), "binding")
    assert not at_chance(reference * (1 - 2 * AT_CHANCE_TOL), "binding")
    assert not at_chance(0.5, "binding")
    assert not at_chance(None, "binding")
    assert not at_chance(float("nan"), "binding")


def test_at_chance_flag_fires_for_a_flat_run(tmp_path):
    """A run whose train loss never leaves ln(vocab) is flagged AT_CHANCE."""
    import math

    import bdhx.tasks  # noqa: F401
    from bdhx.tasks.vocab import VOCAB_SIZE

    root = tmp_path / "results"
    root.mkdir()
    chance = math.log(VOCAB_SIZE)
    flat = _write_run(
        root,
        "flat",
        config_hash="hashFlat",
        seed=1,
        model="looped_transformer",
        task="binding",
        params=1_000_000,
        exact_match=0.0,
    )
    _write_train_log(flat, [chance + 0.4, chance + 0.1, chance - 0.01])
    learned = _write_run(
        root,
        "learned",
        config_hash="hashLearned",
        seed=1,
        model="transformer",
        task="binding",
        params=1_000_000,
        exact_match=1.0,
    )
    _write_train_log(learned, [chance, 2.0, 0.05])

    out = aggregate(root, tmp_path / "reports")
    rows = list(csv.DictReader(out["flags"].open()))
    at_chance_rows = [r for r in rows if r["flag"] == "AT_CHANCE"]
    assert [r["config_hash"] for r in at_chance_rows] == ["hashFlat"]
    assert "ln(vocab)" in at_chance_rows[0]["detail"]
