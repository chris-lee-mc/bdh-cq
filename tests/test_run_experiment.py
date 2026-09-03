"""End-to-end smoke: tools/run_experiment.py writes a valid results.json.

One subprocess per registered model that imports, on the tiny_smoke config with
a shortened step budget. `model.params_target` stays at the config's 200_000:
the smallest width the solvers accept for a 4128-token vocabulary already costs
about 70k parameters for the Transformer family and 135k for BDH, so a 50_000
target fails the width solver's hard tolerance before any training starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from bdhx.config import PROJECT_ROOT
from bdhx.results.schema import ResultsWriter
from bdhx.training.evaluate import EVAL_SPLITS

MODELS = ["bdh", "bdh_cq", "looped_transformer", "transformer", "gated_deltanet"]
RECURRENT = {"bdh_cq", "looped_transformer"}
REASONING_STEPS = [1, 2, 4]
STEPS = 10
DEPTH = 2  # configs/base/default.yaml model.depth


def run_experiment(model: str, out_dir, data_root) -> tuple[str, dict]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_experiment.py"),
        "--config",
        "configs/base/tiny_smoke.yaml",
        "--run-id",
        model,
        "--out",
        str(out_dir),
        "--sync-bucket",
        "s3://example/stub",
        "--overrides",
        f"model.name={model}",
        f"training.steps={STEPS}",
        "training.eval_every_steps=5",
        "training.checkpoint_every_steps=5",
        "task.n_eval_episodes=16",
        f"evaluation.reasoning_steps={REASONING_STEPS}",
    ]
    env = {"BDHX_DATA_ROOT": str(data_root), "OMP_NUM_THREADS": "2"}
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, **env},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, json.loads((out_dir / model / "metadata.json").read_text())


@pytest.mark.parametrize("model", MODELS)
def test_run_experiment_smoke(model, tmp_path):
    if model.startswith("bdh"):
        pytest.importorskip("bdh_cq")
    out_dir = tmp_path / "results"
    stdout, metadata = run_experiment(model, out_dir, tmp_path / "data")
    run_dir = out_dir / model
    assert "stub, nothing uploaded" in stdout
    assert metadata["steps_completed"] == STEPS
    assert metadata["param_total"] > 0 and metadata["config_hash"]
    assert metadata["examples_seen"] == STEPS * 8

    results = ResultsWriter.load(run_dir)
    assert results.status == "ok"
    assert results.model == model and results.task == "compose"
    assert results.params and results.solved_width

    final = [row for row in results.evaluations if row.step == STEPS]
    assert {row.split for row in final} == set(EVAL_SPLITS)
    assert {row.reasoning_steps for row in final} == set(REASONING_STEPS)
    assert {row.step for row in results.evaluations} == {5, 10}  # intermediate curve

    for row in final:
        assert row.n_episodes > 0
        assert 0.0 <= row.exact_match <= 1.0 and 0.0 <= row.token_acc <= 1.0
        assert row.inference_flops_per_episode > 0
        assert row.latency_ms_per_episode >= 0.0
        assert row.adaptation is None
        diagnostics = row.diagnostics
        assert diagnostics is not None
        # length R for the recurrent models (one entry per reasoning step, even
        # though a step applies `depth` layers); fixed-depth baselines run their
        # stack once per episode, so their series is one entry per block.
        expected = row.reasoning_steps if model in RECURRENT else DEPTH
        for key in ("state_norm", "update_norm", "cos_consecutive", "active_neuron_frac"):
            assert len(getattr(diagnostics, key)) == expected, key
        assert len(diagnostics.activation_percentiles) == expected
        assert all(len(p) == 5 for p in diagnostics.activation_percentiles)
        assert diagnostics.nan_count == 0

    log = (run_dir / "train_log.csv").read_text().strip().splitlines()
    assert len(log) == STEPS + 1
    assert (run_dir / "log.txt").exists()
    assert list((run_dir / "checkpoints").glob("step_*.pt"))


def test_jacobian_only_at_the_documented_depths(tmp_path):
    """`jacobian_eigenvalue_estimate` is filled at R in {8, 32}, empty elsewhere."""
    from bdhx.config import load_config
    from bdhx.training.evaluate import run_evaluation
    from bdhx.training.trainer import build_model, build_task

    cfg = load_config(
        PROJECT_ROOT / "configs" / "base" / "tiny_smoke.yaml",
        {
            "model.name": "looped_transformer",
            "evaluation.reasoning_steps": [4, 8],
            "task.n_eval_episodes": 8,
        },
    )
    task = build_task(cfg)
    model = build_model(cfg, task, 1)
    rows = run_evaluation(model, task, cfg, 0, splits=("interp",))
    by_depth = {row.reasoning_steps: row for row in rows}
    assert by_depth[4].diagnostics.jacobian_eigenvalue_estimate == []
    estimate = by_depth[8].diagnostics.jacobian_eigenvalue_estimate
    assert len(estimate) == 1 and estimate[0] >= 0.0
    assert len(by_depth[8].diagnostics.state_norm) == 8
