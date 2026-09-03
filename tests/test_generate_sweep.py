from __future__ import annotations

import sys

import pytest
import yaml

from bdhx.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from generate_sweep import (
    expand_sweep,
    main,
    total_estimated_gpu_hours,
    write_jobs,
)

A1 = PROJECT_ROOT / "configs" / "stage_a" / "a1_first_experiment.yaml"
C1 = PROJECT_ROOT / "configs" / "stage_c" / "c1_recurrence_engineering.yaml"


def test_a1_expands_to_18_configs():
    jobs, meta = expand_sweep(A1)
    assert len(jobs) == 18  # 3 models x 2 tasks x 3 seeds
    assert meta["name"] == "a1_first_experiment"
    # The hash is seed-invariant (config.HASH_EXCLUDED_FIELDS): one hash per
    # sweep arm, and (hash, seed) identifies a job.
    assert len({j.hash for j in jobs}) == 6  # 3 models x 2 tasks
    assert len({(j.hash, j.seed) for j in jobs}) == 18


def test_seed_policy_refuses_below_3_without_dev(tmp_path):
    sweep = yaml.safe_load(A1.read_text())
    sweep["seeds"] = [1]
    p = tmp_path / "one_seed.yaml"
    p.write_text(yaml.safe_dump(sweep))

    with pytest.raises(ValueError, match="seed"):
        expand_sweep(p)

    jobs, _ = expand_sweep(p, dev=True)
    assert len(jobs) == 6  # 3 models x 2 tasks x 1 seed


def test_seed_policy_dev_flag_in_sweep_file(tmp_path):
    sweep = yaml.safe_load(A1.read_text())
    sweep["seeds"] = [1]
    sweep["dev"] = True
    p = tmp_path / "dev_sweep.yaml"
    p.write_text(yaml.safe_dump(sweep))
    jobs, _ = expand_sweep(p)  # no --dev needed; the file itself opts out
    assert len(jobs) == 6


def test_matched_controls_add_one_per_parameter_adding_kind():
    jobs, _ = expand_sweep(C1, matched_controls=True)
    base_jobs = [j for j in jobs if j.control_of is None]
    control_jobs = [j for j in jobs if j.control_of is not None]

    # grid: 7 kinds x 2 tasks x 3 seeds
    assert len(base_jobs) == 42
    # plain, residual add 0 params; the other 5 kinds each get exactly one control
    assert len(control_jobs) == 5
    assert {j.control_of for j in control_jobs} == {
        "step_gate",
        "init_skip",
        "attn_residual",
        "step_emb",
        "adapter",
    }
    for job in control_jobs:
        assert job.cfg.model.recurrence.kind == "plain"
        assert job.cfg.model.params_target > 10_000_000


def test_matched_controls_noop_without_flag():
    jobs, _ = expand_sweep(C1)
    assert all(j.control_of is None for j in jobs)
    assert len(jobs) == 42


def test_large_sweep_gate(tmp_path):
    jobs, _ = expand_sweep(A1)
    estimates = {j.hash: 1000.0 for j in jobs}  # 18000 minutes = 300 GPU-hours
    est_path = tmp_path / "estimates.json"
    import json

    est_path.write_text(json.dumps(estimates))

    with pytest.raises(SystemExit):
        main([str(A1), "--estimates", str(est_path), "--out", str(tmp_path / "out")])

    # allowed through with --allow-large-sweep
    rc = main(
        [
            str(A1),
            "--estimates",
            str(est_path),
            "--allow-large-sweep",
            "--out",
            str(tmp_path / "out2"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out2" / "manifest.csv").exists()


def test_total_estimated_gpu_hours_ignores_missing():
    jobs, _ = expand_sweep(A1)
    assert total_estimated_gpu_hours(jobs) == 0.0


def test_write_jobs_writes_configs_and_manifest(tmp_path):
    jobs, _ = expand_sweep(A1)
    manifest = write_jobs(jobs, tmp_path)
    assert manifest.exists()
    rows = manifest.read_text().strip().splitlines()
    assert len(rows) == 1 + len(jobs)  # header + one row per job
    yamls = sorted(tmp_path.glob("exp_*.yaml"))
    assert len(yamls) == len(jobs)
