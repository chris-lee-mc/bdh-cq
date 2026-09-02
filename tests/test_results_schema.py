from bdhx.results.schema import (
    TRAIN_LOG_COLUMNS,
    EvaluationRow,
    ResultsWriter,
    TrainLogWriter,
)


def test_results_roundtrip(results_dir):
    w = ResultsWriter(
        results_dir,
        run_id="run_abc",
        config_hash="0123456789ab",
        seed=1,
        model="bdh_cq",
        task="compose",
        params=10_023_456,
        solved_width=256,
        gate_extrapolation="hold_last",
    )
    w.add_evaluation(
        {
            "step": 50000,
            "reasoning_steps": 8,
            "split": "strong",
            "difficulty": {"depth": 6},
            "n_episodes": 1000,
            "exact_match": 0.412,
            "token_acc": 0.77,
            "inference_flops_per_episode": 1.2e9,
            "latency_ms_per_episode": 3.1,
            "task_metrics": {"stale_rate": 0.08, "distractor_answer_rate": 0.01, "custom": 0.5},
            "adaptation": {
                "adaptation_latency_ms": 12.4,
                "adaptation_flops": 3.1e7,
                "interference_rate": 0.05,
            },
            "diagnostics": {"state_norm": [1.0, 2.0], "nan_count": 0},
        }
    )
    w.add_evaluation(
        EvaluationRow(
            step=50000,
            reasoning_steps=1,
            split="interp",
            n_episodes=10,
            exact_match=1.0,
            token_acc=1.0,
        )
    )
    w.flush()

    loaded = ResultsWriter.load(results_dir)
    assert loaded == w.results
    assert loaded.evaluations[0].task_metrics.stale_rate == 0.08
    assert loaded.evaluations[0].task_metrics.custom == 0.5
    assert loaded.evaluations[0].adaptation.interference_rate == 0.05
    assert loaded.evaluations[1].adaptation is None
    assert loaded.gate_extrapolation == "hold_last"
    assert loaded.training_curve == "train_log.csv"


def test_train_log_writer(results_dir):
    with TrainLogWriter(results_dir) as log:
        log.log(
            step=1,
            loss=2.0,
            lr=1e-4,
            grad_norm=0.5,
            r_train=2,
            examples_seen=8,
            wall_s=0.1,
            vram=None,
        )
    lines = (results_dir / "train_log.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == list(TRAIN_LOG_COLUMNS)
    assert lines[1].startswith("1,2.0,0.0001,0.5,2,8,0.1,")
