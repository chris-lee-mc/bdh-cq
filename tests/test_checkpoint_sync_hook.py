"""The trainer's off-box checkpoint hook (RUNPOD.md section 3, task 21).

Section 3 asks for an upload on every checkpoint, where "every checkpoint"
means `checkpoint_every_steps` OR the 5 minute wall-clock guard, whichever
comes first. That trigger policy already lived in `Trainer.save_checkpoint`;
these tests pin down that the sync hook rides on it rather than reimplementing
its own schedule, and that a broken bucket cannot kill a training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_trainer import TinyModel

from bdhx.seeding import seed_everything
from bdhx.tasks.binding import BindingTask
from bdhx.training.trainer import CHECKPOINT_MAX_INTERVAL_S, Trainer


def make(tiny_cfg, **overrides):
    cfg = tiny_cfg(
        **{
            "training.steps": 6,
            "training.batch_size": 4,
            "training.warmup_steps": 2,
            "training.eval_every_steps": 0,
            "reasoning.train_steps": [1, 2],
            **overrides,
        }
    )
    seed_everything(cfg.training.seed)
    return cfg, TinyModel(), BindingTask()


def test_hook_fires_on_the_step_interval_trigger(tiny_cfg, tmp_path):
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 2})
    seen: list[tuple[int, str]] = []
    Trainer(
        cfg,
        model,
        task,
        tmp_path / "run",
        on_checkpoint=lambda path, step, reason: seen.append((step, reason)),
    ).train()
    assert [s for s, _ in seen] == [2, 4, 6]
    assert {r for _, r in seen} == {"interval"}


def test_hook_receives_a_checkpoint_file_that_exists(tiny_cfg, tmp_path):
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 3})
    paths: list[Path] = []
    Trainer(
        cfg,
        model,
        task,
        tmp_path / "run",
        on_checkpoint=lambda path, step, reason: paths.append(Path(path)),
    ).train()
    assert paths and all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_hook_fires_on_the_wall_clock_trigger_without_a_step_trigger(tiny_cfg, tmp_path):
    """With checkpoint_every_steps disabled, the 5 minute guard is the trigger."""
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 0})
    seen: list[str] = []
    trainer = Trainer(
        cfg,
        model,
        task,
        tmp_path / "run",
        on_checkpoint=lambda path, step, reason: seen.append(reason),
    )
    # Pretend the last checkpoint was longer ago than the guard allows.
    trainer._last_ckpt_time -= CHECKPOINT_MAX_INTERVAL_S + 1
    trainer.train()
    assert "interval" in seen, "no checkpoint fired from the wall-clock trigger"


def test_hook_fires_on_the_final_checkpoint(tiny_cfg, tmp_path):
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 0})
    seen: list[str] = []
    Trainer(
        cfg,
        model,
        task,
        tmp_path / "run",
        on_checkpoint=lambda path, step, reason: seen.append(reason),
    ).train()
    assert "final" in seen


def test_a_failing_sync_is_logged_but_never_ends_the_run(tiny_cfg, tmp_path):
    """The local checkpoint is already on disk; a dead bucket costs an upload,
    not the whole run."""
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 2})
    messages: list[str] = []

    def explode(path, step, reason):
        raise TimeoutError("endpoint unreachable")

    state = Trainer(
        cfg,
        model,
        task,
        tmp_path / "run",
        on_checkpoint=explode,
        log=messages.append,
    ).train()

    assert state.step == 6
    assert state.status == "ok"
    assert any("checkpoint sync failed" in m for m in messages)


def test_no_hook_is_the_default_and_changes_nothing(tiny_cfg, tmp_path):
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 2})
    trainer = Trainer(cfg, model, task, tmp_path / "run")
    assert trainer.on_checkpoint is None
    assert trainer.train().status == "ok"


@pytest.mark.parametrize("reason", ["sigterm", "wall_clock"])
def test_hook_fires_on_preemption_and_wall_clock_checkpoints(tiny_cfg, tmp_path, reason):
    cfg, model, task = make(tiny_cfg, **{"training.checkpoint_every_steps": 0})
    seen: list[str] = []
    trainer = Trainer(
        cfg,
        model,
        task,
        tmp_path / "run",
        on_checkpoint=lambda path, step, r: seen.append(r),
    )
    trainer.save_checkpoint(reason)
    assert seen == [reason]
