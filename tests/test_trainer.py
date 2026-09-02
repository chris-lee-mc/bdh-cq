"""Trainer checkpoint/resume, NaN policy, and the R_train curriculum."""

from __future__ import annotations

import os
import signal

import pytest
import torch
from torch import nn

from bdhx.models.base import FlopsReport, ReasoningModel, SolveOutput
from bdhx.results.schema import TRAIN_LOG_COLUMNS
from bdhx.seeding import seed_everything
from bdhx.tasks.binding import BindingTask
from bdhx.tasks.vocab import VOCAB_SIZE
from bdhx.training.curriculum import (
    RTrainSampler,
    curriculum_value,
    delayed_start_active,
    progress,
)
from bdhx.training.trainer import MAX_CONSECUTIVE_NANS, Trainer

WIDTH = 8


class TinyModel(ReasoningModel):
    """Smallest model that satisfies the interface; `nan_until` forces NaN losses."""

    requires_serialized = True

    def __init__(self, target_length: int = 1, nan_until: int = 0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, WIDTH)
        self.step_mix = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB_SIZE)
        self.target_length = target_length
        self.nan_until = nan_until
        self.calls = 0

    def reset_context(self) -> None:
        self._prefix = None

    def ingest_context(self, demonstrations) -> None:
        self._prefix = demonstrations

    def solve(self, query, reasoning_steps, collect_diagnostics=False, target_length=None):
        h = self.embed(query if query.dim() == 2 else query.unsqueeze(0)).mean(dim=1)
        diag = {"state_norm": [], "update_norm": [], "cos_consecutive": [], "nan_count": []}
        for _ in range(max(int(reasoning_steps), 1)):
            prev = h
            h = torch.tanh(self.step_mix(h))
            if collect_diagnostics:
                diag["state_norm"].append(float(h.norm()))
                diag["update_norm"].append(float((h - prev).norm()))
                diag["cos_consecutive"].append(1.0)
                diag["nan_count"].append(0)
        logits = self.head(h).unsqueeze(1)
        return SolveOutput(logits.argmax(-1), logits, diag, int(reasoning_steps))

    def forward_episode(self, batch, reasoning_steps):
        self.calls += 1
        h = self.embed(batch.serialized).mean(dim=1)
        for _ in range(max(int(reasoning_steps), 1)):
            h = torch.tanh(self.step_mix(h))
        logits = self.head(h).unsqueeze(1).expand(-1, batch.target.shape[1], -1)
        if self.calls <= self.nan_until:
            logits = logits * float("nan")
        return logits

    def flops_estimate(self, batch, reasoning_steps):
        return FlopsReport(total=0.0, per_episode=0.0, breakdown={})


def make(cfg_loader, **overrides):
    cfg = cfg_loader(
        **{
            "task.name": "binding",
            "training.batch_size": 4,
            "training.warmup_steps": 2,
            "training.eval_every_steps": 0,
            "reasoning.train_steps": [1, 2],
            **overrides,
        }
    )
    seed_everything(cfg.training.seed)
    return cfg, TinyModel(), BindingTask()


# -- checkpoint / resume ---------------------------------------------------
def test_checkpoint_resume(tiny_cfg, tmp_path):
    """A 20-step run interrupted at 10 resumes to bit-identical parameters."""
    cfg, model, task = make(
        tiny_cfg, **{"training.steps": 20, "training.checkpoint_every_steps": 5}
    )
    Trainer(cfg, model, task, tmp_path / "full").train()
    reference = [p.detach().clone() for p in model.parameters()]

    cfg, interrupted, task = make(
        tiny_cfg, **{"training.steps": 20, "training.checkpoint_every_steps": 5}
    )
    run_dir = tmp_path / "resumed"
    trainer = Trainer(cfg, interrupted, task, run_dir)
    state = _interrupt_at(trainer, 10)
    assert state.step == 10
    assert state.status == "preempted"
    assert state.preemptions == 1

    seed_everything(cfg.training.seed)
    resumed = TinyModel()
    trainer2 = Trainer(cfg, resumed, task, run_dir, resume=True)
    assert trainer2.state.step == 10
    assert trainer2.state.resumed_from is not None
    final = trainer2.train()
    assert final.step == 20
    for got, want in zip(resumed.parameters(), reference):
        assert torch.equal(got, want)

    rows = (run_dir / "train_log.csv").read_text().strip().splitlines()
    assert rows[0].split(",") == list(TRAIN_LOG_COLUMNS)
    assert len(rows) == 21  # header plus 20 steps, the resume appends


def _interrupt_at(trainer, step: int):
    """Runs `trainer.train()` with a SIGTERM raised at the end of `step`."""
    original = trainer.train_step

    def wrapped(batch, r_train):
        out = original(batch, r_train)
        if trainer.state.step + 1 == step:
            os.kill(os.getpid(), signal.SIGTERM)
        return out

    trainer.train_step = wrapped
    return trainer.train()


def test_resume_rejects_a_different_config(tiny_cfg, tmp_path):
    cfg, model, task = make(tiny_cfg, **{"training.steps": 2})
    Trainer(cfg, model, task, tmp_path / "run").train()
    other, model2, task2 = make(tiny_cfg, **{"training.steps": 2, "training.lr": 1.0e-3})
    with pytest.raises(ValueError, match="config hash"):
        Trainer(other, model2, task2, tmp_path / "run", resume=True)


# -- NaN policy ------------------------------------------------------------
def test_nan_policy_aborts_after_consecutive_nans(tiny_cfg, tmp_path):
    cfg, _, task = make(tiny_cfg, **{"training.steps": 40})
    model = TinyModel(nan_until=1000)
    state = Trainer(cfg, model, task, tmp_path / "nan").train()
    assert state.status == "diverged"
    assert state.nan_events == MAX_CONSECUTIVE_NANS
    assert state.step == MAX_CONSECUTIVE_NANS
    assert (tmp_path / "nan" / "checkpoints" / "latest.json").exists()


def test_nan_policy_skips_and_recovers(tiny_cfg, tmp_path):
    cfg, _, task = make(tiny_cfg, **{"training.steps": 8})
    model = TinyModel(nan_until=3)
    state = Trainer(cfg, model, task, tmp_path / "nan2").train()
    assert state.status == "ok"
    assert state.nan_events == 3
    assert state.consecutive_nans == 0
    assert state.step == 8


# -- curriculum ------------------------------------------------------------
def test_progress_and_curriculum_schedule(tiny_cfg):
    cfg = tiny_cfg(
        **{
            "training.steps": 100,
            "reasoning.train_step_sampling": "curriculum",
            "reasoning.curriculum": {"schedule": [1, 2, 4, 8], "switch_at": [0.0, 0.25, 0.5, 0.75]},
        }
    )
    assert progress(0, 100) == 0.0 and progress(100, 100) == 1.0
    sampler = RTrainSampler(cfg.reasoning, cfg.training.steps, seed=cfg.training.seed)
    assert [sampler.sample(s) for s in (0, 24, 25, 49, 50, 74, 75, 99)] == [
        1,
        1,
        2,
        2,
        4,
        4,
        8,
        8,
    ]
    assert sampler.r_max == 8
    assert curriculum_value(cfg.reasoning.curriculum, 60, 100) == 4


def test_delayed_start_forces_one_step(tiny_cfg):
    cfg = tiny_cfg(
        **{
            "training.steps": 100,
            "reasoning.train_steps": [4],
            "reasoning.train_step_sampling": "fixed",
            "reasoning.delayed_start_fraction": 0.3,
        }
    )
    sampler = RTrainSampler(cfg.reasoning, cfg.training.steps, seed=1)
    assert delayed_start_active(cfg.reasoning, 29, 100)
    assert not delayed_start_active(cfg.reasoning, 30, 100)
    assert [sampler.sample(s) for s in (0, 29, 30, 99)] == [1, 1, 4, 4]


def test_uniform_sampling_is_a_pure_function_of_step(tiny_cfg):
    cfg = tiny_cfg(**{"training.steps": 50, "reasoning.train_steps": [1, 2, 4]})
    a = RTrainSampler(cfg.reasoning, 50, seed=7)
    b = RTrainSampler(cfg.reasoning, 50, seed=7)
    drawn = [a.sample(s) for s in range(50)]
    assert drawn == [b.sample(s) for s in range(50)]
    assert set(drawn) == {1, 2, 4}


def test_curriculum_sampling_requires_a_curriculum(tiny_cfg):
    cfg = tiny_cfg(**{"reasoning.train_step_sampling": "curriculum", "reasoning.curriculum": None})
    with pytest.raises(ValueError, match="curriculum"):
        RTrainSampler(cfg.reasoning, 10, seed=1)
