"""Training loop (FRAMEWORK_SPEC section 7).

AdamW, cosine schedule with warmup, gradient clipping, per-batch R_train
sampling, online episode sampling, checkpoint/resume with every rng state, a
SIGTERM handler for spot preemption, the NaN policy and the wall-clock guard.

Notes:

- bf16 autocast is enabled only on cuda; on cpu the loop runs fp32 (torch cpu
  autocast is bf16-emulated and would make `test_checkpoint_resume`
  device-dependent), so `training.precision` is a no-op on cpu.
- Batches are a pure function of `(training.seed, step, worker)`, and so is
  R_train, which is what makes resume bit-exact without persisting a data
  iterator.
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from inspect import signature
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bdhx.config import config_hash as compute_config_hash
from bdhx.registry import get_model, get_task
from bdhx.results.schema import TrainLogWriter
from bdhx.seeding import episode_id as make_episode_id
from bdhx.seeding import get_rng_states, set_rng_states, task_rng
from bdhx.tasks.base import Episode, pad_and_batch
from bdhx.training.curriculum import RTrainSampler
from bdhx.training.diagnostics import GradNormTracker

MAX_CONSECUTIVE_NANS = 20
CHECKPOINT_MAX_INTERVAL_S = 300.0
KEEP_CHECKPOINTS = 2


# -- construction helpers --------------------------------------------------
def resolve_device(cfg) -> torch.device:
    name = cfg.compute.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def apply_compute_settings(cfg) -> torch.device:
    if cfg.compute.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    return resolve_device(cfg)


def build_task(cfg):
    return get_task(cfg.task.name)()


def max_target_length(cfg, task) -> int:
    """Longest target over one sample of every train and eval difficulty."""
    difficulties = list(cfg.task.train_difficulties or task.train_difficulties())
    eval_difficulties = cfg.task.eval_difficulties or task.eval_difficulties()
    for group in eval_difficulties.values():
        difficulties.extend(group)
    longest = 1
    for i, difficulty in enumerate(difficulties):
        ep = task.sample(task_rng(cfg.task.seed, "probe", i), difficulty)
        longest = max(longest, int(ep.target.numel()))
    return longest


def build_model(cfg, task, target_length: int | None = None):
    ctor = get_model(cfg.model.name)
    tl = target_length if target_length is not None else max_target_length(cfg, task)
    kwargs: dict[str, Any] = {}
    params = signature(ctor).parameters
    accepts_loss = "loss" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
    if accepts_loss:
        kwargs["loss"] = cfg.training.loss
    elif cfg.training.loss != "final_answer":
        raise ValueError(f"model '{cfg.model.name}' does not implement loss '{cfg.training.loss}'")
    return ctor(cfg.model, task.vocab_size, target_length=tl, **kwargs)


def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    if cfg.training.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer '{cfg.training.optimizer}'")
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )


def lr_multiplier(step: int, cfg) -> float:
    """Linear warmup then the configured decay (cosine to zero, or constant)."""
    warmup = max(int(cfg.training.warmup_steps), 0)
    total = max(int(cfg.training.steps), 1)
    if warmup and step < warmup:
        return (step + 1) / warmup
    if cfg.training.schedule == "constant":
        return 1.0
    if cfg.training.schedule != "cosine":
        raise ValueError(f"unsupported schedule '{cfg.training.schedule}'")
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def build_scheduler(optimizer, cfg) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, cfg))


def episode_loss(model, batch, reasoning_steps: int) -> torch.Tensor:
    """`model.episode_loss` when the model defines one, else CE on the target tokens."""
    fn = getattr(model, "episode_loss", None)
    if callable(fn):
        return fn(batch, reasoning_steps)
    logits = model.forward_episode(batch, reasoning_steps)
    target = batch.target.to(logits.device)
    mask = batch.target_mask.to(logits.device)
    labels = target.masked_fill(~mask, -1)
    return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-1)


# -- state -----------------------------------------------------------------
@dataclass
class TrainState:
    step: int = 0
    examples_seen: int = 0
    tokens_seen: int = 0
    nan_events: int = 0
    consecutive_nans: int = 0
    preemptions: int = 0
    status: str = "ok"
    wall_clock_train_s: float = 0.0
    train_flops_estimate: float = 0.0
    peak_vram_bytes: int = 0
    resumed_from: str | None = None
    checkpoint_path: str | None = None
    grad_norm: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """One training run: `Trainer(cfg, model, task, run_dir).train()`."""

    def __init__(
        self,
        cfg,
        model,
        task,
        run_dir: str | Path,
        *,
        device: torch.device | None = None,
        eval_fn: Callable[[int], None] | None = None,
        log: Callable[[str], None] | None = None,
        resume: bool = False,
        config_hash: str | None = None,
        worker: int = 0,
    ):
        self.cfg = cfg
        self.model = model
        self.task = task
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or resolve_device(cfg)
        self.eval_fn = eval_fn
        self.log = log or (lambda msg: None)
        self.worker = int(worker)
        self.config_hash = config_hash or compute_config_hash(cfg)
        self.model.to(self.device)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.sampler = RTrainSampler(cfg.reasoning, cfg.training.steps, seed=cfg.training.seed)
        self.grad_norms = GradNormTracker()
        self.state = TrainState()
        self._stop = False
        self._last_ckpt_time = time.perf_counter()
        self._last_ckpt_step = -1
        if resume:
            path = self.latest_checkpoint()
            if path is None:
                raise FileNotFoundError(f"--resume: no checkpoint under {self.ckpt_dir}")
            self.load_checkpoint(path)

    # -- data --------------------------------------------------------------
    def train_difficulties(self) -> list[dict]:
        return list(self.cfg.task.train_difficulties or self.task.train_difficulties())

    def sample_batch(self, step: int):
        """Online sampling; a pure function of (training.seed, step, worker)."""
        difficulties = self.train_difficulties()
        rng = task_rng(self.cfg.training.seed, f"train:{self.worker}", int(step))
        batch_size = int(self.cfg.training.batch_size)
        episodes: list[Episode] = []
        for i in range(batch_size):
            index = step * batch_size + i
            ep = self.task.sample(rng, difficulties[index % len(difficulties)])
            episodes.append(
                Episode(
                    demonstrations=ep.demonstrations,
                    query=ep.query,
                    target=ep.target,
                    difficulty=ep.difficulty,
                    split="train",
                    episode_id=make_episode_id(self.cfg.training.seed, "train", index),
                    extras=ep.extras,
                )
            )
        return pad_and_batch(episodes, self.task).to(self.device)

    def autocast(self):
        if self.device.type == "cuda" and self.cfg.training.precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    # -- one step ----------------------------------------------------------
    def train_step(self, batch, r_train: int) -> tuple[float | None, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            loss = episode_loss(self.model, batch, r_train)
        loss = loss.float()
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return None, float("nan")
        loss.backward()
        params = [p for p in self.model.parameters() if p.grad is not None]
        clip = float(self.cfg.training.grad_clip or 0.0)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else math.inf))
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return None, grad_norm
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach()), grad_norm

    # -- checkpointing -----------------------------------------------------
    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "step": self.state.step,
            "config_hash": self.config_hash,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng_states": get_rng_states(),
            "state": self.state.as_dict(),
        }

    def save_checkpoint(self, reason: str = "interval") -> Path:
        path = self.ckpt_dir / f"step_{self.state.step:08d}.pt"
        torch.save(self.checkpoint_payload(), path)
        (self.ckpt_dir / "latest.json").write_text(
            json.dumps({"path": str(path), "step": self.state.step, "reason": reason}, indent=2)
        )
        self.state.checkpoint_path = str(path)
        self._last_ckpt_time = time.perf_counter()
        self._last_ckpt_step = self.state.step
        self._prune_checkpoints()
        self.log(f"checkpoint step={self.state.step} reason={reason} path={path}")
        return path

    def _prune_checkpoints(self) -> None:
        files = sorted(self.ckpt_dir.glob("step_*.pt"))
        for old in files[:-KEEP_CHECKPOINTS]:
            old.unlink(missing_ok=True)

    def latest_checkpoint(self) -> Path | None:
        pointer = self.ckpt_dir / "latest.json"
        if pointer.exists():
            path = Path(json.loads(pointer.read_text())["path"])
            if path.exists():
                return path
        files = sorted(self.ckpt_dir.glob("step_*.pt"))
        return files[-1] if files else None

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("config_hash") != self.config_hash:
            raise ValueError(
                f"--resume: checkpoint config hash {payload.get('config_hash')} != "
                f"{self.config_hash}; refusing to resume a different experiment"
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        set_rng_states(payload["rng_states"])
        saved = dict(payload.get("state") or {})
        saved.pop("grad_norm", None)
        for key, value in saved.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self.state.step = int(payload["step"])
        self.state.consecutive_nans = 0
        self.state.status = "ok"
        self.state.resumed_from = str(path)
        self.log(f"resumed from {path} at step {self.state.step}")

    # -- signals -----------------------------------------------------------
    def _install_sigterm(self):
        def handler(signum, _frame):
            self._stop = True
            self.state.preemptions += 1
            self.log(f"signal {signum} received; checkpointing after the current step")

        try:
            previous = signal.signal(signal.SIGTERM, handler)
        except ValueError:  # not the main thread
            return None
        return previous

    def _restore_sigterm(self, previous) -> None:
        if previous is not None:
            try:
                signal.signal(signal.SIGTERM, previous)
            except ValueError:
                pass

    # -- loop --------------------------------------------------------------
    def train(self) -> TrainState:
        cfg = self.cfg
        total = int(cfg.training.steps)
        deadline = float(cfg.compute.max_wall_clock_minutes) * 60.0
        previous = self._install_sigterm()
        started = time.perf_counter()
        base_wall = self.state.wall_clock_train_s  # a resumed run keeps its earlier time
        resuming = self.state.step > 0 and (self.run_dir / "train_log.csv").exists()
        log_writer = _open_train_log(self.run_dir, append=resuming)
        try:
            while self.state.step < total:
                step = self.state.step
                r_train = self.sampler.sample(step)
                batch = self.sample_batch(step)
                loss, grad_norm = self.train_step(batch, r_train)
                self.state.examples_seen += len(batch)
                self.state.tokens_seen += int(batch.serialized_mask.sum())
                self.state.train_flops_estimate += (
                    3.0 * self.model.flops_estimate(batch, r_train).per_episode * len(batch)
                )
                diverged = False
                if loss is None:
                    self.state.nan_events += 1
                    self.state.consecutive_nans += 1
                    self.log(f"nan loss at step {step} ({self.state.consecutive_nans} in a row)")
                    diverged = self.state.consecutive_nans >= MAX_CONSECUTIVE_NANS
                else:
                    self.state.consecutive_nans = 0
                    self.grad_norms.update(grad_norm)
                self.state.step = step + 1
                self.state.wall_clock_train_s = base_wall + time.perf_counter() - started
                log_writer.log(
                    step=self.state.step,
                    loss=loss if loss is not None else float("nan"),
                    lr=self.optimizer.param_groups[0]["lr"],
                    grad_norm=grad_norm,
                    r_train=r_train,
                    examples_seen=self.state.examples_seen,
                    wall_s=round(self.state.wall_clock_train_s, 3),
                    vram=self._vram(),
                )
                log_writer.flush()

                if diverged:
                    self.state.status = "diverged"
                    self.save_checkpoint("diverged")
                    self.log(f"diverged after {MAX_CONSECUTIVE_NANS} consecutive nan steps")
                    break
                if self._stop:
                    self.state.status = "preempted"
                    self.save_checkpoint("sigterm")
                    break
                if deadline and self.state.wall_clock_train_s >= deadline:
                    self.state.status = "incomplete"
                    self.save_checkpoint("wall_clock")
                    self.log(f"wall clock limit reached at step {self.state.step}")
                    break
                every = int(cfg.training.checkpoint_every_steps or 0)
                due = every and self.state.step % every == 0
                if due or time.perf_counter() - self._last_ckpt_time >= CHECKPOINT_MAX_INTERVAL_S:
                    self.save_checkpoint("interval")
                eval_every = int(cfg.training.eval_every_steps or 0)
                due_eval = eval_every and self.state.step % eval_every == 0
                if self.eval_fn and due_eval and self.state.step < total:
                    self.eval_fn(self.state.step)
        finally:
            log_writer.close()
            self._restore_sigterm(previous)
            self.state.wall_clock_train_s = base_wall + time.perf_counter() - started
            self.state.grad_norm = self.grad_norms.summary()
            self.state.peak_vram_bytes = self._vram()
        done = self.state.status == "ok" and self.state.step >= total
        if done and self._last_ckpt_step != self.state.step:
            self.save_checkpoint("final")
        return self.state

    def _vram(self) -> int:
        return int(torch.cuda.max_memory_allocated()) if self.device.type == "cuda" else 0


def _open_train_log(run_dir: Path, append: bool = False) -> TrainLogWriter:
    """`TrainLogWriter`, in append mode after a resume so the curve is not truncated."""
    if not append:
        return TrainLogWriter(run_dir)
    import csv

    from bdhx.results.schema import TRAIN_LOG_COLUMNS

    writer = TrainLogWriter.__new__(TrainLogWriter)
    writer.run_dir = Path(run_dir)
    writer.path = writer.run_dir / "train_log.csv"
    writer._fh = writer.path.open("a", newline="")
    writer._writer = csv.DictWriter(writer._fh, fieldnames=list(TRAIN_LOG_COLUMNS))
    return writer
