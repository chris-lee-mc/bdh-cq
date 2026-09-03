"""Guards for the Gate A diagnosis (EXPERIMENT_PLAN section 10).

Three defects made the Stage A dev sweep flat at chance and none of them was
visible in the results schema:

- the tied unembedding was initialized at `nn.Embedding`'s default N(0, 1), so
  the initial cross-entropy was tens of nats instead of ln(vocab_size);
- `model.depth` was ignored by the looped models, leaving one layer per
  reasoning step;
- nothing checked that a run had left the chance plateau at all.

These tests pin the first two. The third is `tools/sanity_learnability.py`
plus the `AT_CHANCE` flag (`tests/test_aggregate.py`). The oracle test below
pins the loss/target alignment those diagnoses had to rule out first.
"""

from __future__ import annotations

import math

import pytest
import torch

import bdhx.models
import bdhx.tasks  # noqa: F401  (registers the tasks)
from bdhx.config import load_config
from bdhx.models.base import FlopsReport, ReasoningModel
from bdhx.registry import list_models
from bdhx.tasks.vocab import ANSWER, MAP, QUERY, SEP
from bdhx.training.evaluate import run_evaluation
from bdhx.training.trainer import Trainer, build_model, build_task, episode_loss

INIT_LOSS_TOL = 0.5  # nats around ln(vocab_size)
PARAMS_TARGET = 350_000


def _cfg(model_name: str, **overrides):
    base = {
        "model.name": model_name,
        "model.params_target": PARAMS_TARGET,
        "task.name": "binding",
        "task.train_difficulties": [{"n_bindings": 1}, {"n_bindings": 2}],
        "training.batch_size": 8,
        "compute.device": "cpu",
        "compute.deterministic": False,
    }
    base.update(overrides)
    return load_config("configs/base/tiny_smoke.yaml", base)


def _batch(cfg, model, task, tmp_path):
    return Trainer(cfg, model, task, tmp_path).sample_batch(0)


@pytest.mark.parametrize("model_name", list_models())
def test_init_loss_is_near_ln_vocab(model_name, tmp_path):
    """No model may start training away from the uniform-prediction loss.

    A tied head over an N(0, 1) embedding put the looped Transformer at 219
    nats at step 1; the run then spent its whole budget walking back to chance.
    """
    if model_name.startswith("bdh") or model_name == "unified_block":
        pytest.importorskip("bdh_cq")
    cfg = _cfg(model_name)
    task = build_task(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, task, target_length=1)
    batch = _batch(cfg, model, task, tmp_path)
    with torch.no_grad():
        loss = float(episode_loss(model, batch, 1))
    assert loss == pytest.approx(math.log(task.vocab_size), abs=INIT_LOSS_TOL)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_looped_depth_counts_layers_inside_the_shared_block(depth, tmp_path):
    """`model.depth` for a looped model is layers per step, not distinct blocks."""
    from bdhx.models.base import BlockCounter

    cfg = _cfg("looped_transformer", **{"model.depth": depth})
    task = build_task(cfg)
    model = build_model(cfg, task, target_length=1)
    assert len(model.block.layers) == depth
    with BlockCounter(list(model.block.layers)) as counter:
        out = model.solve(torch.tensor([[100]]), 3)
    assert counter.count == 3 * depth
    assert out.block_applications == 3 * depth


def test_looped_prelude_and_coda_run_once_around_the_loop():
    """Optional prelude/coda layers are applied once each, outside the loop."""
    from bdhx.models.base import BlockCounter
    from bdhx.models.looped_transformer import LoopedTransformer

    cfg = _cfg("looped_transformer", **{"model.depth": 2})
    task = build_task(cfg)
    model = LoopedTransformer(cfg.model, task.vocab_size, target_length=1, prelude=1, coda=1)
    with BlockCounter(list(model.prelude) + list(model.coda)) as counter:
        out = model.solve(torch.tensor([[100]]), 4)
    assert counter.count == 2
    assert out.block_applications == 4 * 2 + 2


class OracleCopy(ReasoningModel):
    """Reads the value that follows the query key in the serialized context.

    Not a learned model: it is the reference the evaluation path must score at
    exact match 1.0. If it does not, the target alignment or the loss masking
    is wrong and no training result means anything.
    """

    requires_serialized = True

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.scale = torch.nn.Parameter(torch.tensor(10.0))

    def reset_context(self) -> None: ...

    def ingest_context(self, demonstrations) -> None: ...

    def solve(self, query, reasoning_steps, collect_diagnostics=False, target_length=None):
        raise NotImplementedError("oracle scores through forward_episode only")

    @staticmethod
    def _answer(tokens: list[int]) -> list[int]:
        qi, ai = tokens.index(QUERY), tokens.index(ANSWER)
        key = tokens[qi + 1 : ai]
        chunk: list[int] = []
        for token in tokens[1:qi] + [SEP]:
            if token == SEP:
                if MAP in chunk:
                    m = chunk.index(MAP)
                    if chunk[:m] == key:
                        return chunk[m + 1 :]
                chunk = []
            else:
                chunk.append(token)
        return []

    def forward_episode(self, batch, reasoning_steps):
        lt = batch.target.shape[1]
        logits = torch.zeros(len(batch), lt, self.vocab_size)
        for i in range(len(batch)):
            row = [int(t) for t in batch.serialized[i][batch.serialized_mask[i]].tolist()]
            answer = self._answer(row)
            for j in range(lt):
                if j < len(answer):
                    logits[i, j, answer[j]] = 1.0
        return logits * self.scale

    def flops_estimate(self, batch, reasoning_steps) -> FlopsReport:
        return FlopsReport(total=0.0, per_episode=1.0)


def test_oracle_solver_scores_exact_match_one(tmp_path):
    """The evaluation path scores a known-correct solver at 1.0 on every split."""
    cfg = _cfg(
        "transformer",
        **{
            "task.n_eval_episodes": 32,
            "evaluation.reasoning_steps": [1, 2],
            "evaluation.diagnostics": False,
        },
    )
    task = build_task(cfg)
    model = OracleCopy(task.vocab_size)
    rows = run_evaluation(model, task, cfg, step=0, collect_diagnostics=False)
    assert rows
    for row in rows:
        assert row.exact_match == 1.0, (row.split, row.difficulty, row.reasoning_steps)
        assert row.token_acc == 1.0


def test_oracle_loss_is_the_margin_not_chance(tmp_path):
    """`training.loss: final_answer` scores the oracle at its logit margin.

    Pins that the loss is taken at the target positions: a correct answer with
    a margin of 10 gives ln(1 + (V - 1) e^-10), not ln(V).
    """
    cfg = _cfg("transformer")
    task = build_task(cfg)
    model = OracleCopy(task.vocab_size)
    batch = _batch(cfg, model, task, tmp_path)
    with torch.no_grad():
        loss = float(episode_loss(model, batch, 1))
    expected = math.log(1.0 + (task.vocab_size - 1) * math.exp(-10.0))
    assert loss == pytest.approx(expected, abs=1e-3)


def test_training_batches_are_fresh_and_reproducible(tmp_path):
    """Each step draws new episodes, and the query key is always in the demos."""
    cfg = _cfg("transformer")
    task = build_task(cfg)
    model = OracleCopy(task.vocab_size)
    trainer = Trainer(cfg, model, task, tmp_path)
    first, second = trainer.sample_batch(0), trainer.sample_batch(1)
    assert not torch.equal(first.serialized, second.serialized)
    assert torch.equal(trainer.sample_batch(0).serialized, first.serialized)
    for i in range(len(first)):
        row = [int(t) for t in first.serialized[i][first.serialized_mask[i]].tolist()]
        target = [int(t) for t in first.target[i][first.target_mask[i]].tolist()]
        assert OracleCopy._answer(row) == target
        assert int(first.answer_start[i]) == row.index(ANSWER) + 1
